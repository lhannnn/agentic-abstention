"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import re
import time
from typing import Any, Literal, Optional

import openai
from pydantic import BaseModel, ValidationError

from recipe.action_space import (
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_TOOL_CALL,
    parse_action_space_response,
)
from recipe.multiturn_alcuna_prompts import (
    build_alcuna_multiturn_follow_up_prompt,
    build_alcuna_response_generator_prompt,
    build_alcuna_retrievability_checker_prompt,
    extract_original_question_from_multiturn_prompt,
)
from recipe.utils import permissive_makedirs

SEARCH_QUERY_PATTERN = re.compile(
    r"^(?:[-*]\s*|\d+\.\s*)?Search\s+(.+)$",
    flags=re.IGNORECASE,
)
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple[Any, ...] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as exc:
                raise exc

    return wrapper


class RetrievabilityCheckerResult(BaseModel):
    retrievability: Literal["KNOWN_RETRIEVABLE", "UNKNOWN_OR_UNGROUNDED"]
    explanation: str


class SearchTurnTrace(BaseModel):
    turn_index: int
    agent_raw_response: str
    parsed_action: str
    search_query: Optional[str] = None
    checker_raw_response: Optional[str] = None
    checker_parsed: Optional[RetrievabilityCheckerResult] = None
    environment_response: Optional[str] = None


class EpisodeTrace(BaseModel):
    request_ordinal: int
    original_question: str
    initial_agent_prompt: str
    turns: list[SearchTurnTrace]
    termination_reason: str
    num_turns: int
    final_action: str
    final_response: str


def extract_search_query_from_response(response_text: str) -> Optional[str]:
    for line in response_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        match = SEARCH_QUERY_PATTERN.match(stripped_line)
        if match:
            search_query = match.group(1).strip()
            if search_query:
                return search_query
    return None


def extract_search_query(response_text: str) -> Optional[str]:
    return extract_search_query_from_response(response_text)


def _fallback_checker_result() -> RetrievabilityCheckerResult:
    return RetrievabilityCheckerResult(
        retrievability="UNKNOWN_OR_UNGROUNDED",
        explanation="checker_output_parse_failure",
    )


def parse_retrievability_checker_response(
    response_text: str,
) -> RetrievabilityCheckerResult:
    try:
        return RetrievabilityCheckerResult.model_validate_json(response_text)
    except (ValidationError, ValueError):
        pass

    json_match = JSON_BLOCK_PATTERN.search(response_text)
    if json_match is not None:
        try:
            return RetrievabilityCheckerResult.model_validate_json(
                json_match.group(0)
            )
        except (ValidationError, ValueError):
            pass

    return _fallback_checker_result()


class ALCUNAMultiTurnSearchTogetherModel:
    def __init__(
        self,
        save_dir: str,
        agent_model_name: str,
        checker_model_name: Optional[str] = None,
        response_generator_model_name: Optional[str] = None,
        max_turns: int = 10,
        agent_max_tokens: Optional[int] = 4000,
        checker_max_tokens: Optional[int] = 512,
        response_generator_max_tokens: Optional[int] = 1024,
        temperature: float = 0.8,
        checker_temperature: float = 0.0,
        response_generator_temperature: float = 0.2,
        seed: Optional[int] = None,
        request_timeout_seconds: int = 180,
        client: Any = None,
    ):
        if client is None:
            if "TOGETHER_API_KEY" not in os.environ:
                raise ValueError(
                    "TOGETHER_API_KEY environment variable must be set when using Together API."
                )
            api_key = os.environ["TOGETHER_API_KEY"]
            client = openai.OpenAI(
                api_key=api_key,
                base_url=os.environ.get(
                    "TOGETHER_BASE_URL", "https://api.together.xyz/v1"
                ),
                timeout=request_timeout_seconds,
            )

        self.client = client
        self.save_dir = save_dir
        self.agent_model_name = agent_model_name
        self.checker_model_name = checker_model_name or agent_model_name
        self.response_generator_model_name = (
            response_generator_model_name or agent_model_name
        )
        self.max_turns = max_turns
        self.agent_max_tokens = agent_max_tokens
        self.checker_max_tokens = checker_max_tokens
        self.response_generator_max_tokens = response_generator_max_tokens
        self.temperature = temperature
        self.checker_temperature = checker_temperature
        self.response_generator_temperature = response_generator_temperature
        self.seed = seed
        self.request_timeout_seconds = request_timeout_seconds
        self.request_ordinal = 0

        permissive_makedirs(self.save_dir)
        self.trace_path = os.path.join(self.save_dir, "MultiTurnEpisodeTraces.jsonl")

    @retry_with_exponential_backoff
    def _chat_completion(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        request_kwargs = {
            "model": model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self.seed is not None:
            request_kwargs["seed"] = self.seed

        completion = self.client.chat.completions.create(**request_kwargs)
        content = completion.choices[0].message.content
        return content if content is not None else ""

    def respond(self, questions: list[str]) -> list[str]:
        if isinstance(questions, str):
            questions = [questions]

        responses = []
        for initial_agent_prompt in questions:
            request_ordinal = self.request_ordinal
            self.request_ordinal += 1

            final_response, episode_trace = self._run_episode(
                initial_agent_prompt=initial_agent_prompt,
                request_ordinal=request_ordinal,
            )
            self._append_episode_trace(episode_trace)
            responses.append(final_response)

        return responses

    def _run_episode(
        self,
        initial_agent_prompt: str,
        request_ordinal: int,
    ) -> tuple[str, EpisodeTrace]:
        original_question = extract_original_question_from_multiturn_prompt(
            initial_agent_prompt
        )
        messages = [{"role": "user", "content": initial_agent_prompt}]
        turns: list[SearchTurnTrace] = []
        termination_reason = "invalid_action"
        final_action = ACTION_INVALID
        final_response = ""

        for turn_index in range(1, self.max_turns + 1):
            agent_raw_response = self._chat_completion(
                model_name=self.agent_model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.agent_max_tokens,
            )
            parsed_agent_response = parse_action_space_response(agent_raw_response)
            parsed_action = parsed_agent_response.predicted_action

            if parsed_action in (ACTION_ANSWER, ACTION_ABSTAIN):
                turns.append(
                    SearchTurnTrace(
                        turn_index=turn_index,
                        agent_raw_response=agent_raw_response,
                        parsed_action=parsed_action,
                    )
                )
                termination_reason = parsed_action.lower()
                final_action = parsed_action
                final_response = agent_raw_response
                break

            if parsed_action != ACTION_TOOL_CALL:
                turns.append(
                    SearchTurnTrace(
                        turn_index=turn_index,
                        agent_raw_response=agent_raw_response,
                        parsed_action=ACTION_INVALID,
                    )
                )
                termination_reason = "invalid_action"
                final_action = ACTION_INVALID
                final_response = agent_raw_response
                break

            search_query = extract_search_query_from_response(agent_raw_response)
            if search_query is None:
                turns.append(
                    SearchTurnTrace(
                        turn_index=turn_index,
                        agent_raw_response=agent_raw_response,
                        parsed_action=ACTION_TOOL_CALL,
                    )
                )
                termination_reason = "invalid_search_query"
                final_action = ACTION_INVALID
                final_response = (
                    "INVALID_SEARCH_QUERY: agent selected SEARCH but did not provide "
                    "a parseable query."
                )
                break

            if turn_index == self.max_turns:
                turns.append(
                    SearchTurnTrace(
                        turn_index=turn_index,
                        agent_raw_response=agent_raw_response,
                        parsed_action=ACTION_TOOL_CALL,
                        search_query=search_query,
                    )
                )
                termination_reason = "max_turns_reached"
                final_action = ACTION_INVALID
                final_response = (
                    f"MAX_TURNS_REACHED: agent still requested SEARCH at turn {self.max_turns}."
                )
                break

            checker_raw_response = self._run_retrievability_check(
                original_question=original_question,
                search_query=search_query,
            )
            checker_parsed = parse_retrievability_checker_response(
                checker_raw_response
            )
            environment_response = self._run_response_generator(
                original_question=original_question,
                search_query=search_query,
                checker_result=checker_parsed,
            )
            turns.append(
                SearchTurnTrace(
                    turn_index=turn_index,
                    agent_raw_response=agent_raw_response,
                    parsed_action=ACTION_TOOL_CALL,
                    search_query=search_query,
                    checker_raw_response=checker_raw_response,
                    checker_parsed=checker_parsed,
                    environment_response=environment_response,
                )
            )

            messages.append({"role": "assistant", "content": agent_raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": build_alcuna_multiturn_follow_up_prompt(
                        original_question=original_question,
                        search_query=search_query,
                        environment_response=environment_response,
                    ),
                }
            )

        episode_trace = EpisodeTrace(
            request_ordinal=request_ordinal,
            original_question=original_question,
            initial_agent_prompt=initial_agent_prompt,
            turns=turns,
            termination_reason=termination_reason,
            num_turns=len(turns),
            final_action=final_action,
            final_response=final_response,
        )
        return final_response, episode_trace

    def _run_retrievability_check(
        self,
        original_question: str,
        search_query: str,
    ) -> str:
        prompt = build_alcuna_retrievability_checker_prompt(
            original_question=original_question,
            search_query=search_query,
        )
        return self._chat_completion(
            model_name=self.checker_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.checker_temperature,
            max_tokens=self.checker_max_tokens,
        )

    def _run_response_generator(
        self,
        original_question: str,
        search_query: str,
        checker_result: RetrievabilityCheckerResult,
    ) -> str:
        prompt = build_alcuna_response_generator_prompt(
            original_question=original_question,
            search_query=search_query,
            checker_result_json=checker_result.model_dump_json(indent=2),
        )
        return self._chat_completion(
            model_name=self.response_generator_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.response_generator_temperature,
            max_tokens=self.response_generator_max_tokens,
        )

    def _append_episode_trace(self, episode_trace: EpisodeTrace) -> None:
        with open(self.trace_path, "a") as trace_file:
            trace_file.write(episode_trace.model_dump_json())
            trace_file.write("\n")
