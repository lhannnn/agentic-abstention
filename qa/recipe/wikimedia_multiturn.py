"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any, Callable, Optional
import unicodedata

import openai
from pydantic import BaseModel

from recipe.action_space import (
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_TOOL_CALL,
)
from recipe.utils import permissive_makedirs
from recipe.wikimedia_json_actions import parse_wikimedia_json_response
from recipe.wikimedia_prompts import (
    build_wikimedia_multiturn_follow_up_prompt,
    extract_original_question_from_multiturn_prompt,
    format_retrieved_docs_for_prompt,
)
from recipe.wikimedia_retriever import RetrievedDocument, WikimediaDenseRetriever


OPENAI_REQUEST_JSON_INVALID_ERROR = "We could not parse the JSON body of your request"
OPENAI_INVALID_PROMPT_ERROR = (
    "Invalid prompt: your prompt was flagged as potentially violating our usage policy"
)
OPENAI_OUTPUT_LIMIT_ERROR = (
    "Could not finish the message because max_tokens or model output limit was reached"
)
OPENAI_RETRY_MAX_COMPLETION_TOKENS = 8000
OPENAI_REQUEST_JSON_RETRY_ATTEMPTS = 3


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple[Any, ...] = (
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APIConnectionError,
        openai.APITimeoutError,
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


class SearchTurnTrace(BaseModel):
    turn_index: int
    agent_raw_response: str
    parsed_action: str
    search_query: Optional[str] = None
    search_call_index: Optional[int] = None
    retrieved_docs: Optional[list[RetrievedDocument]] = None
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
    final_success: Optional[bool] = None


def _sanitize_openai_text(text: str) -> str:
    sanitized_characters = []
    for character in text:
        if character in ("\n", "\r", "\t"):
            sanitized_characters.append(character)
            continue

        codepoint = ord(character)
        if unicodedata.category(character) == "Cs":
            sanitized_characters.append("\uFFFD")
            continue
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            sanitized_characters.append("\uFFFD")
            continue

        sanitized_characters.append(character)
    return "".join(sanitized_characters)


def _sanitize_openai_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized_messages = []
    for message in messages:
        sanitized_message = dict(message)
        content = sanitized_message.get("content")
        if isinstance(content, str):
            sanitized_message["content"] = _sanitize_openai_text(content)
        sanitized_messages.append(sanitized_message)
    return sanitized_messages


def _serialize_openai_request_payload(request_kwargs: dict[str, Any]) -> bytes:
    serialized_payload = json.dumps(
        request_kwargs,
        ensure_ascii=False,
        allow_nan=False,
    )
    return serialized_payload.encode("utf-8", "strict")


def _validate_openai_request_payload(request_kwargs: dict[str, Any]) -> None:
    _serialize_openai_request_payload(request_kwargs)


def _build_openai_payload_metadata(
    request_kwargs: dict[str, Any],
    payload_bytes: Optional[bytes] = None,
) -> dict[str, Any]:
    payload_bytes = (
        _serialize_openai_request_payload(request_kwargs)
        if payload_bytes is None
        else payload_bytes
    )
    return {
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_byte_length": len(payload_bytes),
    }


def _serialize_retrieved_docs_for_debug(
    retrieved_docs: Optional[list[RetrievedDocument]],
) -> Optional[list[dict[str, Any]]]:
    if retrieved_docs is None:
        return None
    return [document.model_dump(mode="json") for document in retrieved_docs]


class WikimediaMultiTurnSearchTogetherModel:
    def __init__(
        self,
        save_dir: str,
        index_root: str,
        agent_model_name: str,
        encoder_name: str = "intfloat/e5-base-v2",
        top_k: int = 3,
        max_search_calls: int = 10,
        agent_max_tokens: Optional[int] = 4000,
        temperature: float = 0.8,
        seed: Optional[int] = None,
        request_timeout_seconds: int = 180,
        device: Optional[str] = None,
        client: Any = None,
        retriever: Optional[WikimediaDenseRetriever] = None,
    ):
        if client is None:
            if "TOGETHER_API_KEY" not in os.environ:
                raise ValueError(
                    "TOGETHER_API_KEY environment variable must be set when using Together API."
                )
            client = openai.OpenAI(
                api_key=os.environ["TOGETHER_API_KEY"],
                base_url=os.environ.get(
                    "TOGETHER_BASE_URL", "https://api.together.xyz/v1"
                ),
                timeout=request_timeout_seconds,
            )

        self.client = client
        self.save_dir = save_dir
        self.agent_model_name = agent_model_name
        self.top_k = top_k
        self.max_search_calls = max_search_calls
        self.agent_max_tokens = agent_max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_timeout_seconds = request_timeout_seconds
        self.request_ordinal = 0
        self.retriever = retriever or WikimediaDenseRetriever(
            index_root=index_root,
            encoder_name=encoder_name,
            top_k=top_k,
            device=device,
        )

        permissive_makedirs(self.save_dir)
        self.trace_path = os.path.join(self.save_dir, "MultiTurnEpisodeTraces.jsonl")
        run_name = os.path.basename(os.path.dirname(self.save_dir))
        self.dataset_name = run_name.split("_Wikimedia", 1)[0] if run_name else None

    @retry_with_exponential_backoff
    def _chat_completion(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        debug_context: Optional[dict[str, Any]] = None,
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
        search_calls_used = 0

        while True:
            turn_index = len(turns) + 1
            debug_context = {
                "dataset_name": self.dataset_name,
                "request_ordinal": request_ordinal,
                "turn_index": turn_index,
                "original_question": original_question,
            }
            if turns:
                last_turn = turns[-1]
                debug_context["agent_raw_response"] = last_turn.agent_raw_response
                debug_context["search_query"] = last_turn.search_query
                debug_context["retrieved_docs"] = last_turn.retrieved_docs
            agent_raw_response = self._chat_completion(
                model_name=self.agent_model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.agent_max_tokens,
                debug_context=debug_context,
            )
            parsed_agent_response = parse_wikimedia_json_response(agent_raw_response)
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

            search_query = parsed_agent_response.search_query
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

            search_calls_used += 1
            if search_calls_used >= self.max_search_calls:
                turns.append(
                    SearchTurnTrace(
                        turn_index=turn_index,
                        agent_raw_response=agent_raw_response,
                        parsed_action=ACTION_TOOL_CALL,
                        search_query=search_query,
                        search_call_index=search_calls_used,
                    )
                )
                termination_reason = "max_search_calls_reached"
                final_action = ACTION_TOOL_CALL
                final_response = agent_raw_response
                break

            retrieved_docs = self.retriever.search(search_query, top_k=self.top_k)
            environment_response = format_retrieved_docs_for_prompt(retrieved_docs)
            turns.append(
                SearchTurnTrace(
                    turn_index=turn_index,
                    agent_raw_response=agent_raw_response,
                    parsed_action=ACTION_TOOL_CALL,
                    search_query=search_query,
                    search_call_index=search_calls_used,
                    retrieved_docs=retrieved_docs,
                    environment_response=environment_response,
                )
            )

            messages.append({"role": "assistant", "content": agent_raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": build_wikimedia_multiturn_follow_up_prompt(
                        original_question=original_question,
                        search_query=search_query,
                        retrieved_docs_text=environment_response,
                        search_call_index=search_calls_used,
                        top_k=self.top_k,
                        max_search_calls=self.max_search_calls,
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

    def _append_episode_trace(self, episode_trace: EpisodeTrace) -> None:
        with open(self.trace_path, "a") as trace_file:
            trace_file.write(episode_trace.model_dump_json())
            trace_file.write("\n")


class WikimediaMultiTurnSearchOpenAIModel(WikimediaMultiTurnSearchTogetherModel):
    def __init__(
        self,
        save_dir: str,
        index_root: str,
        openai_model_name: str = "gpt-5.4-mini",
        reasoning_effort: Optional[str] = "medium",
        encoder_name: str = "intfloat/e5-base-v2",
        top_k: int = 3,
        max_search_calls: int = 10,
        agent_max_tokens: Optional[int] = 4000,
        # gpt-5.4-mini currently only accepts the default temperature value.
        temperature: float = 1.0,
        seed: Optional[int] = None,
        request_timeout_seconds: int = 180,
        device: Optional[str] = None,
        client: Any = None,
        client_factory: Optional[Callable[[], Any]] = None,
        retriever: Optional[WikimediaDenseRetriever] = None,
    ):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.request_timeout_seconds = request_timeout_seconds
        if client is None:
            if client_factory is None:
                client_factory = self._default_client_factory
            client = client_factory()

        if client_factory is None:
            client_factory = lambda: client

        self._client_factory = client_factory

        self.reasoning_effort = reasoning_effort
        super().__init__(
            save_dir=save_dir,
            index_root=index_root,
            agent_model_name=openai_model_name,
            encoder_name=encoder_name,
            top_k=top_k,
            max_search_calls=max_search_calls,
            agent_max_tokens=agent_max_tokens,
            temperature=temperature,
            seed=seed,
            request_timeout_seconds=request_timeout_seconds,
            device=device,
            client=client,
            retriever=retriever,
        )

    def _default_client_factory(self) -> Any:
        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set when using OpenAI API."
            )
        return openai.OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            timeout=self.request_timeout_seconds,
        )

    def _debug_provider_name(self) -> str:
        return "openai"

    def _debug_provider_base_url(self) -> str:
        return self.openai_base_url

    def _build_openai_request_kwargs(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": _sanitize_openai_messages(messages),
        }
        if max_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        return request_kwargs

    def _append_openai_debug_failure(
        self,
        *,
        error_kind: str,
        request_kwargs: dict[str, Any],
        error_message: str,
        debug_context: Optional[dict[str, Any]],
        attempt_index: Optional[int] = None,
        fresh_client: Optional[bool] = None,
        transport_mode: Optional[str] = None,
        payload_bytes: Optional[bytes] = None,
    ) -> None:
        messages = request_kwargs.get("messages", [])
        last_user_message = None
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                last_user_message = message["content"]
                break

        try:
            payload_metadata = _build_openai_payload_metadata(
                request_kwargs=request_kwargs,
                payload_bytes=payload_bytes,
            )
        except (TypeError, ValueError, UnicodeError):
            payload_metadata = {
                "payload_sha256": None,
                "payload_byte_length": None,
            }
        debug_record = {
            "error_kind": error_kind,
            "error_message": error_message,
            "dataset_name": None if debug_context is None else debug_context.get("dataset_name"),
            "request_ordinal": None if debug_context is None else debug_context.get("request_ordinal"),
            "turn_index": None if debug_context is None else debug_context.get("turn_index"),
            "original_question": None if debug_context is None else debug_context.get("original_question"),
            "agent_raw_response": None if debug_context is None else debug_context.get("agent_raw_response"),
            "search_query": None if debug_context is None else debug_context.get("search_query"),
            "retrieved_docs": _serialize_retrieved_docs_for_debug(
                None if debug_context is None else debug_context.get("retrieved_docs")
            ),
            "request_kwargs": request_kwargs,
            "provider_name": self._debug_provider_name(),
            "provider_base_url": self._debug_provider_base_url(),
            "model": request_kwargs.get("model"),
            "max_completion_tokens": request_kwargs.get("max_completion_tokens"),
            "temperature": request_kwargs.get("temperature"),
            "seed": request_kwargs.get("seed"),
            "reasoning_effort": request_kwargs.get("reasoning_effort"),
            "extra_body": request_kwargs.get("extra_body"),
            "message_count": len(messages),
            "message_char_lengths": [
                len(message.get("content", ""))
                for message in messages
                if isinstance(message.get("content"), str)
            ],
            "messages": messages,
            "last_user_message": last_user_message,
            "last_message_preview": (
                messages[-1]["content"][:500]
                if messages and isinstance(messages[-1].get("content"), str)
                else None
            ),
            "attempt_index": attempt_index,
            "fresh_client": fresh_client,
            "transport_mode": transport_mode,
            **payload_metadata,
        }
        debug_filename = (
            "WikimediaPolicyBlockedFailures.jsonl"
            if error_kind == "provider_policy_blocked"
            else "WikimediaOpenAIDebugFailures.jsonl"
        )
        debug_path = os.path.join(self.save_dir, debug_filename)
        with open(debug_path, "a") as debug_file:
            debug_file.write(json.dumps(debug_record, ensure_ascii=False))
            debug_file.write("\n")

    def _is_output_limit_error(self, error: openai.BadRequestError) -> bool:
        return OPENAI_OUTPUT_LIMIT_ERROR in str(error)

    def _is_request_json_invalid_error(self, error: openai.BadRequestError) -> bool:
        return OPENAI_REQUEST_JSON_INVALID_ERROR in str(error)

    def _is_invalid_prompt_error(self, error: openai.BadRequestError) -> bool:
        return OPENAI_INVALID_PROMPT_ERROR in str(error)

    def _execute_openai_request_with_request_json_retry(
        self,
        *,
        request_kwargs: dict[str, Any],
        debug_context: Optional[dict[str, Any]],
        payload_bytes: bytes,
    ):
        last_request_json_error: Optional[openai.BadRequestError] = None

        for attempt_index in range(1, OPENAI_REQUEST_JSON_RETRY_ATTEMPTS + 1):
            fresh_client = attempt_index > 1
            current_client = self.client if not fresh_client else self._client_factory()
            try:
                completion = current_client.chat.completions.create(**request_kwargs)
                if fresh_client:
                    self.client = current_client
                return completion
            except openai.BadRequestError as exc:
                if self._is_invalid_prompt_error(exc):
                    self._append_openai_debug_failure(
                        error_kind="provider_policy_blocked",
                        request_kwargs=request_kwargs,
                        error_message=str(exc),
                        debug_context=debug_context,
                        attempt_index=attempt_index,
                        fresh_client=fresh_client,
                        transport_mode="sdk",
                        payload_bytes=payload_bytes,
                    )
                    raise
                if self._is_request_json_invalid_error(exc):
                    last_request_json_error = exc
                    self._append_openai_debug_failure(
                        error_kind="request_json_invalid",
                        request_kwargs=request_kwargs,
                        error_message=str(exc),
                        debug_context=debug_context,
                        attempt_index=attempt_index,
                        fresh_client=fresh_client,
                        transport_mode="sdk",
                        payload_bytes=payload_bytes,
                    )
                    if attempt_index < OPENAI_REQUEST_JSON_RETRY_ATTEMPTS:
                        continue
                raise

        if last_request_json_error is not None:
            raise last_request_json_error
        raise RuntimeError("OpenAI request retry loop exited without completion.")

    @retry_with_exponential_backoff
    def _chat_completion(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        debug_context: Optional[dict[str, Any]] = None,
    ) -> str:
        request_kwargs = self._build_openai_request_kwargs(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            payload_bytes = _serialize_openai_request_payload(request_kwargs)
        except (TypeError, ValueError, UnicodeError) as exc:
            self._append_openai_debug_failure(
                error_kind="request_json_prevalidation_failed",
                request_kwargs=request_kwargs,
                error_message=str(exc),
                debug_context=debug_context,
                attempt_index=0,
                fresh_client=False,
                transport_mode="sdk",
            )
            raise RuntimeError(
                "OpenAI request payload could not be serialized to valid JSON."
            ) from exc

        try:
            completion = self._execute_openai_request_with_request_json_retry(
                request_kwargs=request_kwargs,
                debug_context=debug_context,
                payload_bytes=payload_bytes,
            )
        except openai.BadRequestError as exc:
            if (
                self._is_output_limit_error(exc)
                and max_tokens is not None
                and max_tokens < OPENAI_RETRY_MAX_COMPLETION_TOKENS
            ):
                retry_request_kwargs = self._build_openai_request_kwargs(
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=OPENAI_RETRY_MAX_COMPLETION_TOKENS,
                )
                retry_payload_bytes = _serialize_openai_request_payload(
                    retry_request_kwargs
                )
                completion = self._execute_openai_request_with_request_json_retry(
                    request_kwargs=retry_request_kwargs,
                    debug_context=debug_context,
                    payload_bytes=retry_payload_bytes,
                )
            else:
                raise

        content = completion.choices[0].message.content
        return content if content is not None else ""


class WikimediaMultiTurnSearchOpenRouterModel(WikimediaMultiTurnSearchOpenAIModel):
    def __init__(
        self,
        save_dir: str,
        index_root: str,
        openrouter_model_name: Optional[str],
        reasoning_effort: Optional[str] = None,
        encoder_name: str = "intfloat/e5-base-v2",
        top_k: int = 3,
        max_search_calls: int = 10,
        agent_max_tokens: Optional[int] = 4000,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        request_timeout_seconds: int = 180,
        device: Optional[str] = None,
        client: Any = None,
        client_factory: Optional[Callable[[], Any]] = None,
        retriever: Optional[WikimediaDenseRetriever] = None,
    ):
        if not openrouter_model_name:
            raise ValueError(
                "openrouter_model_name must be set when using OpenRouter Wikimedia multi-turn."
            )

        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.openrouter_http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        self.openrouter_title = os.environ.get("OPENROUTER_TITLE")

        super().__init__(
            save_dir=save_dir,
            index_root=index_root,
            openai_model_name=openrouter_model_name,
            reasoning_effort=reasoning_effort,
            encoder_name=encoder_name,
            top_k=top_k,
            max_search_calls=max_search_calls,
            agent_max_tokens=agent_max_tokens,
            temperature=temperature,
            seed=seed,
            request_timeout_seconds=request_timeout_seconds,
            device=device,
            client=client,
            client_factory=client_factory,
            retriever=retriever,
        )

    def _default_client_factory(self) -> Any:
        if self.openrouter_api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable must be set when using OpenRouter."
            )

        default_headers = {}
        if self.openrouter_http_referer:
            default_headers["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_title:
            default_headers["X-OpenRouter-Title"] = self.openrouter_title

        client_kwargs = {
            "api_key": self.openrouter_api_key,
            "base_url": self.openrouter_base_url,
            "timeout": self.request_timeout_seconds,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        return openai.OpenAI(**client_kwargs)

    def _debug_provider_name(self) -> str:
        return "openrouter"

    def _debug_provider_base_url(self) -> str:
        return self.openrouter_base_url

    def _build_openai_request_kwargs(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": _sanitize_openai_messages(messages),
        }
        if max_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.reasoning_effort is not None:
            request_kwargs["extra_body"] = {
                "reasoning": {"effort": self.reasoning_effort}
            }
        return request_kwargs
