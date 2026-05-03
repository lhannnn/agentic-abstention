"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
import re
from typing import Optional

ACTION_SPACE_PROMPT_TEMPLATE = """I will ask you a question. You must choose exactly one action from the following action space.

1. ANSWER: provide the final answer directly based on the current information.
2. ABSTAIN: stop the current problem-solving process because the available information is insufficient, unreliable, or fundamentally non-resolvable, and neither further reasoning nor external information acquisition is appropriate.
3. TOOL_CALL: obtain additional information through an external tool, retrieval step, or environment interaction.

Output format:
Line 1: exactly one of [ANSWER], [ABSTAIN], [TOOL_CALL]
Line 2:
- if [ANSWER], provide the final answer only
- if [ABSTAIN], provide a brief reason for abstaining
- if [TOOL_CALL], provide a brief description of what external information or tool is needed

Question: {question}"""

ACTION_SPACE_SEARCH_PROMPT_TEMPLATE = """I will ask you a question. You must choose exactly one action from the following action space.

1. ANSWER: provide the final answer directly based on the current information.
2. ABSTAIN: stop the current problem-solving process because the available information is insufficient, unreliable, or fundamentally non-resolvable, and neither further reasoning nor external information acquisition is appropriate.
3. SEARCH: obtain additional information by performing an external search.

Output format:
Line 1: exactly one of [ANSWER], [ABSTAIN], [SEARCH]

Line 2:
- if [ANSWER], provide the final answer only
- if [ABSTAIN], provide a brief reason for abstaining
- if [SEARCH], give the keywords or question to search in the format: Search [keywords or question]

Question: {question}"""

ACTION_ANSWER = "ANSWER"
ACTION_ABSTAIN = "ABSTAIN"
ACTION_TOOL_CALL = "TOOL_CALL"
ACTION_INVALID = "INVALID"

ACTION_HEADER_TO_LABEL = {
    "[ANSWER]": ACTION_ANSWER,
    "[ABSTAIN]": ACTION_ABSTAIN,
    "[TOOL_CALL]": ACTION_TOOL_CALL,
    "[SEARCH]": ACTION_TOOL_CALL,
}

VALID_ACTIONS = tuple(dict.fromkeys(ACTION_HEADER_TO_LABEL.values()))
ACTION_KEYWORD_TO_LABEL = {
    ACTION_ANSWER: ACTION_ANSWER,
    ACTION_ABSTAIN: ACTION_ABSTAIN,
    ACTION_TOOL_CALL: ACTION_TOOL_CALL,
    "SEARCH": ACTION_TOOL_CALL,
}

ACTION_ALIASES = tuple(ACTION_KEYWORD_TO_LABEL)
ACTION_LINE_1_PATTERN = re.compile(
    r"Line\s*1\s*:\s*(?:\[\s*(ANSWER|ABSTAIN|TOOL_CALL|SEARCH)\s*\]|(?:\d+\.\s*)?(ANSWER|ABSTAIN|TOOL_CALL|SEARCH)\b)"
)
ACTION_BRACKET_AT_START_PATTERN = re.compile(
    r"^\[(ANSWER|ABSTAIN|TOOL_CALL|SEARCH)\](?:\s|$)"
)
ACTION_LEADING_PATTERN = re.compile(
    r"^(?:\d+\.\s*)?(ANSWER|ABSTAIN|TOOL_CALL|SEARCH)\b"
)
ACTION_BRACKET_ANYWHERE_PATTERN = re.compile(
    r"\[(ANSWER|ABSTAIN|TOOL_CALL|SEARCH)\]"
)


@dataclass(frozen=True)
class ActionSpaceParseResult:
    predicted_action: str
    is_valid: bool
    error: Optional[str] = None


def build_action_space_prompt(question: str) -> str:
    return ACTION_SPACE_PROMPT_TEMPLATE.format(question=question)


def build_action_space_search_prompt(question: str) -> str:
    return ACTION_SPACE_SEARCH_PROMPT_TEMPLATE.format(question=question)


def _map_action_alias(action_alias: str) -> str:
    return ACTION_KEYWORD_TO_LABEL[action_alias]


def parse_action_space_response(response_text: Optional[str]) -> ActionSpaceParseResult:
    if response_text is None:
        return ActionSpaceParseResult(
            predicted_action=ACTION_INVALID,
            is_valid=False,
            error="Response is None.",
        )

    stripped_lines = [
        line.strip() for line in response_text.splitlines() if line.strip()
    ]
    if not stripped_lines:
        return ActionSpaceParseResult(
            predicted_action=ACTION_INVALID,
            is_valid=False,
            error="Response does not contain a non-empty action header.",
        )

    for stripped_line in reversed(stripped_lines):
        line_1_match = ACTION_LINE_1_PATTERN.search(stripped_line)
        if line_1_match:
            action_alias = line_1_match.group(1) or line_1_match.group(2)
            return ActionSpaceParseResult(
                predicted_action=_map_action_alias(action_alias),
                is_valid=True,
            )

    for stripped_line in reversed(stripped_lines):
        if stripped_line in ACTION_HEADER_TO_LABEL:
            return ActionSpaceParseResult(
                predicted_action=ACTION_HEADER_TO_LABEL[stripped_line],
                is_valid=True,
            )

        bracket_at_start_match = ACTION_BRACKET_AT_START_PATTERN.match(stripped_line)
        if bracket_at_start_match:
            return ActionSpaceParseResult(
                predicted_action=_map_action_alias(bracket_at_start_match.group(1)),
                is_valid=True,
            )

        bracket_matches = ACTION_BRACKET_ANYWHERE_PATTERN.findall(stripped_line)
        if bracket_matches:
            return ActionSpaceParseResult(
                predicted_action=_map_action_alias(bracket_matches[-1]),
                is_valid=True,
            )

        leading_match = ACTION_LEADING_PATTERN.match(stripped_line)
        if leading_match:
            return ActionSpaceParseResult(
                predicted_action=_map_action_alias(leading_match.group(1)),
                is_valid=True,
            )

    return ActionSpaceParseResult(
        predicted_action=ACTION_INVALID,
        is_valid=False,
        error=(
            "Response does not contain a recognizable action label."
        ),
    )


def gold_action_from_should_abstain(
    should_abstain: Optional[bool],
) -> Optional[str]:
    if should_abstain is None:
        return None
    if should_abstain:
        return ACTION_ABSTAIN
    return ACTION_ANSWER
