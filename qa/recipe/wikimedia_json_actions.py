"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from recipe.action_space import (
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_TOOL_CALL,
)


SEARCH_QUERY_PREFIX = "Search "
ACTION_TO_REQUIRED_FIELD = {
    ACTION_ANSWER: "answer",
    ACTION_ABSTAIN: "reason",
    "SEARCH": "search_query",
}


@dataclass(frozen=True)
class WikimediaJsonActionParseResult:
    predicted_action: str
    is_valid: bool
    error: Optional[str] = None
    answer: Optional[str] = None
    reason: Optional[str] = None
    search_query: Optional[str] = None


def _invalid(error: str) -> WikimediaJsonActionParseResult:
    return WikimediaJsonActionParseResult(
        predicted_action=ACTION_INVALID,
        is_valid=False,
        error=error,
    )


def _parse_single_response_payload(
    response_payload: object,
) -> WikimediaJsonActionParseResult:
    if not isinstance(response_payload, dict):
        return _invalid("Response JSON must be an object.")

    action = response_payload.get("action")
    if action not in ACTION_TO_REQUIRED_FIELD:
        return _invalid(
            "Response JSON must contain action equal to ANSWER, ABSTAIN, or SEARCH."
        )

    required_field = ACTION_TO_REQUIRED_FIELD[action]
    allowed_keys = {"action", required_field}
    if set(response_payload) != allowed_keys:
        return _invalid(
            f"Response JSON for action {action} must contain exactly keys: "
            f"{sorted(allowed_keys)}."
        )

    field_value = response_payload.get(required_field)
    if not isinstance(field_value, str) or not field_value.strip():
        return _invalid(
            f"Response JSON field {required_field} must be a non-empty string."
        )

    normalized_value = field_value.strip()
    if action == ACTION_ANSWER:
        return WikimediaJsonActionParseResult(
            predicted_action=ACTION_ANSWER,
            is_valid=True,
            answer=normalized_value,
        )

    if action == ACTION_ABSTAIN:
        return WikimediaJsonActionParseResult(
            predicted_action=ACTION_ABSTAIN,
            is_valid=True,
            reason=normalized_value,
        )

    if not normalized_value.startswith(SEARCH_QUERY_PREFIX):
        normalized_value = f"{SEARCH_QUERY_PREFIX}{normalized_value}"

    search_query = normalized_value[len(SEARCH_QUERY_PREFIX) :].strip()
    if not search_query:
        return _invalid("SEARCH search_query must contain keywords after 'Search '.")

    return WikimediaJsonActionParseResult(
        predicted_action=ACTION_TOOL_CALL,
        is_valid=True,
        search_query=search_query,
    )


def _decode_json_object_sequence(response_text: str) -> Optional[list[object]]:
    decoder = json.JSONDecoder()
    decoded_payloads: list[object] = []
    index = 0
    text_length = len(response_text)

    while index < text_length:
        while index < text_length and response_text[index].isspace():
            index += 1
        if index >= text_length:
            break
        try:
            payload, next_index = decoder.raw_decode(response_text, index)
        except json.JSONDecodeError:
            return None
        decoded_payloads.append(payload)
        index = next_index

    if len(decoded_payloads) < 2:
        return None
    return decoded_payloads


def _repair_multi_action_sequence(
    response_text: str,
) -> Optional[WikimediaJsonActionParseResult]:
    response_payloads = _decode_json_object_sequence(response_text)
    if response_payloads is None:
        return None

    parse_results = [
        _parse_single_response_payload(response_payload)
        for response_payload in response_payloads
    ]

    first_valid_abstain = next(
        (
            parse_result
            for parse_result in parse_results
            if parse_result.is_valid
            and parse_result.predicted_action == ACTION_ABSTAIN
        ),
        None,
    )
    if first_valid_abstain is not None:
        return first_valid_abstain

    first_valid_answer = next(
        (
            parse_result
            for parse_result in parse_results
            if parse_result.is_valid
            and parse_result.predicted_action == ACTION_ANSWER
        ),
        None,
    )
    if first_valid_answer is not None:
        return first_valid_answer

    if all(
        parse_result.is_valid and parse_result.predicted_action == ACTION_TOOL_CALL
        for parse_result in parse_results
    ):
        return parse_results[0]

    return None


def parse_wikimedia_json_response(
    response_text: Optional[str],
) -> WikimediaJsonActionParseResult:
    if response_text is None:
        return _invalid("Response is None.")

    stripped_response = response_text.strip()
    if not stripped_response:
        return _invalid("Response is empty.")

    try:
        response_payload = json.loads(stripped_response)
    except json.JSONDecodeError as exc:
        repaired_parse_result = _repair_multi_action_sequence(stripped_response)
        if repaired_parse_result is not None:
            return repaired_parse_result
        return _invalid(f"Response is not valid JSON: {exc.msg}")

    return _parse_single_response_payload(response_payload)
