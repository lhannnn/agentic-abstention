#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


CODEX_OBSERVATION_TYPES = {"function_call_output", "custom_tool_call_output"}
# Keep this registry explicit. Add agent names only after validating that one
# ATIF agent step really corresponds to one observe-act decision turn.
ATIF_STEP_AGENT_NAMES: frozenset[str] = frozenset(
    {
        "terminus-2",
    }
)


def normalize_round_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def load_result_json(trial_dir: Path) -> dict[str, Any]:
    return json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))


def actual_decision_from_result(result: dict[str, Any]) -> str:
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    metadata = metadata or {}

    decision = metadata.get("decision")
    if decision in {"abstain", "continue", "no_decision"}:
        return str(decision)

    abstain = metadata.get("abstain")
    if abstain is True:
        return "abstain"
    if abstain is False:
        return "continue"
    return "no_decision"


def legacy_n_interaction_rounds(result: dict[str, Any]) -> int | None:
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, dict) else None
    metadata = metadata or {}
    return normalize_round_count(metadata.get("n_interaction_rounds"))


def _extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        inner_text = part.get("content")
        if isinstance(inner_text, str):
            parts.append(inner_text)
    return "".join(parts).strip()


def _step_records_abstain(step: dict[str, Any]) -> bool:
    message = step.get("message")
    text = _extract_message_text(message)
    if text.lstrip().startswith("ABSTAIN "):
        return True

    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("function_name") == "mark_task_abstained":
                return True

    observation = step.get("observation")
    if isinstance(observation, dict):
        results = observation.get("results")
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                content = result.get("content")
                content_text = _extract_message_text(content)
                if "Agent abstained" in content_text:
                    return True

    return False


def _find_codex_session_file(trial_dir: Path) -> Path | None:
    session_files = sorted((trial_dir / "agent" / "sessions").glob("**/*.jsonl"))
    return session_files[-1] if session_files else None


def _extract_codex_turn_metrics_from_session(session_file: Path) -> dict[str, Any]:
    awaiting_decision = True
    observe_act_turns = 0
    first_abstain_turn: int | None = None
    assistant_messages = 0
    observation_events = 0
    turn_events: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        session_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") != "response_item":
            continue

        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        if payload_type == "message" and payload.get("role") == "assistant":
            assistant_messages += 1
            counted_new_turn = False
            if awaiting_decision:
                observe_act_turns += 1
                awaiting_decision = False
                counted_new_turn = True

            text = _extract_message_text(payload.get("content"))
            if first_abstain_turn is None and text.lstrip().startswith("ABSTAIN "):
                first_abstain_turn = observe_act_turns

            turn_events.append(
                {
                    "line_no": line_no,
                    "event": "assistant_message",
                    "counted_new_turn": counted_new_turn,
                    "observe_act_turn": observe_act_turns,
                    "message_prefix": text[:120],
                    "is_abstain": text.lstrip().startswith("ABSTAIN "),
                }
            )
            continue

        if payload_type in CODEX_OBSERVATION_TYPES:
            observation_events += 1
            awaiting_decision = True
            turn_events.append(
                {
                    "line_no": line_no,
                    "event": payload_type,
                    "observe_act_turn": observe_act_turns,
                }
            )

    return {
        "observe_act_turns": observe_act_turns,
        "observe_act_turn_source": "codex_session_jsonl",
        "first_abstain_turn": first_abstain_turn,
        "turn_debug": {
            "session_file": str(session_file),
            "assistant_message_count": assistant_messages,
            "observation_event_count": observation_events,
            "turn_events": turn_events,
        },
    }


def _extract_atif_step_turn_metrics(trial_dir: Path) -> dict[str, Any] | None:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        return None

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return None

    observe_act_turns = 0
    first_abstain_turn: int | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("source") != "agent":
            continue
        if step.get("is_copied_context") is True:
            continue
        observe_act_turns += 1
        if first_abstain_turn is None and _step_records_abstain(step):
            first_abstain_turn = observe_act_turns

    return {
        "observe_act_turns": observe_act_turns,
        "observe_act_turn_source": "atif_step_adapter",
        "first_abstain_turn": first_abstain_turn,
        "turn_debug": {
            "trajectory_path": str(trajectory_path),
            "agent_step_count": observe_act_turns,
        },
    }


def _registered_backend_for_agent(agent_name: str | None) -> tuple[str, Any] | None:
    if agent_name == "codex":
        return "codex_session_jsonl", _extract_codex_turn_metrics_from_session
    if isinstance(agent_name, str) and agent_name in ATIF_STEP_AGENT_NAMES:
        return "atif_step_adapter", _extract_atif_step_turn_metrics
    return None


def compute_trial_turn_metrics(
    trial_dir: Path, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = result or load_result_json(trial_dir)
    agent_info = result.get("agent_info")
    agent_name = agent_info.get("name") if isinstance(agent_info, dict) else None
    legacy_rounds = legacy_n_interaction_rounds(result)
    actual_decision = actual_decision_from_result(result)

    backend = _registered_backend_for_agent(agent_name)
    if backend is not None and backend[0] == "codex_session_jsonl":
        session_file = _find_codex_session_file(trial_dir)
        if session_file is not None:
            metrics = backend[1](session_file)
            metrics["legacy_n_interaction_rounds"] = legacy_rounds
            return metrics

    if backend is not None and backend[0] == "atif_step_adapter":
        metrics = backend[1](trial_dir)
        if metrics is not None:
            metrics["legacy_n_interaction_rounds"] = legacy_rounds
            return metrics

    first_abstain_turn = (
        legacy_rounds if actual_decision == "abstain" and legacy_rounds is not None else None
    )
    return {
        "observe_act_turns": legacy_rounds,
        "observe_act_turn_source": "legacy_fallback"
        if legacy_rounds is not None
        else "missing",
        "first_abstain_turn": first_abstain_turn,
        "legacy_n_interaction_rounds": legacy_rounds,
        "turn_debug": None,
    }


def summarize_turn_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observe_distribution = Counter()
    legacy_distribution = Counter()
    source_counts = Counter()

    for row in rows:
        observe_act_turns = row.get("observe_act_turns")
        if isinstance(observe_act_turns, int):
            observe_distribution[str(observe_act_turns)] += 1
        legacy_rounds = row.get("legacy_n_interaction_rounds")
        if isinstance(legacy_rounds, int):
            legacy_distribution[str(legacy_rounds)] += 1
        source = row.get("observe_act_turn_source")
        if isinstance(source, str):
            source_counts[source] += 1

    return {
        "trial_count": len(rows),
        "observe_act_turn_distribution": dict(
            sorted(observe_distribution.items(), key=lambda item: int(item[0]))
        ),
        "legacy_turn_distribution": dict(
            sorted(legacy_distribution.items(), key=lambda item: int(item[0]))
        ),
        "observe_act_turn_source_counts": dict(sorted(source_counts.items())),
    }
