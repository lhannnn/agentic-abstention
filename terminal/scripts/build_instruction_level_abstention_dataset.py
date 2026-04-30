#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "terminalbench_instructions.jsonl"
OUTPUT_PATH = ROOT / "terminalbench_instruction_level_abstention_267.jsonl"
PROMPTS_PATH = ROOT / "prompts" / "instruction_level_abstention_prompts.md"
CACHE_DIR = ROOT / ".cache" / "instruction_level_abstention_generation"
SKIPPED_OUTPUT_SUFFIX = ".skipped.jsonl"
SUMMARY_OUTPUT_SUFFIX = ".summary.json"
EXPECTED_BASE_TASKS = 89
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 8000
DEFAULT_AUDIT_MAX_OUTPUT_TOKENS = 2000

ORIGINAL_VARIANT = "original"
REWRITTEN_VARIANT = "rewritten"
COMBINED_CATEGORY = "false_premise_or_contradiction"
UNDERSPECIFIED_CATEGORY = "underspecified_intent"
TRANSFORMED_CATEGORIES = (COMBINED_CATEGORY, UNDERSPECIFIED_CATEGORY)

GENERATION_FAILURE_CODES = (
    "empty_output",
    "unchanged",
    "trivial_prefix_suffix_wrap",
    "contains_meta_language",
    "lost_task_surface",
    "not_instruction_level",
    "category_mismatch",
    "multiple_mechanisms",
    "environment_failure_instead",
    "not_critically_underspecified",
    "other",
)

GENERATION_RESPONSE_SCHEMA = {
    COMBINED_CATEGORY: {
        "name": "instruction_rewrite_false_premise_or_contradiction",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rewritten_instruction": {"type": "string"},
                "chosen_subtype": {
                    "type": "string",
                    "enum": ["false_premise", "contradiction"],
                },
            },
            "required": ["rewritten_instruction", "chosen_subtype"],
        },
    },
    UNDERSPECIFIED_CATEGORY: {
        "name": "instruction_rewrite_underspecified_intent",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rewritten_instruction": {"type": "string"},
            },
            "required": ["rewritten_instruction"],
        },
    },
}

AUDIT_RESPONSE_SCHEMA = {
    "name": "instruction_rewrite_audit",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pass": {"type": "boolean"},
            "failure_codes": {
                "type": "array",
                "items": {"type": "string", "enum": list(GENERATION_FAILURE_CODES)},
            },
            "reason": {"type": "string"},
            "rewrite_guidance": {"type": "string"},
        },
        "required": ["pass", "failure_codes", "reason", "rewrite_guidance"],
    },
}

META_LANGUAGE_PATTERNS = (
    "you should abstain",
    "the instruction is invalid",
    "this task is impossible",
    "this task is ambiguous",
    "the correct action is to abstain",
)


class StructuredResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.response_data = response_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the 267-row instruction-level abstention dataset by calling "
            "OpenAI GPT-5.4 on each of the 89 base instructions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help=f"Input base-instruction JSONL. Default: {INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output combined JSONL. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=PROMPTS_PATH,
        help=f"Prompt template file. Default: {PROMPTS_PATH}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Cache directory for raw request/response artifacts. Default: {CACHE_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Responses API model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=f"Responses API reasoning.effort value. Default: {DEFAULT_REASONING_EFFORT}",
    )
    parser.add_argument(
        "--expected-base-tasks",
        type=int,
        default=EXPECTED_BASE_TASKS,
        help=f"Expected number of base tasks. Default: {EXPECTED_BASE_TASKS}",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum rewrite attempts per task/category. Default: {DEFAULT_MAX_ATTEMPTS}",
    )
    parser.add_argument(
        "--generation-max-output-tokens",
        type=int,
        default=DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
        help=(
            "Max output tokens for rewrite generation calls. Default: "
            f"{DEFAULT_GENERATION_MAX_OUTPUT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--audit-max-output-tokens",
        type=int,
        default=DEFAULT_AUDIT_MAX_OUTPUT_TOKENS,
        help=(
            "Max output tokens for audit calls. Default: "
            f"{DEFAULT_AUDIT_MAX_OUTPUT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N base tasks after filtering.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Only process the specified base task name. May be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore cached successful rewrites and rebuild from scratch.",
    )
    parser.add_argument(
        "--skip-failed-after-max-attempts",
        action="store_true",
        help="Continue the full build by skipping rewrites that still fail after max attempts.",
    )
    parser.add_argument(
        "--skipped-output",
        type=Path,
        default=None,
        help="Where to write skipped rewrite records. Default: <output>.skipped.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Where to write the build summary JSON. Default: <output>.summary.json",
    )
    return parser.parse_args()


def load_base_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise RuntimeError(f"Input file not found: {input_path}")

    rows: list[dict[str, str]] = []
    for line_no, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"Blank line in input JSONL at line {line_no}")
        row = json.loads(line)
        if set(row) != {"task_name", "instruction"}:
            raise RuntimeError(
                f"Unexpected schema at line {line_no}: expected only "
                "'task_name' and 'instruction'"
            )
        if not isinstance(row["task_name"], str) or not row["task_name"].strip():
            raise RuntimeError(f"Invalid task_name at line {line_no}")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise RuntimeError(f"Empty instruction at line {line_no}")
        rows.append(row)
    return rows


def validate_base_rows(rows: list[dict[str, str]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} base tasks, found {len(rows)} in input."
        )

    task_names = [row["task_name"] for row in rows]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in base instructions.")


def filter_rows(
    rows: list[dict[str, str]],
    task_names: list[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    filtered = rows
    if task_names:
        wanted = set(task_names)
        filtered = [row for row in rows if row["task_name"] in wanted]
        missing = sorted(wanted - {row["task_name"] for row in filtered})
        if missing:
            raise RuntimeError(f"Unknown task_name filter(s): {', '.join(missing)}")

    if limit is not None:
        if limit <= 0:
            raise RuntimeError("--limit must be a positive integer")
        filtered = filtered[:limit]

    if not filtered:
        raise RuntimeError("No base tasks selected for processing")

    return filtered


def load_prompt_templates(prompts_path: Path) -> dict[str, str]:
    if not prompts_path.is_file():
        raise RuntimeError(f"Prompt template file not found: {prompts_path}")

    text = prompts_path.read_text(encoding="utf-8")
    templates: dict[str, str] = {}
    sections = {
        COMBINED_CATEGORY: "## false_premise_or_contradiction",
        UNDERSPECIFIED_CATEGORY: "## underspecified_intent",
    }
    for category, heading in sections.items():
        start = text.find(heading)
        if start == -1:
            raise RuntimeError(f"Missing heading in prompt file: {heading}")
        fence_start = text.find("```text", start)
        fence_end = text.find("```", fence_start + len("```text"))
        if fence_start == -1 or fence_end == -1:
            raise RuntimeError(f"Missing fenced text block for {heading}")
        body = text[fence_start + len("```text") : fence_end].strip()
        if not body:
            raise RuntimeError(f"Empty prompt body for {heading}")
        templates[category] = body
    return templates


def render_generator_prompt(
    template: str,
    task_name: str,
    original_instruction: str,
    rewrite_guidance: str | None,
) -> str:
    prompt = template.replace("{{TASK_NAME}}", task_name).replace(
        "{{ORIGINAL_INSTRUCTION}}", original_instruction
    )
    if rewrite_guidance:
        prompt += (
            "\n\nPrevious attempt failed QA. Fix the issues below while keeping the "
            "rewrite minimal and instruction-level:\n"
            f"{rewrite_guidance.strip()}\n"
        )
    return prompt


def render_audit_prompt(
    task_name: str,
    category: str,
    original_instruction: str,
    rewritten_instruction: str,
    chosen_subtype: str | None,
) -> str:
    category_rules = {
        COMBINED_CATEGORY: (
            "Pass only if the rewrite introduces exactly one valid instruction-level "
            "issue of type false premise or contradiction. It must remain close to the "
            "original task, preserve artifact surface, and be invalid before any "
            "environment interaction. Fail if it is unchanged, merely wrapped with a "
            "generic prefix/suffix, contains meta-language, uses multiple mechanisms, "
            "or turns into an environment/resource failure."
        ),
        UNDERSPECIFIED_CATEGORY: (
            "Pass only if the rewrite introduces a critical underspecification that "
            "makes the target no longer uniquely determined before any environment "
            "interaction. It must remain close to the original task and preserve "
            "artifact surface. Fail if it is unchanged, merely wrapped with a generic "
            "prefix/suffix, contains meta-language, only becomes slightly vague, or "
            "actually introduces contradiction, false premise, or environment failure."
        ),
    }
    subtype_text = chosen_subtype if chosen_subtype is not None else "null"
    return (
        "You are auditing a rewritten TerminalBench instruction.\n\n"
        f"TASK_NAME: {task_name}\n"
        f"TARGET_PUBLIC_CATEGORY: {category}\n"
        f"CHOSEN_SUBTYPE: {subtype_text}\n\n"
        "Audit policy:\n"
        f"{category_rules[category]}\n\n"
        "Return JSON matching the provided schema.\n"
        "If you fail the rewrite, include concise failure codes and actionable "
        "rewrite guidance.\n\n"
        "ORIGINAL_INSTRUCTION:\n"
        f"{original_instruction}\n\n"
        "REWRITTEN_INSTRUCTION:\n"
        f"{rewritten_instruction}\n"
    )


def create_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [serialize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump_json"):
        return json.loads(value.model_dump_json())
    if hasattr(value, "__dict__"):
        return {
            key: serialize_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def get_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def serialize_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
    return serialize_jsonable(response)


def summarize_response_output(response_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(response_data, dict):
        return []
    output = response_data.get("output")
    if not isinstance(output, list):
        return []

    summary: list[dict[str, Any]] = []
    for item in output[:5]:
        item_type = get_field(item, "type")
        content = get_field(item, "content")
        content_types: list[str] = []
        if isinstance(content, list):
            for content_item in content[:5]:
                content_type = get_field(content_item, "type")
                if isinstance(content_type, str):
                    content_types.append(content_type)
        entry: dict[str, Any] = {}
        if isinstance(item_type, str):
            entry["type"] = item_type
        if content_types:
            entry["content_types"] = content_types
        if entry:
            summary.append(entry)
    return summary


def format_response_context(response_data: dict[str, Any] | None) -> str:
    if not isinstance(response_data, dict):
        return ""

    parts: list[str] = []
    status = response_data.get("status")
    if status is not None:
        parts.append(f"status={status}")
    incomplete_details = response_data.get("incomplete_details")
    if incomplete_details not in (None, {}):
        parts.append(
            "incomplete_details="
            + json.dumps(incomplete_details, ensure_ascii=False, sort_keys=True)
        )
    error = response_data.get("error")
    if error not in (None, {}):
        parts.append("error=" + json.dumps(error, ensure_ascii=False, sort_keys=True))
    output_summary = summarize_response_output(response_data)
    if output_summary:
        parts.append(
            "output_summary="
            + json.dumps(output_summary, ensure_ascii=False, sort_keys=True)
        )
    return "; ".join(parts)


def collect_output_text_chunks(container: Any) -> list[str]:
    output = get_field(container, "output")
    if not isinstance(output, list):
        return []

    chunks: list[str] = []
    for item in output:
        content = get_field(item, "content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if get_field(content_item, "type") != "output_text":
                continue
            text = get_field(content_item, "text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return chunks


def extract_response_text(
    response: Any,
    *,
    response_data: dict[str, Any] | None = None,
) -> str:
    output_text = get_field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    if isinstance(response_data, dict):
        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

    chunks = collect_output_text_chunks(response)
    if not chunks and isinstance(response_data, dict):
        chunks = collect_output_text_chunks(response_data)
    if chunks:
        return "".join(chunks)

    context = format_response_context(response_data)
    if context:
        raise RuntimeError(f"Responses API returned no output_text. {context}")
    raise RuntimeError("Responses API returned no output_text")


def build_stage_error_record(exc: Exception) -> dict[str, Any]:
    response_data = getattr(exc, "response_data", None)
    payload: dict[str, Any] = {"error": str(exc)}
    if isinstance(response_data, dict):
        payload["response"] = response_data
        status = response_data.get("status")
        if status is not None:
            payload["response_status"] = status
    return payload


def call_structured_response(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str,
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    metadata: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
        metadata=metadata,
    )
    response_data = serialize_response(response)
    try:
        response_text = extract_response_text(response, response_data=response_data)
    except Exception as exc:  # noqa: BLE001
        raise StructuredResponseError(str(exc), response_data=response_data) from exc
    try:
        return json.loads(response_text), response_data
    except json.JSONDecodeError as exc:
        snippet = response_text[:300]
        raise StructuredResponseError(
            "Responses API returned invalid JSON. "
            f"decoder_error={exc.msg}; raw_text_snippet={snippet!r}",
            response_data=response_data,
        ) from exc


def local_rewrite_checks(
    original_instruction: str,
    rewritten_instruction: str,
) -> list[str]:
    failures: list[str] = []
    original = original_instruction.strip()
    rewritten = rewritten_instruction.strip()
    if not rewritten:
        failures.append("empty_output")
        return failures
    if rewritten == original:
        failures.append("unchanged")
    if rewritten.startswith(original) or rewritten.endswith(original):
        if rewritten != original:
            failures.append("trivial_prefix_suffix_wrap")
    lowered = rewritten.lower()
    if any(pattern in lowered for pattern in META_LANGUAGE_PATTERNS):
        failures.append("contains_meta_language")
    return failures


def rewrite_task_name(base_task_name: str, category: str) -> str:
    return f"{base_task_name}__{category}"


def derive_sidecar_path(output_path: Path, suffix: str) -> Path:
    if output_path.name.endswith(".jsonl"):
        base_name = output_path.name[: -len(".jsonl")]
    else:
        base_name = output_path.name
    return output_path.with_name(base_name + suffix)


def build_output_row(
    *,
    task_name: str,
    source_task_name: str,
    variant: str,
    instruction_level_category: str,
    instruction: str,
) -> dict[str, str]:
    return {
        "task_name": task_name,
        "source_task_name": source_task_name,
        "variant": variant,
        "instruction_level_category": instruction_level_category,
        "instruction": instruction,
    }


def build_rewrite_guidance(
    local_failure_codes: list[str], audit_payload: dict[str, Any]
) -> str:
    guidance_parts: list[str] = []
    if local_failure_codes:
        guidance_parts.append(
            "Local QA failures: " + ", ".join(sorted(set(local_failure_codes)))
        )
    audit_reason = str(audit_payload.get("reason") or "").strip()
    if audit_reason:
        guidance_parts.append("Audit reason: " + audit_reason)
    rewrite_guidance = str(audit_payload.get("rewrite_guidance") or "").strip()
    if rewrite_guidance:
        guidance_parts.append("Rewrite guidance: " + rewrite_guidance)
    return "\n".join(guidance_parts).strip()


def build_skipped_record(
    *,
    task_name: str,
    category: str,
    cache_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": rewrite_task_name(task_name, category),
        "source_task_name": task_name,
        "instruction_level_category": category,
        "variant": REWRITTEN_VARIANT,
        "status": "skipped",
        "failed_stage": record.get("failed_stage"),
        "failed_attempt": record.get("failed_attempt"),
        "error": record.get("error"),
        "last_failure_codes": list(record.get("last_failure_codes") or []),
        "last_rewrite_guidance": str(record.get("last_rewrite_guidance") or ""),
        "cache_path": str(cache_path.resolve()),
    }


def generate_rewrite(
    *,
    client: OpenAI,
    task_name: str,
    original_instruction: str,
    category: str,
    template: str,
    cache_root: Path,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    generation_max_output_tokens: int,
    audit_max_output_tokens: int,
    overwrite: bool,
    skip_failed_after_max_attempts: bool,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    cache_path = cache_root / f"{task_name}__{category}.json"
    record: dict[str, Any]
    if cache_path.is_file() and not overwrite:
        record = read_json(cache_path)
        if record.get("status") == "success" and isinstance(record.get("row"), dict):
            row = record["row"]
            if (
                row.get("task_name") == rewrite_task_name(task_name, category)
                and row.get("instruction_level_category") == category
            ):
                return row, None
    record = {
        "task_name": task_name,
        "category": category,
        "status": "in_progress",
        "attempts": [],
    }
    write_json(cache_path, record)

    rewrite_guidance: str | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_record: dict[str, Any] = {"attempt": attempt, "passed": False}
        generator_prompt = render_generator_prompt(
            template=template,
            task_name=task_name,
            original_instruction=original_instruction,
            rewrite_guidance=rewrite_guidance,
        )
        try:
            generation_payload, generation_raw = call_structured_response(
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=generator_prompt,
                schema_name=GENERATION_RESPONSE_SCHEMA[category]["name"],
                schema=GENERATION_RESPONSE_SCHEMA[category]["schema"],
                max_output_tokens=generation_max_output_tokens,
                metadata={
                    "task_name": task_name,
                    "category": category,
                    "stage": "generate",
                    "attempt": str(attempt),
                },
            )
        except Exception as exc:  # noqa: BLE001
            attempt_record["generation"] = build_stage_error_record(exc)
            record["attempts"].append(attempt_record)
            record["status"] = "retrying" if attempt < max_attempts else "failed"
            record["failed_stage"] = "generate"
            record["failed_attempt"] = attempt
            record["error"] = str(exc)
            write_json(cache_path, record)
            if attempt < max_attempts:
                continue
            if skip_failed_after_max_attempts:
                return None, build_skipped_record(
                    task_name=task_name,
                    category=category,
                    cache_path=cache_path,
                    record=record,
                )
            raise RuntimeError(
                "OpenAI generation request failed for "
                f"{task_name} [{category}] on attempt {attempt}: {exc}"
            ) from exc
        attempt_record["generation"] = {
            "payload": generation_payload,
            "response": generation_raw,
        }
        rewritten_instruction = str(
            generation_payload.get("rewritten_instruction") or ""
        ).strip()
        chosen_subtype = generation_payload.get("chosen_subtype")

        audit_prompt = render_audit_prompt(
            task_name=task_name,
            category=category,
            original_instruction=original_instruction,
            rewritten_instruction=rewritten_instruction,
            chosen_subtype=(
                str(chosen_subtype)
                if isinstance(chosen_subtype, str)
                else None
            ),
        )
        try:
            audit_payload, audit_raw = call_structured_response(
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt=audit_prompt,
                schema_name=AUDIT_RESPONSE_SCHEMA["name"],
                schema=AUDIT_RESPONSE_SCHEMA["schema"],
                max_output_tokens=audit_max_output_tokens,
                metadata={
                    "task_name": task_name,
                    "category": category,
                    "stage": "audit",
                    "attempt": str(attempt),
                },
            )
        except Exception as exc:  # noqa: BLE001
            attempt_record["audit"] = build_stage_error_record(exc)
            record["attempts"].append(attempt_record)
            record["status"] = "retrying" if attempt < max_attempts else "failed"
            record["failed_stage"] = "audit"
            record["failed_attempt"] = attempt
            record["error"] = str(exc)
            write_json(cache_path, record)
            if attempt < max_attempts:
                continue
            if skip_failed_after_max_attempts:
                return None, build_skipped_record(
                    task_name=task_name,
                    category=category,
                    cache_path=cache_path,
                    record=record,
                )
            raise RuntimeError(
                "OpenAI audit request failed for "
                f"{task_name} [{category}] on attempt {attempt}: {exc}"
            ) from exc
        local_failure_codes = local_rewrite_checks(
            original_instruction=original_instruction,
            rewritten_instruction=rewritten_instruction,
        )
        audit_failure_codes = [
            code
            for code in audit_payload.get("failure_codes", [])
            if isinstance(code, str)
        ]
        passed = bool(audit_payload.get("pass")) and not local_failure_codes
        attempt_record["audit"] = {
            "payload": audit_payload,
            "response": audit_raw,
        }
        attempt_record["local_failure_codes"] = local_failure_codes
        attempt_record["passed"] = passed
        record["attempts"].append(attempt_record)
        if passed:
            row = build_output_row(
                task_name=rewrite_task_name(task_name, category),
                source_task_name=task_name,
                variant=REWRITTEN_VARIANT,
                instruction_level_category=category,
                instruction=rewritten_instruction,
            )
            record["status"] = "success"
            record["row"] = row
            write_json(cache_path, record)
            return row, None

        rewrite_guidance = build_rewrite_guidance(local_failure_codes, audit_payload)
        record["status"] = "retrying"
        record["last_failure_codes"] = sorted(
            set(local_failure_codes + audit_failure_codes)
        )
        record["last_rewrite_guidance"] = rewrite_guidance
        write_json(cache_path, record)

    record["status"] = "failed"
    record["failed_stage"] = "validation"
    record["failed_attempt"] = max_attempts
    record["error"] = (
        f"Failed to generate a valid rewrite for {task_name} [{category}] after "
        f"{max_attempts} attempt(s)."
    )
    write_json(cache_path, record)
    if skip_failed_after_max_attempts:
        return None, build_skipped_record(
            task_name=task_name,
            category=category,
            cache_path=cache_path,
            record=record,
        )
    raise RuntimeError(
        f"Failed to generate a valid rewrite for {task_name} [{category}] after "
        f"{max_attempts} attempt(s). Cache: {cache_path}"
    )


def build_dataset(
    *,
    rows: list[dict[str, str]],
    templates: dict[str, str],
    client: OpenAI,
    cache_dir: Path,
    model: str,
    reasoning_effort: str,
    max_attempts: int,
    generation_max_output_tokens: int,
    audit_max_output_tokens: int,
    overwrite: bool,
    skip_failed_after_max_attempts: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    combined: list[dict[str, str]] = []
    skipped_rewrites: list[dict[str, Any]] = []

    for row in rows:
        task_name = row["task_name"]
        instruction = row["instruction"]
        combined.append(
            build_output_row(
                task_name=task_name,
                source_task_name=task_name,
                variant=ORIGINAL_VARIANT,
                instruction_level_category=ORIGINAL_VARIANT,
                instruction=instruction,
            )
        )
        for category in TRANSFORMED_CATEGORIES:
            rewrite_row, skipped_record = generate_rewrite(
                client=client,
                task_name=task_name,
                original_instruction=instruction,
                category=category,
                template=templates[category],
                cache_root=cache_dir,
                model=model,
                reasoning_effort=reasoning_effort,
                max_attempts=max_attempts,
                generation_max_output_tokens=generation_max_output_tokens,
                audit_max_output_tokens=audit_max_output_tokens,
                overwrite=overwrite,
                skip_failed_after_max_attempts=skip_failed_after_max_attempts,
            )
            if rewrite_row is not None:
                combined.append(rewrite_row)
            if skipped_record is not None:
                skipped_rewrites.append(skipped_record)

    return combined, skipped_rewrites


def validate_dataset(
    dataset: list[dict[str, str]],
    *,
    full_build: bool,
    expected_base_tasks: int,
    expected_selected_base_tasks: int,
    skip_failed_after_max_attempts: bool,
) -> None:
    task_names = [row["task_name"] for row in dataset]
    if len(set(task_names)) != len(task_names):
        raise RuntimeError("Duplicate task_name found in combined dataset.")

    for row in dataset:
        if not row["instruction"].strip():
            raise RuntimeError(f"Empty instruction in output row: {row['task_name']}")
        if row["variant"] not in {ORIGINAL_VARIANT, REWRITTEN_VARIANT}:
            raise RuntimeError(f"Unexpected variant in output row: {row['task_name']}")
        if row["instruction_level_category"] not in {
            ORIGINAL_VARIANT,
            COMBINED_CATEGORY,
            UNDERSPECIFIED_CATEGORY,
        }:
            raise RuntimeError(
                f"Unexpected instruction_level_category in output row: {row['task_name']}"
            )

    variant_counts = Counter(row["variant"] for row in dataset)
    category_counts = Counter(row["instruction_level_category"] for row in dataset)
    expected_original_count = (
        expected_base_tasks if full_build else expected_selected_base_tasks
    )
    if variant_counts[ORIGINAL_VARIANT] != expected_original_count:
        raise RuntimeError(
            "Unexpected original variant count: "
            f"{variant_counts[ORIGINAL_VARIANT]} != {expected_original_count}"
        )
    if category_counts[ORIGINAL_VARIANT] != expected_original_count:
        raise RuntimeError(
            "Unexpected original category count: "
            f"{category_counts[ORIGINAL_VARIANT]} != {expected_original_count}"
        )

    if skip_failed_after_max_attempts:
        if variant_counts[REWRITTEN_VARIANT] > expected_selected_base_tasks * 2:
            raise RuntimeError("Too many rewritten rows in partial dataset.")
        if category_counts[COMBINED_CATEGORY] > expected_selected_base_tasks:
            raise RuntimeError("Too many combined-category rows in partial dataset.")
        if category_counts[UNDERSPECIFIED_CATEGORY] > expected_selected_base_tasks:
            raise RuntimeError(
                "Too many underspecified-category rows in partial dataset."
            )
        return

    if full_build:
        expected_total = expected_base_tasks * 3
        if len(dataset) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} rows in combined dataset, found {len(dataset)}."
            )
        expected_variant_counts = {
            ORIGINAL_VARIANT: expected_base_tasks,
            REWRITTEN_VARIANT: expected_base_tasks * 2,
        }
        expected_category_counts = {
            ORIGINAL_VARIANT: expected_base_tasks,
            COMBINED_CATEGORY: expected_base_tasks,
            UNDERSPECIFIED_CATEGORY: expected_base_tasks,
        }
        if dict(variant_counts) != expected_variant_counts:
            raise RuntimeError(
                f"Unexpected variant counts: {dict(variant_counts)} != {expected_variant_counts}"
            )
        if dict(category_counts) != expected_category_counts:
            raise RuntimeError(
                "Unexpected instruction_level_category counts: "
                f"{dict(category_counts)} != {expected_category_counts}"
            )


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    tmp_path.replace(output_path)


def build_summary_payload(
    *,
    dataset: list[dict[str, str]],
    skipped_rewrites: list[dict[str, Any]],
    expected_selected_base_tasks: int,
    full_build: bool,
) -> dict[str, Any]:
    category_counts = Counter(row["instruction_level_category"] for row in dataset)
    skipped_category_counts = Counter(
        row["instruction_level_category"] for row in skipped_rewrites
    )
    return {
        "full_build": full_build,
        "selected_base_tasks": expected_selected_base_tasks,
        "original_count": category_counts[ORIGINAL_VARIANT],
        "successful_rewrite_count": len(dataset) - category_counts[ORIGINAL_VARIANT],
        "skipped_rewrite_count": len(skipped_rewrites),
        "dataset_row_count": len(dataset),
        "successful_category_counts": dict(category_counts),
        "skipped_category_counts": dict(skipped_category_counts),
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    skipped_output_path = (
        args.skipped_output.expanduser().resolve()
        if args.skipped_output is not None
        else derive_sidecar_path(output_path, SKIPPED_OUTPUT_SUFFIX)
    )
    summary_output_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else derive_sidecar_path(output_path, SUMMARY_OUTPUT_SUFFIX)
    )
    prompts_path = args.prompts.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    try:
        rows = load_base_rows(input_path)
        validate_base_rows(rows, args.expected_base_tasks)
        rows = filter_rows(rows, args.task_name, args.limit)
        selected_base_tasks = len(rows)
        templates = load_prompt_templates(prompts_path)
        client = create_openai_client()
        full_build = args.task_name is None and args.limit is None
        if not full_build and output_path == OUTPUT_PATH and not args.overwrite:
            raise RuntimeError(
                "Refusing to overwrite the canonical 267-row output during a filtered "
                "run. Pass --overwrite or set --output to a sample path."
            )

        dataset, skipped_rewrites = build_dataset(
            rows=rows,
            templates=templates,
            client=client,
            cache_dir=cache_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
            generation_max_output_tokens=args.generation_max_output_tokens,
            audit_max_output_tokens=args.audit_max_output_tokens,
            overwrite=args.overwrite,
            skip_failed_after_max_attempts=args.skip_failed_after_max_attempts,
        )
        validate_dataset(
            dataset,
            full_build=full_build,
            expected_base_tasks=args.expected_base_tasks,
            expected_selected_base_tasks=selected_base_tasks,
            skip_failed_after_max_attempts=args.skip_failed_after_max_attempts,
        )
        write_jsonl(dataset, output_path)
        write_jsonl(skipped_rewrites, skipped_output_path)
        write_json(
            summary_output_path,
            build_summary_payload(
                dataset=dataset,
                skipped_rewrites=skipped_rewrites,
                expected_selected_base_tasks=selected_base_tasks,
                full_build=full_build,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    category_counts = Counter(row["instruction_level_category"] for row in dataset)
    print(f"Wrote {len(dataset)} rows to {output_path}")
    print(f"Category counts: {dict(category_counts)}")
    print(f"Skipped rewrites: {len(skipped_rewrites)}")
    print(f"Skipped report: {skipped_output_path}")
    print(f"Summary: {summary_output_path}")
    print(f"Cache dir: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
