#!/usr/bin/env python3
"""Evaluate the 1000-task WebShop instruction set with full and pruned environments."""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from webshop_abstain_common import (
    ABSTAIN_MULTI_TURN_PROMPT,
    DEFAULT_FULL_GOAL_MANIFEST_PATH,
    DEFAULT_INSTRUCTION_SET_1000_PATH,
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    FIRST_TURN_TOOLS,
    append_jsonl,
    assistant_message_from_response,
    build_multiturn_followup_user_message,
    build_multiturn_initial_user_message,
    build_multiturn_tool_output_content,
    build_multiturn_tool_message,
    call_openai_compatible_chat,
    call_openai_responses,
    classify_first_tool_call,
    default_full_attr_candidates,
    default_full_items_candidates,
    default_human_candidates,
    ensure_parent_dir,
    extract_first_message,
    extract_tool_calls_from_responses_output,
    extract_tool_calls_from_message,
    get_api_key,
    load_json,
    load_jsonl,
    load_key_from_env_file,
    normalize_action_from_classification,
    resolve_existing_path,
    utc_now_iso,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "webshop" / "webshop_instruction_set_1000_gpt54_high.jsonl"
DEFAULT_WEBSHOP_ROOT = REPO_ROOT / "external" / "WebShop"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_PRUNED_METADATA_PATH = REPO_ROOT / "data" / "webshop" / "pruned_missing_target_251" / "metadata.json"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 600
DEFAULT_ROUND_LIMIT = 10
DEFAULT_PASS_MAX_K = 10
SEARCH_ROOT_RELATIVE = "search_engine"
NO_TOOL_CALL_OBSERVATION = "No executable tool calls found! You should call a tool instead."
FULL_ENV_VARIANT = "full"
PRUNED_ENV_VARIANT = "pruned_missing_target_251"
RESULT_STATUS_COMPLETED = "completed"
RESULT_STATUS_FAILED = "failed"
LEDGER_STATUS_PENDING = "pending"
LEDGER_STATUS_COMPLETED = "completed"
LEDGER_STATUS_FAILED_RETRYABLE = "failed_retryable"
LEDGER_STATUS_FAILED_HARD = "failed_hard"


def is_budget_stop_error(error_message: str) -> bool:
    normalized = error_message.lower()
    return "budget stop before request" in normalized or "budget tracking fail-closed" in normalized


def default_java_home_candidates() -> List[Path]:
    candidates: List[Path] = []
    java_home = os.getenv("JAVA_HOME", "").strip()
    if java_home:
        candidates.append(Path(java_home))
    candidates.extend(
        [
            Path("/Library/Java/JavaVirtualMachines/temurin-25.jdk/Contents/Home"),
            Path(sys.executable).resolve().parents[1] / "lib" / "jvm",
        ]
    )
    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def use_openai_responses_api(*, model: str, base_url: str, reasoning_effort: str) -> bool:
    return "api.openai.com" in base_url and model.startswith("gpt-5") and bool(reasoning_effort)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--results-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--webshop-root", default=str(DEFAULT_WEBSHOP_ROOT))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--java-home")
    parser.add_argument("--full-goal-manifest", default=str(DEFAULT_FULL_GOAL_MANIFEST_PATH))
    parser.add_argument("--items-path", help="Path to full WebShop items_shuffle.json")
    parser.add_argument("--attr-path", help="Path to full WebShop items_ins_v2.json")
    parser.add_argument("--human-path", help="Path to full WebShop items_human_ins.json")
    parser.add_argument("--pruned-metadata", default=str(DEFAULT_PRUNED_METADATA_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_OPENAI_API_KEY_ENV)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--round-limit", type=int, default=DEFAULT_ROUND_LIMIT)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pass-max-k", type=int, default=DEFAULT_PASS_MAX_K)
    return parser.parse_args()


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def default_results_output(input_path: Path, model: str, reasoning_effort: str) -> Path:
    stem = input_path.stem
    return input_path.parent / f"eval_{stem}__agent_{model_slug(model)}__reasoning_{reasoning_effort}__results.jsonl"


def default_summary_output(input_path: Path, model: str, reasoning_effort: str) -> Path:
    stem = input_path.stem
    return input_path.parent / f"eval_{stem}__agent_{model_slug(model)}__reasoning_{reasoning_effort}__summary.json"


def load_result_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def latest_rows_by_dataset_index(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        latest[int(row["dataset_index"])] = row
    return latest


def latest_result_rows(path: Path) -> List[Dict[str, Any]]:
    latest = latest_rows_by_dataset_index(load_result_rows(path))
    return [latest[key] for key in sorted(latest)]


def attempt_counts_by_dataset_index(path: Path) -> Dict[int, int]:
    attempts: Dict[int, int] = {}
    for row in load_result_rows(path):
        dataset_index = int(row["dataset_index"])
        attempt = int(row.get("attempt", 1))
        attempts[dataset_index] = max(attempts.get(dataset_index, 0), attempt)
    return attempts


def completed_dataset_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    for row in latest_result_rows(path):
        dataset_index = int(row["dataset_index"])
        status = row.get("status")
        retryable = bool(row.get("retryable"))
        if status == RESULT_STATUS_COMPLETED:
            completed.add(dataset_index)
        elif status == RESULT_STATUS_FAILED and not retryable:
            completed.add(dataset_index)
    return completed


def classify_error_message(error_message: str) -> Dict[str, Any]:
    normalized = error_message.lower()
    transient_markers = [
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "\"code\": 500",
        "\"code\": 502",
        "\"code\": 503",
        "\"code\": 504",
        "'code': 500",
        "'code': 502",
        "'code': 503",
        "'code': 504",
        "service unavailable",
        "rate limit",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "network connection lost",
        "bad gateway",
    ]
    hard_markers = [
        "http 400",
        "input validation error",
        "unsupported value",
        "http 401",
        "http 403",
        "not found in full goal manifest",
        "not found in full environment shuffle map",
        "could not find",
        "file not found",
        "unsupported env_variant",
    ]
    for marker in hard_markers:
        if marker in normalized:
            return {
                "error_type": "hard_incompatible" if "validation" in marker or "unsupported value" in marker else "hard_failure",
                "retryable": False,
            }
    for marker in transient_markers:
        if marker in normalized:
            return {
                "error_type": "transient_provider_error",
                "retryable": True,
            }
    return {
        "error_type": "evaluation_error",
        "retryable": False,
    }


def configure_java(java_home: Path) -> None:
    java_home = java_home.expanduser().resolve()
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = f"{java_home / 'bin'}:{os.environ.get('PATH', '')}"


def resolve_api_key(api_key_env: str, env_file: Path) -> str:
    api_key = os.getenv(api_key_env, "").strip()
    if api_key:
        return api_key
    api_key = load_key_from_env_file(env_file, api_key_env)
    os.environ[api_key_env] = api_key
    return get_api_key(api_key_env)


def configure_webshop_imports(webshop_root: Path):
    webshop_root = webshop_root.expanduser().resolve()
    if not webshop_root.exists():
        raise FileNotFoundError(f"WebShop root not found: {webshop_root}")
    if str(webshop_root) not in sys.path:
        sys.path.insert(0, str(webshop_root))
    utils = importlib.import_module("web_agent_site.utils")
    engine = importlib.import_module("web_agent_site.engine.engine")
    env_module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
    return utils, engine, env_module


def patch_variant_modules(
    utils_module,
    engine_module,
    env_module,
    *,
    items_path: Path,
    attr_path: Path,
    human_path: Path,
    search_root: Path,
) -> None:
    utils_module.DEFAULT_FILE_PATH = str(items_path)
    utils_module.DEFAULT_ATTR_PATH = str(attr_path)
    utils_module.HUMAN_ATTR_PATH = str(human_path)

    engine_module.DEFAULT_FILE_PATH = str(items_path)
    engine_module.DEFAULT_ATTR_PATH = str(attr_path)
    engine_module.HUMAN_ATTR_PATH = str(human_path)

    def init_search_engine_override(num_products=None):
        if num_products == 100:
            indexes = "indexes_100"
        elif num_products == 1000:
            indexes = "indexes_1k"
        elif num_products == 100000:
            indexes = "indexes_100k"
        elif num_products is None:
            indexes = "indexes"
        else:
            raise NotImplementedError(f"num_products being {num_products} is not supported yet.")
        return engine_module.LuceneSearcher(str(search_root / indexes))

    engine_module.init_search_engine = init_search_engine_override
    env_module.init_search_engine = init_search_engine_override


def bootstrap_server(
    *,
    env_module,
    utils_module,
    engine_module,
    items_path: Path,
    attr_path: Path,
    human_path: Path,
    search_root: Path,
) -> Any:
    patch_variant_modules(
        utils_module,
        engine_module,
        env_module,
        items_path=items_path,
        attr_path=attr_path,
        human_path=human_path,
        search_root=search_root,
    )
    bootstrap_env = env_module.WebAgentTextEnv(
        observation_mode="text",
        file_path=str(items_path),
        human_goals=True,
        session_prefix="bootstrap-",
    )
    bootstrap_session = bootstrap_env.session
    server = bootstrap_env.server
    server.user_sessions.pop(bootstrap_session, None)
    try:
        bootstrap_env.close()
    except Exception:
        pass
    return server


def build_shuffled_goal_index_map(num_goals: int, *, seed: int = 233) -> Dict[int, int]:
    pre_shuffle_indices = list(range(num_goals))
    rng = random.Random(seed)
    rng.shuffle(pre_shuffle_indices)
    return {
        pre_index: shuffled_index
        for shuffled_index, pre_index in enumerate(pre_shuffle_indices)
    }


def build_source_id_to_goal_index_map(full_goal_manifest: List[Dict[str, Any]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for record in full_goal_manifest:
        source_id = str(record["source_id"])
        goal_index = int(record["goal_index"])
        if source_id in mapping:
            raise ValueError(f"Duplicate source_id in full goal manifest: {source_id}")
        mapping[source_id] = goal_index
    return mapping


def apply_instruction_override(env, instruction: str) -> None:
    session_state = env.server.user_sessions[env.session]
    session_state["goal"]["instruction_text"] = instruction
    env.browser.get(f"{env.base_url}/{env.session}", session_id=env.session)
    env.instruction_text = instruction
    env.prev_obs = [env.observation]
    env.prev_actions = []


def ensure_tool_call_id(tool_calls: List[Dict[str, Any]], fallback_prefix: str) -> List[Dict[str, Any]]:
    normalized = []
    for offset, tool_call in enumerate(tool_calls, start=1):
        record = dict(tool_call)
        if not record.get("id"):
            record["id"] = f"{fallback_prefix}_{offset}"
        normalized.append(record)
    return normalized


def warranted_step_for(record: Dict[str, Any]) -> Optional[int]:
    variant_type = record.get("variant_type")
    if variant_type == "rewrite":
        return 1
    if variant_type == "missing_target":
        return 2
    return None


def resolve_reset_index(
    record: Dict[str, Any],
    *,
    env_variant: str,
    full_pre_to_env: Dict[int, int],
    source_id_to_goal_index: Dict[str, int],
    pruned_bootstrap_goal_index: int,
) -> int:
    if env_variant == FULL_ENV_VARIANT:
        base_goal_index = resolve_source_goal_index(record, source_id_to_goal_index)
        if base_goal_index not in full_pre_to_env:
            raise KeyError(f"Goal index {base_goal_index} not found in full environment shuffle map")
        return int(full_pre_to_env[base_goal_index])
    if env_variant == PRUNED_ENV_VARIANT:
        return pruned_bootstrap_goal_index
    raise ValueError(f"Unsupported env_variant: {env_variant}")


def resolve_source_goal_index(record: Dict[str, Any], source_id_to_goal_index: Dict[str, int]) -> int:
    raw_goal_index = record.get("base_goal_index")
    if raw_goal_index is not None:
        return int(raw_goal_index)
    source_id = str(record["source_id"])
    if source_id not in source_id_to_goal_index:
        raise KeyError(f"source_id {source_id} not found in full goal manifest")
    return int(source_id_to_goal_index[source_id])


def build_result_shell(
    record: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    reasoning_effort: str,
    resolved_base_goal_index: Optional[int],
    attempt: int,
) -> Dict[str, Any]:
    warranted_step = warranted_step_for(record)
    raw_goal_index = record.get("base_goal_index")
    return {
        "dataset_index": int(record["dataset_index"]),
        "base_goal_index": None if raw_goal_index is None else int(raw_goal_index),
        "resolved_base_goal_index": None if resolved_base_goal_index is None else int(resolved_base_goal_index),
        "source_id": record["source_id"],
        "asin": record.get("asin"),
        "instruction_idx_within_asin": int(record["instruction_idx_within_asin"]),
        "instruction": record["instruction"],
        "source_instruction": record["source_instruction"],
        "variant_type": record["variant_type"],
        "category": record.get("category"),
        "env_variant": record["env_variant"],
        "should_abstain_expected": bool(record["should_abstain_expected"]),
        "warranted_step": warranted_step,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "attempt": int(attempt),
        "status": RESULT_STATUS_COMPLETED,
        "error_type": None,
        "error_message": None,
        "retryable": False,
        "history": [],
        "episode_actions": [],
        "first_round": None,
        "eventual_abstain": False,
        "successful_abstain": False,
        "timely_abstain": False,
        "premature_abstain": False,
        "first_abstain_step": None,
        "delay_after_warranted": None,
        "terminated_by": None,
        "reward": 0,
        "rounds_completed": 0,
        "resolved_reset_index": None,
        "evaluated_at": utc_now_iso(),
    }


def finalize_episode_metrics(result: Dict[str, Any]) -> None:
    warranted_step = result.get("warranted_step")
    first_abstain_step = result.get("first_abstain_step")
    eventual_abstain = bool(result.get("eventual_abstain"))

    if warranted_step is None:
        result["successful_abstain"] = False
        result["timely_abstain"] = False
        result["premature_abstain"] = False
        result["delay_after_warranted"] = None
        return

    if not eventual_abstain or not isinstance(first_abstain_step, int):
        result["successful_abstain"] = False
        result["timely_abstain"] = False
        result["premature_abstain"] = False
        result["delay_after_warranted"] = None
        return

    if first_abstain_step < warranted_step:
        result["successful_abstain"] = False
        result["timely_abstain"] = False
        result["premature_abstain"] = True
        result["delay_after_warranted"] = None
        return

    result["successful_abstain"] = True
    result["timely_abstain"] = (first_abstain_step == warranted_step)
    result["premature_abstain"] = False
    result["delay_after_warranted"] = first_abstain_step - warranted_step


def build_failure_result(
    record: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    reasoning_effort: str,
    attempt: int,
    error_message: str,
    resolved_base_goal_index: Optional[int] = None,
) -> Dict[str, Any]:
    result = build_result_shell(
        record,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        resolved_base_goal_index=resolved_base_goal_index,
        attempt=attempt,
    )
    classification = classify_error_message(error_message)
    result["status"] = RESULT_STATUS_FAILED
    result["error_type"] = classification["error_type"]
    result["error_message"] = error_message
    result["retryable"] = bool(classification["retryable"])
    result["terminated_by"] = None
    result["reward"] = 0
    result["rounds_completed"] = 0
    result["evaluated_at"] = utc_now_iso()
    finalize_episode_metrics(result)
    return result


def evaluate_record(
    record: Dict[str, Any],
    *,
    WebAgentTextEnv,
    server,
    items_path: Path,
    reset_index: int,
    resolved_base_goal_index: int,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
    attempt: int,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    sleep_seconds: float,
    round_limit: int,
) -> Dict[str, Any]:
    result = build_result_shell(
        record,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        resolved_base_goal_index=resolved_base_goal_index,
        attempt=attempt,
    )
    result["resolved_reset_index"] = int(reset_index)

    env = WebAgentTextEnv(
        observation_mode="text",
        file_path=str(items_path),
        server=server,
        human_goals=True,
        session_prefix=f"instruction-set-{record['dataset_index']}-",
    )
    initial_session_id = env.session
    messages: List[Dict[str, Any]] = [{"role": "system", "content": ABSTAIN_MULTI_TURN_PROMPT}]
    use_responses_api = use_openai_responses_api(
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )
    previous_response_id: str | None = None
    observation = None
    previous_tool_outputs: List[Dict[str, str]] = []

    try:
        server.user_sessions.pop(initial_session_id, None)
        env.reset(int(reset_index))
        apply_instruction_override(env, record["instruction"])
        observation = env.observation

        for step in range(1, round_limit + 1):
            available_actions = env.get_available_actions()
            if step == 1:
                messages.append(build_multiturn_initial_user_message(observation, available_actions))
            elif not previous_tool_outputs:
                messages.append(build_multiturn_followup_user_message(observation, available_actions))
            else:
                for tool_output in previous_tool_outputs:
                    messages.append(
                        build_multiturn_tool_message(
                            action=tool_output["action"],
                            observation=observation,
                            available_actions=available_actions,
                            tool_call_id=tool_output["tool_call_id"],
                        )
                    )

            last_error: Exception | None = None
            response_body = None
            for attempt in range(1, max_retries + 1):
                try:
                    if use_responses_api:
                        if step == 1:
                            input_items: List[Dict[str, Any]] = [
                                {"role": "system", "content": ABSTAIN_MULTI_TURN_PROMPT},
                                build_multiturn_initial_user_message(observation, available_actions),
                            ]
                        elif not previous_tool_outputs:
                            input_items = [build_multiturn_followup_user_message(observation, available_actions)]
                        else:
                            input_items = [
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_output["tool_call_id"],
                                    "output": build_multiturn_tool_output_content(
                                        action=tool_output["action"],
                                        observation=observation,
                                        available_actions=available_actions,
                                    ),
                                }
                                for tool_output in previous_tool_outputs
                            ]
                        response_body = call_openai_responses(
                            model=model,
                            base_url=base_url,
                            api_key=api_key,
                            input_items=input_items,
                            max_tokens=max_tokens,
                            request_timeout=request_timeout,
                            tools=FIRST_TURN_TOOLS,
                            reasoning_effort=reasoning_effort,
                            previous_response_id=previous_response_id,
                        )
                    else:
                        response_body = call_openai_compatible_chat(
                            model=model,
                            base_url=base_url,
                            api_key=api_key,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            request_timeout=request_timeout,
                            tools=FIRST_TURN_TOOLS,
                            reasoning_effort=reasoning_effort,
                        )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Failed API call for dataset_index={record['dataset_index']} step={step}: {last_error}"
                        ) from exc
                    time.sleep(sleep_seconds)

            if use_responses_api:
                previous_response_id = response_body.get("id")
                tool_calls = ensure_tool_call_id(
                    extract_tool_calls_from_responses_output(response_body),
                    fallback_prefix=f"call_{record['dataset_index']}_{step}",
                )
            else:
                message = extract_first_message(response_body)
                tool_calls = ensure_tool_call_id(
                    extract_tool_calls_from_message(
                        message,
                        model=model,
                        available_actions=available_actions,
                    ),
                    fallback_prefix=f"call_{record['dataset_index']}_{step}",
                )
                message["tool_calls"] = tool_calls
                messages.append(
                    assistant_message_from_response(
                        message,
                        model=model,
                        available_actions=available_actions,
                    )
                )

            classification = classify_first_tool_call(tool_calls, available_actions)
            normalized_action = normalize_action_from_classification(classification)
            next_tool_outputs: List[Dict[str, str]] = []

            current_step = {
                "step": step,
                "available_actions": available_actions,
                "predicted_class": classification.get("predicted_class"),
                "predicted_tool_name": classification.get("predicted_tool_name"),
                "predicted_arguments": classification.get("predicted_arguments"),
                "invalid": classification.get("invalid"),
                "invalid_reason": classification.get("invalid_reason"),
                "invalid_arguments": classification.get("invalid_arguments"),
                "invalid_arguments_reason": classification.get("invalid_arguments_reason"),
                "action": normalized_action,
                "reward": 0,
                "done": False,
            }

            if step == 1:
                result["first_round"] = dict(current_step)

            if classification.get("predicted_class") in {"search", "click"} and not classification.get("invalid"):
                result["episode_actions"].append(classification["predicted_class"].upper())
            elif classification.get("predicted_class") == "abstain" and not classification.get("invalid"):
                result["episode_actions"].append("ABSTAIN")
                result["eventual_abstain"] = True
                result["first_abstain_step"] = step
                result["terminated_by"] = "abstain"
                current_step["done"] = True
                result["history"].append(current_step)
                break
            else:
                result["episode_actions"].append("INVALID")

            if normalized_action is None:
                observation = NO_TOOL_CALL_OBSERVATION
                if tool_calls:
                    predicted_tool_name = classification.get("predicted_tool_name") or "unknown_tool"
                    invalid_reason = classification.get("invalid_reason") or classification.get("invalid_arguments_reason")
                    primary_action_label = f"invalid_tool_call[{predicted_tool_name}]"
                    if isinstance(invalid_reason, str) and invalid_reason:
                        primary_action_label = f"{primary_action_label}: {invalid_reason}"
                    next_tool_outputs.append(
                        {
                            "tool_call_id": tool_calls[0]["id"],
                            "action": primary_action_label,
                        }
                    )
                    for extra_tool_call in tool_calls[1:]:
                        extra_name = (
                            extra_tool_call.get("function", {}).get("name")
                            or extra_tool_call.get("name")
                            or "unknown_tool"
                        )
                        next_tool_outputs.append(
                            {
                                "tool_call_id": extra_tool_call["id"],
                                "action": f"ignored_tool_call[{extra_name}]",
                            }
                        )
                previous_tool_outputs = next_tool_outputs
                result["history"].append(current_step)
                continue

            observation, reward, done, _ = env.step(normalized_action)
            current_step["reward"] = reward
            current_step["done"] = done
            result["reward"] = reward
            result["history"].append(current_step)
            next_tool_outputs = []
            if tool_calls:
                next_tool_outputs.append(
                    {
                        "tool_call_id": tool_calls[0]["id"],
                        "action": normalized_action,
                    }
                )
                for extra_tool_call in tool_calls[1:]:
                    extra_name = (
                        extra_tool_call.get("function", {}).get("name")
                        or extra_tool_call.get("name")
                        or "unknown_tool"
                    )
                    next_tool_outputs.append(
                        {
                            "tool_call_id": extra_tool_call["id"],
                            "action": f"ignored_tool_call[{extra_name}]",
                        }
                    )
            previous_tool_outputs = next_tool_outputs

            if done:
                result["terminated_by"] = "buy_now" if normalized_action.lower() == "click[buy now]" else "env_done"
                break
        else:
            result["terminated_by"] = "round_limit"

        result["rounds_completed"] = len(result["history"])
        finalize_episode_metrics(result)
        return result
    finally:
        try:
            server.user_sessions.pop(env.session, None)
            server.user_sessions.pop(initial_session_id, None)
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


def compute_split_metrics(records: List[Dict[str, Any]], *, max_k: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_total": len(records),
        "termination_counts": dict(Counter(record.get("terminated_by") for record in records)),
        "eventual_abstain_count": sum(1 for record in records if record.get("eventual_abstain")),
    }
    if not records:
        return summary

    warranted = [record for record in records if record.get("warranted_step") is not None]
    if not warranted:
        return summary

    timely_successes = sum(1 for record in warranted if record.get("timely_abstain"))
    successful_abstains = sum(1 for record in warranted if record.get("successful_abstain"))
    premature_abstains = sum(1 for record in warranted if record.get("premature_abstain"))
    spl_sum = 0.0
    pass_counts = {k: 0 for k in range(max_k + 1)}

    for record in warranted:
        oracle_steps = int(record["warranted_step"])
        if record.get("successful_abstain"):
            path_steps = int(record["first_abstain_step"])
            spl_sum += oracle_steps / max(path_steps, oracle_steps)
            delay = int(record["delay_after_warranted"])
            for k in range(max_k + 1):
                if delay <= k:
                    pass_counts[k] += 1

    summary.update(
        {
            "n_abstain_warranted": len(warranted),
            "timely_success_count": timely_successes,
            "successful_abstain_count": successful_abstains,
            "premature_abstain_count": premature_abstains,
            "timely_recall": timely_successes / len(warranted),
            "overall_recall": successful_abstains / len(warranted),
            "spl": spl_sum / len(warranted),
        }
    )
    for k in range(max_k + 1):
        summary[f"pass_at_{k}"] = pass_counts[k] / len(warranted)
    return summary


def compute_summary(sample_results: List[Dict[str, Any]], *, max_k: int) -> Dict[str, Any]:
    split_groups = {
        "original": [record for record in sample_results if record.get("variant_type") == "original"],
        "rewrite": [record for record in sample_results if record.get("variant_type") == "rewrite"],
        "missing_target": [record for record in sample_results if record.get("variant_type") == "missing_target"],
    }
    abstain_warranted = split_groups["rewrite"] + split_groups["missing_target"]
    overall = compute_split_metrics(abstain_warranted, max_k=max_k)

    summary = {
        "n_total": len(sample_results),
        "n_answerable": len(split_groups["original"]),
        "n_abstain_warranted": len(abstain_warranted),
        "n_original": len(split_groups["original"]),
        "n_rewrite": len(split_groups["rewrite"]),
        "n_missing_target": len(split_groups["missing_target"]),
        "timely_recall": overall.get("timely_recall", 0.0),
        "overall_recall": overall.get("overall_recall", 0.0),
        "spl": overall.get("spl", 0.0),
        "timely_success_count": overall.get("timely_success_count", 0),
        "successful_abstain_count": overall.get("successful_abstain_count", 0),
        "premature_abstain_count": overall.get("premature_abstain_count", 0),
        "by_split": {
            split_name: compute_split_metrics(records, max_k=max_k)
            for split_name, records in split_groups.items()
        },
    }
    summary["timely_recall_rewrite"] = summary["by_split"]["rewrite"].get("timely_recall", 0.0)
    summary["timely_recall_missing_target"] = summary["by_split"]["missing_target"].get("timely_recall", 0.0)
    summary["overall_recall_rewrite"] = summary["by_split"]["rewrite"].get("overall_recall", 0.0)
    summary["overall_recall_missing_target"] = summary["by_split"]["missing_target"].get("overall_recall", 0.0)
    summary["spl_rewrite"] = summary["by_split"]["rewrite"].get("spl", 0.0)
    summary["spl_missing_target"] = summary["by_split"]["missing_target"].get("spl", 0.0)
    for k in range(max_k + 1):
        summary[f"pass_at_{k}"] = overall.get(f"pass_at_{k}", 0.0)
        summary[f"pass_at_{k}_rewrite"] = summary["by_split"]["rewrite"].get(f"pass_at_{k}", 0.0)
        summary[f"pass_at_{k}_missing_target"] = summary["by_split"]["missing_target"].get(f"pass_at_{k}", 0.0)
    return summary


def process_variant_records(
    *,
    records: List[Dict[str, Any]],
    env_variant: str,
    utils_module,
    engine_module,
    env_module,
    items_path: Path,
    attr_path: Path,
    human_path: Path,
    search_root: Path,
    api_key: str,
    model: str,
    base_url: str,
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    sleep_seconds: float,
    round_limit: int,
    results_output: Path,
    full_pre_to_env: Dict[int, int],
    source_id_to_goal_index: Dict[str, int],
    pruned_bootstrap_goal_index: int,
    attempt_counts: Dict[int, int],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not records:
        return [], []

    server = bootstrap_server(
        env_module=env_module,
        utils_module=utils_module,
        engine_module=engine_module,
        items_path=items_path,
        attr_path=attr_path,
        human_path=human_path,
        search_root=search_root,
    )
    WebAgentTextEnv = env_module.WebAgentTextEnv

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        for offset, record in enumerate(records, start=1):
            dataset_index = int(record["dataset_index"])
            attempt = int(attempt_counts.get(dataset_index, 0)) + 1
            try:
                resolved_base_goal_index = resolve_source_goal_index(record, source_id_to_goal_index)
                reset_index = resolve_reset_index(
                    record,
                    env_variant=env_variant,
                    full_pre_to_env=full_pre_to_env,
                    source_id_to_goal_index=source_id_to_goal_index,
                    pruned_bootstrap_goal_index=pruned_bootstrap_goal_index,
                )
                result = evaluate_record(
                    record,
                    WebAgentTextEnv=WebAgentTextEnv,
                    server=server,
                    items_path=items_path,
                    reset_index=reset_index,
                    resolved_base_goal_index=resolved_base_goal_index,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                    attempt=attempt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    sleep_seconds=sleep_seconds,
                    round_limit=round_limit,
                )
                successes.append(result)
                append_jsonl(results_output, [result])
                print(
                    f"[{env_variant} {offset}/{len(records)}] dataset_index={record['dataset_index']} "
                    f"terminated_by={result['terminated_by']} rounds={result['rounds_completed']}"
                )
            except Exception as exc:
                error_message = str(exc)
                try:
                    resolved_base_goal_index = resolve_source_goal_index(record, source_id_to_goal_index)
                except Exception:
                    resolved_base_goal_index = None
                failure = build_failure_result(
                    record,
                    model=model,
                    base_url=base_url,
                    reasoning_effort=reasoning_effort,
                    attempt=attempt,
                    error_message=error_message,
                    resolved_base_goal_index=resolved_base_goal_index,
                )
                failures.append(failure)
                append_jsonl(results_output, [failure])
                print(
                    f"[{env_variant} {offset}/{len(records)}] dataset_index={record['dataset_index']} "
                    f"failed: {failure['error_type']} retryable={failure['retryable']} {error_message}"
                )
                if is_budget_stop_error(error_message):
                    raise RuntimeError(error_message) from exc
    finally:
        try:
            server.user_sessions.clear()
        except Exception:
            pass
        del server
        gc.collect()
    return successes, failures


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    webshop_root = Path(args.webshop_root).expanduser().resolve()
    pruned_metadata_path = Path(args.pruned_metadata).expanduser().resolve()
    full_goal_manifest_path = Path(args.full_goal_manifest).expanduser().resolve()

    results_output = (
        Path(args.results_output).expanduser().resolve()
        if args.results_output
        else default_results_output(input_path, args.model, args.reasoning_effort)
    )
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else default_summary_output(input_path, args.model, args.reasoning_effort)
    )

    java_home = resolve_existing_path(
        args.java_home,
        default_java_home_candidates(),
        "Could not find a usable JAVA_HOME. Pass --java-home explicitly.",
    )
    configure_java(java_home)

    input_records = load_jsonl(input_path)
    if args.count is not None:
        input_records = input_records[: args.count]

    completed = completed_dataset_indices(results_output) if args.resume else set()
    attempt_counts = attempt_counts_by_dataset_index(results_output) if args.resume else {}
    if results_output.exists() and not args.resume:
        raise RuntimeError(
            f"Output file already exists: {results_output}. Re-run with --resume or choose a different path."
        )

    pending = [record for record in input_records if int(record["dataset_index"]) not in completed]
    if not pending and results_output.exists():
        latest_rows = latest_result_rows(results_output)
        completed_rows = [row for row in latest_rows if row.get("status") == RESULT_STATUS_COMPLETED]
        status_counts = dict(Counter(row.get("status") for row in latest_rows))
        retryable_failures = sum(
            1 for row in latest_rows if row.get("status") == RESULT_STATUS_FAILED and row.get("retryable")
        )
        hard_failures = sum(
            1 for row in latest_rows if row.get("status") == RESULT_STATUS_FAILED and not row.get("retryable")
        )
        summary = compute_summary(completed_rows, max_k=args.pass_max_k)
        summary.update(
            {
                "input_path": str(input_path),
                "results_output": str(results_output),
                "summary_output": str(summary_output),
                "model": args.model,
                "base_url": args.base_url,
                "reasoning_effort": args.reasoning_effort,
                "temperature": args.temperature,
                "round_limit": args.round_limit,
                "n_input_records": len(input_records),
                "n_latest_rows": len(latest_rows),
                "n_completed_rows": len(completed_rows),
                "status_counts": status_counts,
                "retryable_failure_count": retryable_failures,
                "hard_failure_count": hard_failures,
                "complete_run": len(completed_rows) == len(input_records) and hard_failures == 0 and retryable_failures == 0,
                "completed_at": utc_now_iso(),
            }
        )
        ensure_parent_dir(summary_output)
        write_json(summary_output, summary)
        print(f"No pending rows. Refreshed summary at {summary_output}")
        return 0

    items_path = resolve_existing_path(
        args.items_path,
        default_full_items_candidates(),
        "Could not find full items_shuffle.json. Pass --items-path explicitly.",
    )
    attr_path = resolve_existing_path(
        args.attr_path,
        default_full_attr_candidates(),
        "Could not find full items_ins_v2.json. Pass --attr-path explicitly.",
    )
    human_path = resolve_existing_path(
        args.human_path,
        default_human_candidates(),
        "Could not find items_human_ins.json. Pass --human-path explicitly.",
    )
    pruned_metadata = load_json(pruned_metadata_path)
    if not isinstance(pruned_metadata, dict):
        raise ValueError(f"Expected JSON object in {pruned_metadata_path}")

    pruned_items_path = Path(pruned_metadata["pruned_items_path"]).expanduser().resolve()
    pruned_search_root = Path(pruned_metadata["pruned_index_dir"]).expanduser().resolve().parent
    pruned_bootstrap_goal_index = int(pruned_metadata.get("bootstrap_goal_index", 0))

    full_search_root = webshop_root / SEARCH_ROOT_RELATIVE
    api_key = resolve_api_key(args.api_key_env, env_file)
    utils_module, engine_module, env_module = configure_webshop_imports(webshop_root)

    full_goal_manifest = load_jsonl(full_goal_manifest_path)
    full_pre_to_env = build_shuffled_goal_index_map(len(full_goal_manifest))
    source_id_to_goal_index = build_source_id_to_goal_index_map(full_goal_manifest)

    variant_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in pending:
        variant_groups[record["env_variant"]].append(record)

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for env_variant in (FULL_ENV_VARIANT, PRUNED_ENV_VARIANT):
        records = variant_groups.get(env_variant, [])
        if not records:
            continue
        if env_variant == FULL_ENV_VARIANT:
            variant_successes, variant_failures = process_variant_records(
                records=records,
                env_variant=env_variant,
                utils_module=utils_module,
                engine_module=engine_module,
                env_module=env_module,
                items_path=items_path,
                attr_path=attr_path,
                human_path=human_path,
                search_root=full_search_root,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
                round_limit=args.round_limit,
                results_output=results_output,
                full_pre_to_env=full_pre_to_env,
                source_id_to_goal_index=source_id_to_goal_index,
                pruned_bootstrap_goal_index=pruned_bootstrap_goal_index,
                attempt_counts=attempt_counts,
            )
        else:
            variant_successes, variant_failures = process_variant_records(
                records=records,
                env_variant=env_variant,
                utils_module=utils_module,
                engine_module=engine_module,
                env_module=env_module,
                items_path=pruned_items_path,
                attr_path=attr_path,
                human_path=human_path,
                search_root=pruned_search_root,
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
                round_limit=args.round_limit,
                results_output=results_output,
                full_pre_to_env=full_pre_to_env,
                source_id_to_goal_index=source_id_to_goal_index,
                pruned_bootstrap_goal_index=pruned_bootstrap_goal_index,
                attempt_counts=attempt_counts,
            )
        successes.extend(variant_successes)
        failures.extend(variant_failures)

    latest_rows = latest_result_rows(results_output) if results_output.exists() else []
    completed_rows = [row for row in latest_rows if row.get("status") == RESULT_STATUS_COMPLETED]
    status_counts = dict(Counter(row.get("status") for row in latest_rows))
    retryable_failures = sum(
        1 for row in latest_rows if row.get("status") == RESULT_STATUS_FAILED and row.get("retryable")
    )
    hard_failures = sum(
        1 for row in latest_rows if row.get("status") == RESULT_STATUS_FAILED and not row.get("retryable")
    )
    summary = compute_summary(completed_rows, max_k=args.pass_max_k)
    summary.update(
        {
            "input_path": str(input_path),
            "results_output": str(results_output),
            "summary_output": str(summary_output),
            "model": args.model,
            "base_url": args.base_url,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "round_limit": args.round_limit,
            "n_input_records": len(input_records),
            "n_latest_rows": len(latest_rows),
            "n_completed_rows": len(completed_rows),
            "status_counts": status_counts,
            "retryable_failure_count": retryable_failures,
            "hard_failure_count": hard_failures,
            "complete_run": len(completed_rows) == len(input_records) and hard_failures == 0 and retryable_failures == 0,
            "completed_at": utc_now_iso(),
        }
    )
    ensure_parent_dir(summary_output)
    write_json(summary_output, summary)

    print(f"Wrote {len(successes)} completed rows to {results_output}")
    print(f"Wrote summary to {summary_output}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(
                f"  - dataset_index={failure['dataset_index']} "
                f"error_type={failure['error_type']} retryable={failure['retryable']} "
                f"message={failure['error_message']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
