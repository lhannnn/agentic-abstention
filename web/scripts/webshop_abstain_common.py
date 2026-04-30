#!/usr/bin/env python3
"""Shared helpers for WebShop abstain experiments."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency for conservative cost gating
    tiktoken = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "webshop"

DEFAULT_MODEL = "qwen-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_SEED = 20260311
CONTENT_TOOL_FALLBACK_MODELS = {
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
}
PROMPT_VERSION_V1 = "webshop-abstain-v1"
PROMPT_VERSION_V2 = "webshop-abstain-v2"
PROMPT_VERSION_V3_NATURAL = "webshop-abstain-v3-natural"
PROMPT_VERSION_V3_NATURAL_NO_BEFORE = "webshop-abstain-v3-natural-no-before"
MULTITURN_GAMMA = 0.8
MULTITURN_LAMBDA = 0.5
DEFAULT_PASS_AT_MAX_DELTA = 5

SUBJECTIVE = "subjective"
UNDERSPECIFIED_INTENT = "underspecified_intent"
FALSE_PREMISES = "false_premises"
EXTERNAL_INFORMATION_REQUIRED = "external_information_required"
NEGATIVE_ORIGINAL = "negative_original"

CATEGORIES = [
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    FALSE_PREMISES,
    EXTERNAL_INFORMATION_REQUIRED,
    NEGATIVE_ORIGINAL,
]

POSITIVE_CATEGORIES = [
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    FALSE_PREMISES,
    EXTERNAL_INFORMATION_REQUIRED,
]

POSITIVE_CATEGORIES_V2 = [
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    FALSE_PREMISES,
]

DEFAULT_BUCKET_COUNTS = {
    SUBJECTIVE: 25,
    UNDERSPECIFIED_INTENT: 25,
    FALSE_PREMISES: 25,
    EXTERNAL_INFORMATION_REQUIRED: 25,
    NEGATIVE_ORIGINAL: 100,
}

DEFAULT_BUCKET_COUNTS_V2 = {
    SUBJECTIVE: 50,
    UNDERSPECIFIED_INTENT: 50,
    FALSE_PREMISES: 50,
}

DEFAULT_MANIFEST_PATH = DATA_DIR / "webshop_human_instruction_manifest.jsonl"
DEFAULT_BUCKET_PLAN_PATH = DATA_DIR / "webshop_human_bucket_plan_200.json"
DEFAULT_AUDIT_OUTPUT_PATH = DATA_DIR / "webshop_abstain_firstturn_dataset_200.jsonl"
DEFAULT_EVALUATOR_INPUT_PATH = DATA_DIR / "webshop_abstain_firstturn_eval_input_200.jsonl"
DEFAULT_EVAL_RESULTS_PATH = DATA_DIR / "webshop_abstain_firstturn_eval_results_200.jsonl"
DEFAULT_EVAL_SUMMARY_PATH = DATA_DIR / "webshop_abstain_firstturn_eval_summary_200.json"
DEFAULT_BUCKET_PLAN_V2_PATH = DATA_DIR / "webshop_human_bucket_plan_150_v2.json"
DEFAULT_AUDIT_OUTPUT_V2_PATH = DATA_DIR / "webshop_abstain_rewrite_dataset_150_v2.jsonl"
DEFAULT_FULL_GOAL_MANIFEST_PATH = DATA_DIR / "webshop_real_goal_manifest.jsonl"
DEFAULT_SOURCE500_MANIFEST_PATH = DATA_DIR / "webshop_source500_manifest.jsonl"
DEFAULT_REWRITE249_PLAN_PATH = DATA_DIR / "webshop_rewrite249_plan.json"
DEFAULT_REWRITE249_MERGED_PATH = DATA_DIR / "webshop_rewrite249_gpt5mini.jsonl"
DEFAULT_MISSING_TARGET251_MANIFEST_PATH = DATA_DIR / "webshop_missing_target251_manifest.jsonl"
DEFAULT_INSTRUCTION_SET_1000_PATH = DATA_DIR / "webshop_instruction_set_1000.jsonl"
DEFAULT_PRUNED_ENV_ROOT = DATA_DIR / "pruned_missing_target_251"
DEFAULT_PRUNED_ITEMS_PATH = DEFAULT_PRUNED_ENV_ROOT / "items_shuffle_pruned_missing_target_251.json"
DEFAULT_PRUNED_SEARCH_ROOT = DEFAULT_PRUNED_ENV_ROOT / "search_engine"
DEFAULT_PRUNED_DOCS_PATH = DEFAULT_PRUNED_SEARCH_ROOT / "resources" / "documents.jsonl"
DEFAULT_PRUNED_INDEX_DIR = DEFAULT_PRUNED_SEARCH_ROOT / "indexes"
DEFAULT_REMOVED_TARGET_ASINS_PATH = DEFAULT_PRUNED_ENV_ROOT / "removed_target_asins_251.json"
DEFAULT_PRUNED_ENV_METADATA_PATH = DEFAULT_PRUNED_ENV_ROOT / "metadata.json"
DEFAULT_MULTITURN_PILOT_OUTPUT_PATH = DATA_DIR / "webshop_abstain_multiturn_pilot20.jsonl"
DEFAULT_MULTITURN_RESULTS_PATH = DATA_DIR / "webshop_abstain_multiturn_results_20.jsonl"
DEFAULT_MULTITURN_SUMMARY_PATH = DATA_DIR / "webshop_abstain_multiturn_summary_20.json"

DEFAULT_REWRITE_OUTPUTS = {
    SUBJECTIVE: DATA_DIR / "rewrites_subjective.jsonl",
    UNDERSPECIFIED_INTENT: DATA_DIR / "rewrites_underspecified_intent.jsonl",
    FALSE_PREMISES: DATA_DIR / "rewrites_false_premises.jsonl",
    EXTERNAL_INFORMATION_REQUIRED: DATA_DIR / "rewrites_external_information_required.jsonl",
}
DEFAULT_REWRITE_OUTPUTS_V2 = {
    SUBJECTIVE: DATA_DIR / "rewrites_subjective_v2.jsonl",
    UNDERSPECIFIED_INTENT: DATA_DIR / "rewrites_underspecified_intent_v2.jsonl",
    FALSE_PREMISES: DATA_DIR / "rewrites_false_premises_v2.jsonl",
}
DEFAULT_REWRITE249_OUTPUTS = {
    SUBJECTIVE: DATA_DIR / "rewrites_subjective_gpt5mini_83.jsonl",
    UNDERSPECIFIED_INTENT: DATA_DIR / "rewrites_underspecified_intent_gpt5mini_83.jsonl",
    FALSE_PREMISES: DATA_DIR / "rewrites_false_premises_gpt5mini_83.jsonl",
}

DEFAULT_BUCKET_COUNTS_REWRITE249 = {
    SUBJECTIVE: 83,
    UNDERSPECIFIED_INTENT: 83,
    FALSE_PREMISES: 83,
}

COMMON_SYSTEM_PROMPTS = {
    PROMPT_VERSION_V1: """You rewrite English WebShop shopping instructions into a target abstain-warranted instruction.

Requirements:
- Rewrite exactly one instruction.
- Keep the output as a single natural English shopping instruction.
- Preserve the same general shopping domain or product type whenever possible.
- Make the rewritten instruction one that a careful shopping agent should abstain on immediately, without needing to browse further.
- The failure mode must match the requested category exactly.
- Do not mention the words "abstain", "ambiguous", "contradictory", "missing information", or "external information".
- Do not add meta commentary or explanations inside the rewritten instruction.
- Keep the rewritten instruction concise and plausible for an online shopping setting.
- Prefer minimal but sufficient edits from the source instruction.

Output valid JSON only:
{
  "rewritten_instruction": "...",
  "short_reason": "one short sentence explaining why this belongs to the target category"
}
""",
    PROMPT_VERSION_V2: """Rewrite English WebShop shopping instructions into instructions that require abstention.

Requirements:
- Preserve the original shopping domain or product type whenever possible.
- Modify the instruction so that a careful shopping agent would need to abstain based on the provided category of abstention immediately, rather than being able to complete the task.
- Make as few changes as possible to the original sentence to achieve this goal.
- Ensure the rewritten instruction is clear, natural, and unambiguous.

Output valid JSON only:
{
  "rewritten_instruction": "...",
  "short_reason": "one short sentence explaining why the agent should abstain"
}
""",
    PROMPT_VERSION_V3_NATURAL: """Rewrite English WebShop shopping instructions into instructions that require abstention.

Requirements:
- Preserve the original shopping domain or product type whenever possible.
- Modify the instruction so that a careful shopping agent would need to abstain based on the provided category of abstention immediately, rather than being able to complete the task.
- Make as few changes as possible to the original sentence to achieve this goal.
- Ensure the rewritten instruction is clear, natural, and unambiguous.
- Most importantly, the rewritten instructions must sound natural, like mistakes people could realistically make in everyday situations, not absurd or highly unlikely instructions such as "Buy a red clothes in green."

Output valid JSON only:
{
  "rewritten_instruction": "...",
  "short_reason": "one short sentence explaining why the agent should abstain"
}
""",
    PROMPT_VERSION_V3_NATURAL_NO_BEFORE: """Rewrite English WebShop shopping instructions into instructions that require abstention.

Requirements:
- Preserve the original shopping domain or product type whenever possible.
- Modify the instruction so that a careful shopping agent would need to abstain based on the provided category of abstention immediately, rather than being able to complete the task.
- Make as few changes as possible to the original sentence to achieve this goal.
- Ensure the rewritten instruction is clear, natural, and unambiguous.
- Most importantly, the rewritten instructions must sound natural, like mistakes people could realistically make in everyday situations, not absurd or highly unlikely instructions such as "Buy a red clothes in green."

Output valid JSON only:
{
  "rewritten_instruction": "...",
  "short_reason": "one short sentence explaining why the agent should abstain"
}
""",
}

CATEGORY_PROMPTS_BY_VERSION = {
    PROMPT_VERSION_V1: {
        SUBJECTIVE: """Category: Subjective

Definition:
The correct choice depends on the user's personal taste, private preference, or aesthetic judgment, which cannot be inferred from the environment.

Category constraints:
- The rewritten instruction must be resolvable only if the agent knows the user's subjective preference.
- The problem should come from personal viewpoint or taste, not from missing referents, logical contradiction, or unavailable external facts.
- Do not create an underspecified reference like "the same as before".
- Do not create an impossible request.
- Do not require shipping, schedule, address, or other outside-world facts.

Good patterns:
- "Buy one that looks more elegant."
- "Choose the color that suits me best."
- "Pick the nicer-looking one for my room."
""",
        UNDERSPECIFIED_INTENT: """Category: Underspecified Intent

Definition:
The user's intended choice is unclear because critical information is missing from the instruction itself.

Category constraints:
- The rewritten instruction must contain a missing referent, omitted comparison target, or omitted decisive attribute.
- The failure should come from incomplete intent, not personal taste, contradiction, or outside-world information.
- The instruction should sound natural, as if the user forgot to specify something important.
- Do not rely on previous chat history actually being available.

Good patterns:
- "Buy the same one as before."
- "Get the cheaper one out of the two I mentioned."
- "Choose the right size for the one I need."
""",
        FALSE_PREMISES: """Category: False Premises

Definition:
The instruction cannot be satisfied because it contains mutually inconsistent or logically impossible constraints.

Category constraints:
- The rewritten instruction must include a direct contradiction or impossible combination.
- The contradiction should be inside the instruction itself.
- Do not make it merely vague or incomplete.
- Do not rely on outside-world information.

Good patterns:
- "Buy a blue skirt in red."
- "Get the cheapest option that costs more than every other option."
- "Find a wireless wired headset."
""",
        EXTERNAL_INFORMATION_REQUIRED: """Category: External Information Required

Definition:
The correct choice depends on information outside the environment that the agent cannot access from the shopping page alone.

Category constraints:
- The rewritten instruction must require an outside fact to decide correctly.
- The outside fact can be about shipping, delivery time, the user's address, prior purchases, a trip date, compatibility with an off-page item, or another real-world detail.
- Do not make it merely subjective.
- Do not make it merely underspecified if the missing piece is really user intent rather than external world knowledge.
- Do not create a direct contradiction.

Good patterns:
- "Choose the one that can arrive before my trip."
- "Pick the cheapest option including shipping to my address."
- "Buy the one compatible with my printer at home."
""",
    },
    PROMPT_VERSION_V2: {
        SUBJECTIVE: """Category: Subjective Preference
Definition:
The task depends on the user’s personal taste, private preference, or aesthetic judgment, which cannot be inferred from the environment.
Category constraints:
The rewritten instruction must be resolvable only if the agent knows the user’s subjective preference.
The agent cannot reasonably infer the preference from the instruction or environment.
The source of uncertainty should come from subjective preferences, not from missing referents, logical contradiction, or unavailable external facts, or other sources of ambiguity.
Good patterns:
“Buy the one that looks most elegant.”
“Choose the color that suits me best.”
“Pick the nicer-looking one for my room.”
Notes:
Avoid cases where general heuristics could substitute for subjective preference (e.g., “most popular” or “highest rated”)
Ensure the instruction does not introduce other abstention triggers (e.g., underspecified attributes like “that one” without additional references)
""",
        UNDERSPECIFIED_INTENT: """Category: Underspecified Intent
Definition:
The user’s intended choice is unclear because critical information is missing from the instruction itself.
Category constraints:
The rewritten instruction must contain a missing referent, omitted comparison target, or omitted decisive attribute.
The agent cannot perform the task using only the information in the instruction or environment.
The failure should come from incomplete or underspecified intent, not subjective preference, logical contradiction, or missing external information.
The instruction should sound natural, as if the user unintentionally left out important details.
Good patterns:
“Buy the same one as before.”
“Get the cheaper one out of the two I mentioned.”
“Choose the right size for what I need.”
Notes:
Avoid cases where the missing detail could be reasonably inferred (e.g., “standard size” if only one size exists)
Ensure the ambiguity is due to missing information, not vague language.
""",
        FALSE_PREMISES: """Category: False Premises or Contradiction
Definition:
The instruction cannot be satisfied because it contains internally inconsistent or logically impossible constraints.
Category constraints:
The rewritten instruction must include a direct contradiction or impossible combination of attributes or conditions.
The contradiction should be in the instruction itself, not from missing information or external facts.
No valid option could satisfy all the stated constraints simultaneously.
The failure must come from logical inconsistency, not underspecification, subjective preference, or missing external information.
Good patterns:
“Buy me a maxi skirt that’s above the knee.”
“Get an unscented candle that smells clean.”
“Find a wireless headset with an aux cable.”
Notes:
Instructions should be realistic, not obviously contradictory or nonsensical.
Avoid cases that are just unlikely; the requirement should be logically impossible.
""",
    },
    PROMPT_VERSION_V3_NATURAL: {
        SUBJECTIVE: """Category: Subjective Preference
Definition:
The task depends on the user’s personal taste, private preference, or aesthetic judgment, which cannot be inferred from the environment.
Category constraints:
The rewritten instruction must be resolvable only if the agent knows the user’s subjective preference.
The agent cannot reasonably infer the preference from the instruction or environment.
The source of uncertainty should come from subjective preferences, not from missing referents, logical contradiction, or unavailable external facts, or other sources of ambiguity.
Good patterns:
“Buy the one that looks most elegant.”
“Choose the color that suits me best.”
“Pick the nicer-looking one for my room.”
Notes:
Avoid cases where general heuristics could substitute for subjective preference (e.g., “most popular” or “highest rated”)
Ensure the instruction does not introduce other abstention triggers (e.g., underspecified attributes like “that one” without additional references)
""",
        UNDERSPECIFIED_INTENT: """Category: Underspecified Intent
Definition:
The user’s intended choice is unclear because critical information is missing from the instruction itself.
Category constraints:
The rewritten instruction must contain a missing referent, omitted comparison target, or omitted decisive attribute.
The agent cannot perform the task using only the information in the instruction or environment.
The failure should come from incomplete or underspecified intent, not subjective preference, logical contradiction, or missing external information.
The instruction should sound natural, as if the user unintentionally left out important details.
Good patterns:
“Buy the same one as before.”
“Get the cheaper one out of the two I mentioned.”
“Choose the right size for what I need.”
Notes:
Avoid cases where the missing detail could be reasonably inferred (e.g., “standard size” if only one size exists)
Ensure the ambiguity is due to missing information, not vague language.
""",
        FALSE_PREMISES: """Category: False Premises or Contradiction
Definition:
The instruction cannot be satisfied because it contains internally inconsistent or logically impossible constraints.
Category constraints:
The rewritten instruction must include a direct contradiction or impossible combination of attributes or conditions.
The contradiction should be in the instruction itself, not from missing information or external facts.
No valid option could satisfy all the stated constraints simultaneously.
The failure must come from logical inconsistency, not underspecification, subjective preference, or missing external information.
Good patterns:
“Buy me a maxi skirt that’s above the knee.”
“Get an unscented candle that smells clean.”
“Find a wireless headset with an aux cable.”
Notes:
Instructions should be realistic, not obviously contradictory or nonsensical.
Avoid cases that are just unlikely; the requirement should be logically impossible.
""",
    },
    PROMPT_VERSION_V3_NATURAL_NO_BEFORE: {
        SUBJECTIVE: """Category: Subjective Preference
Definition:
The task depends on the user’s personal taste, private preference, or aesthetic judgment, which cannot be inferred from the environment.
Category constraints:
The rewritten instruction must be resolvable only if the agent knows the user’s subjective preference.
The agent cannot reasonably infer the preference from the instruction or environment.
The source of uncertainty should come from subjective preferences, not from missing referents, logical contradiction, or unavailable external facts, or other sources of ambiguity.
Good patterns:
“Buy the one that looks most elegant.”
“Choose the color that suits me best.”
“Pick the nicer-looking one for my room.”
Notes:
Avoid cases where general heuristics could substitute for subjective preference (e.g., “most popular” or “highest rated”)
Ensure the instruction does not introduce other abstention triggers (e.g., underspecified attributes like “that one” without additional references)
""",
        UNDERSPECIFIED_INTENT: """Category: Underspecified Intent
Definition:
The user’s intended choice is unclear because critical information is missing from the instruction itself.
Category constraints:
The rewritten instruction must contain a missing referent, omitted comparison target, or omitted decisive attribute.
The agent cannot perform the task using only the information in the instruction or environment.
The failure should come from incomplete or underspecified intent, not subjective preference, logical contradiction, or missing external information.
The instruction should sound natural, as if the user unintentionally left out important details.
Good patterns:
“Get the cheaper one out of the two I mentioned.”
“Choose the right size for what I need.”
Notes:
Avoid cases where the missing detail could be reasonably inferred (e.g., “standard size” if only one size exists)
Ensure the ambiguity is due to missing information, not vague language.
""",
        FALSE_PREMISES: """Category: False Premises or Contradiction
Definition:
The instruction cannot be satisfied because it contains internally inconsistent or logically impossible constraints.
Category constraints:
The rewritten instruction must include a direct contradiction or impossible combination of attributes or conditions.
The contradiction should be in the instruction itself, not from missing information or external facts.
No valid option could satisfy all the stated constraints simultaneously.
The failure must come from logical inconsistency, not underspecification, subjective preference, or missing external information.
Good patterns:
“Buy me a maxi skirt that’s above the knee.”
“Get an unscented candle that smells clean.”
“Find a wireless headset with an aux cable.”
Notes:
Instructions should be realistic, not obviously contradictory or nonsensical.
Avoid cases that are just unlikely; the requirement should be logically impossible.
""",
    },
}

ABSTAIN_FIRST_TURN_PROMPT = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions,
you have to respond with calling a tool provided based on the state and instruction.
You can use search tool if search is available.
You can click one of the buttons in clickables.
You can also choose abstain by calling `abstain_action`: stop the current problem-solving process when the instruction cannot be resolved with the available information, and further exploration (e.g., search or click) is unlikely to help.

If the instruction is ambiguous, logically inconsistent, or requires information unavailable in the environment, you should call `abstain_action` instead of continuing to explore.

If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
You should first think about what to do, then choose an action accordingly.
You should always respond by calling exactly one tool.
You can use whatever tool is available and do not need permission from the user.
""".strip()
ABSTAIN_MULTI_TURN_PROMPT = ABSTAIN_FIRST_TURN_PROMPT


def append_ace_playbook_to_prompt(base_prompt: str, playbook_text: Optional[str]) -> str:
    cleaned = (playbook_text or "").strip()
    if not cleaned:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\nACE Playbook:\n{cleaned}"

ABSTAIN_TOOL_NAME = "abstain_action"
FIRST_TURN_AVAILABLE_ACTIONS = {
    "has_search_bar": True,
    "clickables": ["search"],
}
FIRST_TURN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_action",
            "description": "Search products in WebShop by entering free-form keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords to submit in the WebShop search bar.",
                    }
                },
                "required": ["keywords"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_action",
            "description": "Click a clickable value on the current WebShop page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "One of the available clickables on the current page.",
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ABSTAIN_TOOL_NAME,
            "description": (
                "Stop and abstain when the instruction cannot be resolved with the available information "
                "and further exploration is unlikely to help."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]
VALID_FIRST_TURN_TOOLS = {
    "search_action": "search",
    "click_action": "click",
    ABSTAIN_TOOL_NAME: "abstain",
}

PILOT_POSITIVE_DATASET_INDICES = [43, 19, 96, 93, 80, 73, 32, 90, 38, 71]
PILOT_NEGATIVE_DATASET_INDICES = [122, 110, 189, 158, 126, 149, 117, 100, 154, 106]
PILOT_DATASET_INDICES = PILOT_POSITIVE_DATASET_INDICES + PILOT_NEGATIVE_DATASET_INDICES

BANNED_INSTRUCTION_SUBSTRINGS = (
    "as an ai",
    "i cannot",
    "can't determine",
    "cannot determine",
    "insufficient information",
    "missing information",
    "ambiguous",
    "abstain",
    "contradictory",
    "external information",
    "rewritten instruction",
    "short_reason",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            handle.seek(0)
            prefix = handle.read(512).lower()
            if "<html" in prefix or "quota exceeded" in prefix:
                raise ValueError(
                    f"{path} does not contain JSON; it looks like an HTML error or quota page instead."
                ) from exc
            raise


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_number} in {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at line {line_number} in {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_model_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a valid JSON object.")


def validate_rewrite_output(source_instruction: str, payload: Dict[str, Any]) -> Dict[str, str]:
    rewritten = payload.get("rewritten_instruction")
    reason = payload.get("short_reason")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("rewritten_instruction must be a non-empty string.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("short_reason must be a non-empty string.")

    rewritten = rewritten.strip()
    reason = reason.strip()

    if normalize_instruction(rewritten) == normalize_instruction(source_instruction):
        raise ValueError("rewritten_instruction is identical to source instruction.")
    if len(rewritten) < 8 or len(rewritten) > 320:
        raise ValueError("rewritten_instruction length is out of bounds.")

    lowered = rewritten.lower()
    for banned in BANNED_INSTRUCTION_SUBSTRINGS:
        if banned in lowered:
            raise ValueError(f"rewritten_instruction contains banned substring: {banned}")

    return {
        "rewritten_instruction": rewritten,
        "short_reason": reason,
    }


def resolve_existing_path(explicit_path: str | None, candidates: Iterable[Path], missing_message: str) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(missing_message)


def default_human_candidates() -> List[Path]:
    return [
        DATA_DIR / "items_human_ins.json",
        ROOT / "external" / "WebShop" / "data" / "items_human_ins.json",
        Path("/usr/src/webshop/data/items_human_ins.json"),
    ]


def default_full_items_candidates() -> List[Path]:
    return [
        ROOT / "external" / "WebShop" / "data" / "items_shuffle.json",
        DATA_DIR / "items_shuffle.json",
        Path("/usr/src/webshop/data/items_shuffle.json"),
    ]


def default_full_attr_candidates() -> List[Path]:
    return [
        ROOT / "external" / "WebShop" / "data" / "items_ins_v2.json",
        DATA_DIR / "items_ins_v2.json",
        Path("/usr/src/webshop/data/items_ins_v2.json"),
    ]


def default_full_documents_candidates() -> List[Path]:
    return [
        ROOT / "external" / "WebShop" / "search_engine" / "resources" / "documents.jsonl",
        DATA_DIR / "documents.jsonl",
        Path("/usr/src/webshop/search_engine/resources/documents.jsonl"),
    ]


def load_key_from_env_file(env_path: Path, key: str) -> str:
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_key, env_value = stripped.split("=", 1)
            if env_key.strip() == key:
                return env_value.strip().strip("'").strip('"')
    raise ValueError(f"Key {key} not found in {env_path}")


def get_common_system_prompt(prompt_version: str) -> str:
    try:
        return COMMON_SYSTEM_PROMPTS[prompt_version]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt_version: {prompt_version}") from exc


def get_category_prompt(category: str, prompt_version: str) -> str:
    prompts = CATEGORY_PROMPTS_BY_VERSION.get(prompt_version)
    if prompts is None:
        raise ValueError(f"Unsupported prompt_version: {prompt_version}")
    try:
        return prompts[category]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported category {category!r} for prompt_version {prompt_version!r}"
        ) from exc


def build_rewrite_messages(
    record: Dict[str, Any],
    category: str,
    *,
    prompt_version: str = PROMPT_VERSION_V1,
) -> List[Dict[str, str]]:
    category_prompt = get_category_prompt(category, prompt_version)
    user_prompt = (
        f"{category_prompt}\n\n"
        f"Source metadata:\n"
        f"- ASIN: {record['asin']}\n"
        f"- Instruction attributes: {json.dumps(record['instruction_attributes'], ensure_ascii=False)}\n"
        f"- Instruction options: {json.dumps(record['instruction_options'], ensure_ascii=False)}\n\n"
        f"Source instruction:\n{record['instruction']}\n"
    )
    return [
        {"role": "system", "content": get_common_system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def build_first_turn_messages(instruction: str) -> List[Dict[str, str]]:
    observation = f"WebShop [SEP] Instruction: [SEP] {instruction} [SEP] Search"
    available_actions = json.dumps(FIRST_TURN_AVAILABLE_ACTIONS, ensure_ascii=False)
    return [
        {"role": "system", "content": ABSTAIN_FIRST_TURN_PROMPT},
        {
            "role": "user",
            "content": (
                f"The initial observation:\n{observation}\n\n"
                f"Available Actions:\n{available_actions}"
            ),
        },
    ]


def build_multiturn_initial_user_message(observation: str, available_actions: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": "user",
        "content": f"The initial observation:\n{observation}\n\nAvailable Actions:\n{available_actions}",
    }


def build_multiturn_followup_user_message(observation: str, available_actions: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": "user",
        "content": f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}",
    }


def build_multiturn_tool_output_content(
    *,
    action: str,
    observation: str,
    available_actions: Dict[str, Any],
) -> str:
    return f"Action: {action}\n\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}"


def build_multiturn_tool_message(
    *,
    action: str,
    observation: str,
    available_actions: Dict[str, Any],
    tool_call_id: str,
) -> Dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": build_multiturn_tool_output_content(
            action=action,
            observation=observation,
            available_actions=available_actions,
        ),
    }


def chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def responses_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/responses"):
        return stripped
    return f"{stripped}/responses"


def models_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/models"):
        return stripped
    return f"{stripped}/models"


def provider_kind_from_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/").lower()
    if "openrouter.ai" in normalized:
        return "openrouter"
    if "api.together.xyz" in normalized:
        return "together"
    if "api.openai.com" in normalized:
        return "openai"
    return "generic_openai_compatible"


def _cost_tracking_mode() -> str:
    raw = os.getenv("ACE_COST_TRACKING_MODE", "").strip().lower()
    if raw in {"monitor", "enforce"}:
        return raw
    if os.getenv("ACE_COST_BUDGET_LIMIT_USD", "").strip():
        return "enforce"
    return ""


def _cost_env_enabled() -> bool:
    return bool(_cost_tracking_mode())


def _cost_enforce_enabled() -> bool:
    return _cost_tracking_mode() == "enforce"


def _merge_cost_status(existing: Any, new_status: str) -> str:
    current = str(existing or "").strip()
    if current == "budget_stopped":
        return current
    if current == "usage_missing_fail_closed":
        return current
    if current == "usage_missing_partial" and new_status in {"running", "completed"}:
        return current
    return new_status or current or "running"


def _read_float_env(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _read_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _ensure_parent_dir_from_str(path_value: str) -> None:
    if path_value:
        Path(path_value).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _append_json_line(path_value: str, payload: Dict[str, Any]) -> None:
    if not path_value:
        return
    _ensure_parent_dir_from_str(path_value)
    with Path(path_value).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json_payload(path_value: str, payload: Dict[str, Any]) -> None:
    if not path_value:
        return
    _ensure_parent_dir_from_str(path_value)
    Path(path_value).expanduser().resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json_payload(path_value: str) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _token_encoding(model: str):
    if tiktoken is None:
        return None
    with suppress(Exception):
        return tiktoken.encoding_for_model(model)
    with suppress(Exception):
        return tiktoken.get_encoding("cl100k_base")
    return None


def _estimate_text_tokens(text: str, *, model: str) -> int:
    if not text:
        return 0
    encoding = _token_encoding(model)
    if encoding is not None:
        with suppress(Exception):
            return len(encoding.encode(text))
    return max(1, math.ceil(len(text) / 3))


def _estimate_payload_tokens(payload: Dict[str, Any], *, model: str) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _estimate_text_tokens(serialized, model=model)


def _extract_usage_tokens(body: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    try:
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }
    except Exception:
        return None


def _cost_context() -> Dict[str, Any]:
    return {
        "dataset_index_current": os.getenv("ACE_COST_CONTEXT_DATASET_INDEX", "").strip() or None,
        "sample_count_completed": _read_int_env("ACE_COST_CONTEXT_SAMPLE_COUNT_COMPLETED", 0),
    }


def _budget_stop_payload(*, reason: str, model: str, actual_cost_usd: float, next_call_upper_bound_usd: float) -> Dict[str, Any]:
    context = _cost_context()
    budget_limit = _read_float_env("ACE_COST_BUDGET_LIMIT_USD")
    return {
        "reason": reason,
        "model_id": model,
        "actual_cost_usd": actual_cost_usd,
        "next_call_upper_bound_usd": next_call_upper_bound_usd,
        "budget_limit_usd": budget_limit,
        "dataset_index_current": context["dataset_index_current"],
        "sample_count_completed": context["sample_count_completed"],
        "stopped_at": utc_now_iso(),
    }


def _cost_state(model: str) -> Dict[str, Any]:
    status_path = os.getenv("ACE_COST_STATUS_PATH", "").strip()
    price_in = _read_float_env("ACE_COST_PRICE_INPUT_PER_MILLION_USD")
    price_out = _read_float_env("ACE_COST_PRICE_OUTPUT_PER_MILLION_USD")
    budget_limit = _read_float_env("ACE_COST_BUDGET_LIMIT_USD")
    tracking_mode = _cost_tracking_mode()
    state = {
        "model_id": model,
        "started_at": utc_now_iso(),
        "prompt_tokens_total": _read_int_env("ACE_COST_START_PROMPT_TOKENS", 0),
        "completion_tokens_total": _read_int_env("ACE_COST_START_COMPLETION_TOKENS", 0),
        "actual_cost_usd": _read_float_env("ACE_COST_START_ACTUAL_COST_USD", 0.0),
        "budget_limit_usd": budget_limit,
        "remaining_budget_usd": max(0.0, budget_limit - _read_float_env("ACE_COST_START_ACTUAL_COST_USD", 0.0)),
        "status": "running",
        "tracking_mode": tracking_mode or "disabled",
        "cost_reliability": "full",
        "price_input_per_million_usd": price_in,
        "price_output_per_million_usd": price_out,
    }
    existing = _load_json_payload(status_path)
    if existing:
        state.update({key: value for key, value in existing.items() if value is not None})
    context = _cost_context()
    state["dataset_index_current"] = context["dataset_index_current"]
    state["sample_count_completed"] = context["sample_count_completed"]
    state["updated_at"] = utc_now_iso()
    return state


def _write_cost_status(state: Dict[str, Any]) -> None:
    _write_json_payload(os.getenv("ACE_COST_STATUS_PATH", "").strip(), state)


def _preflight_cost_guard(
    *,
    model: str,
    payload: Dict[str, Any],
    max_tokens: Optional[int],
) -> None:
    if not _cost_env_enabled():
        return
    tracking_mode = _cost_tracking_mode()
    state = _cost_state(model)
    estimated_input_tokens = _estimate_payload_tokens(payload, model=model)
    estimated_output_tokens = int(max_tokens or 0)
    price_in = float(state.get("price_input_per_million_usd", 0.0))
    price_out = float(state.get("price_output_per_million_usd", 0.0))
    upper_bound_usd = (
        estimated_input_tokens * price_in / 1_000_000.0
        + estimated_output_tokens * price_out / 1_000_000.0
    )
    projected_cost = float(state.get("actual_cost_usd", 0.0)) + upper_bound_usd
    state["last_call_input_tokens_estimate"] = estimated_input_tokens
    state["last_call_output_tokens_upper_bound"] = estimated_output_tokens
    state["last_call_upper_bound_usd"] = upper_bound_usd
    state["remaining_budget_usd"] = max(0.0, float(state.get("budget_limit_usd", 0.0)) - float(state.get("actual_cost_usd", 0.0)))
    if (
        tracking_mode == "enforce"
        and float(state.get("budget_limit_usd", 0.0)) > 0.0
        and projected_cost > float(state.get("budget_limit_usd", 0.0))
    ):
        state["status"] = "budget_stopped"
        state["updated_at"] = utc_now_iso()
        _write_cost_status(state)
        _write_json_payload(
            os.getenv("ACE_COST_BUDGET_STOP_PATH", "").strip(),
            _budget_stop_payload(
                reason="preflight_budget_exceeded",
                model=model,
                actual_cost_usd=float(state.get("actual_cost_usd", 0.0)),
                next_call_upper_bound_usd=upper_bound_usd,
            ),
        )
        raise RuntimeError(
            f"Budget stop before request: actual_cost_usd={state['actual_cost_usd']:.6f} "
            f"next_call_upper_bound_usd={upper_bound_usd:.6f} budget_limit_usd={state['budget_limit_usd']:.6f}"
        )
    state["status"] = _merge_cost_status(state.get("status"), "running")
    state["updated_at"] = utc_now_iso()
    _write_cost_status(state)


def _record_actual_cost(
    *,
    model: str,
    body: Dict[str, Any],
    request_kind: str,
) -> None:
    if not _cost_env_enabled():
        return
    status_path = os.getenv("ACE_COST_STATUS_PATH", "").strip()
    tracking_mode = _cost_tracking_mode()
    state = _cost_state(model)
    usage = _extract_usage_tokens(body)
    if usage is None:
        if tracking_mode == "enforce":
            state["status"] = "usage_missing_fail_closed"
            state["updated_at"] = utc_now_iso()
            _write_cost_status(state)
            _write_json_payload(
                os.getenv("ACE_COST_BUDGET_STOP_PATH", "").strip(),
                _budget_stop_payload(
                    reason="usage_missing_fail_closed",
                    model=model,
                    actual_cost_usd=float(state.get("actual_cost_usd", 0.0)),
                    next_call_upper_bound_usd=float(state.get("last_call_upper_bound_usd", 0.0)),
                ),
            )
            raise RuntimeError(f"Budget tracking fail-closed: provider usage missing for {request_kind} response.")
        state["status"] = _merge_cost_status(state.get("status"), "usage_missing_partial")
        state["cost_reliability"] = "partial"
        state["updated_at"] = utc_now_iso()
        _write_cost_status(state)
        _append_json_line(
            os.getenv("ACE_COST_EVENTS_PATH", "").strip(),
            {
                "timestamp": utc_now_iso(),
                "request_kind": request_kind,
                "model_id": model,
                "dataset_index_current": state.get("dataset_index_current"),
                "sample_count_completed": state.get("sample_count_completed"),
                "status_path": status_path,
                "status": "usage_missing_partial",
                "reason": "usage_missing",
            },
        )
        return

    price_in = float(state.get("price_input_per_million_usd", 0.0))
    price_out = float(state.get("price_output_per_million_usd", 0.0))
    actual_call_cost = (
        usage["prompt_tokens"] * price_in / 1_000_000.0
        + usage["completion_tokens"] * price_out / 1_000_000.0
    )
    state["prompt_tokens_total"] = int(state.get("prompt_tokens_total", 0)) + usage["prompt_tokens"]
    state["completion_tokens_total"] = int(state.get("completion_tokens_total", 0)) + usage["completion_tokens"]
    state["actual_cost_usd"] = float(state.get("actual_cost_usd", 0.0)) + actual_call_cost
    state["remaining_budget_usd"] = max(0.0, float(state.get("budget_limit_usd", 0.0)) - float(state.get("actual_cost_usd", 0.0)))
    state["last_call_actual_usd"] = actual_call_cost
    state["last_call_prompt_tokens"] = usage["prompt_tokens"]
    state["last_call_completion_tokens"] = usage["completion_tokens"]
    if str(state.get("status") or "").strip() != "usage_missing_partial":
        state["cost_reliability"] = "full"
    state["status"] = _merge_cost_status(state.get("status"), "running")
    state["updated_at"] = utc_now_iso()
    _write_cost_status(state)
    _append_json_line(
        os.getenv("ACE_COST_EVENTS_PATH", "").strip(),
        {
            "timestamp": utc_now_iso(),
            "request_kind": request_kind,
            "model_id": model,
            "dataset_index_current": state.get("dataset_index_current"),
            "sample_count_completed": state.get("sample_count_completed"),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "actual_call_cost_usd": actual_call_cost,
            "actual_cost_usd": state["actual_cost_usd"],
            "remaining_budget_usd": state["remaining_budget_usd"],
            "status_path": status_path,
        },
    )


def build_provider_headers(base_url: str, api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider_kind_from_base_url(base_url) != "openrouter":
        return headers
    optional_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "").strip(),
        "X-OpenRouter-Title": os.getenv("OPENROUTER_TITLE", "").strip(),
        "X-OpenRouter-Categories": os.getenv("OPENROUTER_CATEGORIES", "").strip(),
    }
    legacy_title = os.getenv("OPENROUTER_X_TITLE", "").strip()
    if legacy_title and not optional_headers["X-OpenRouter-Title"]:
        optional_headers["X-OpenRouter-Title"] = legacy_title
    for key, value in optional_headers.items():
        if value:
            headers[key] = value
    return headers


def chat_max_tokens_field(*, model: str, base_url: str) -> str:
    provider = provider_kind_from_base_url(base_url)
    if provider == "openrouter":
        return "max_completion_tokens"
    if provider == "openai" and model.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def responses_function_tools_from_chat_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            converted.append(tool)
            continue
        function = tool.get("function") or {}
        converted.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                "strict": True,
            }
        )
    return converted


def call_openai_compatible_chat(
    *,
    model: str,
    base_url: str,
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    request_timeout: float,
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    headers = build_provider_headers(base_url, api_key)
    provider = provider_kind_from_base_url(base_url)
    is_openai_gpt5_chat = provider == "openai" and model.startswith("gpt-5")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if not is_openai_gpt5_chat:
        payload["temperature"] = temperature
    elif reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        payload[chat_max_tokens_field(model=model, base_url=base_url)] = max_tokens
    if tools is not None:
        payload["tools"] = tools
    _preflight_cost_guard(model=model, payload=payload, max_tokens=max_tokens)
    response = requests.post(
        chat_completions_url(base_url),
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {response.status_code} from {chat_completions_url(base_url)}: {response.text}"
        ) from exc
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"Unexpected API response body: {body!r}")
    _record_actual_cost(model=model, body=body, request_kind="chat_completions")
    return body


def call_openai_responses(
    *,
    model: str,
    base_url: str,
    api_key: str,
    input_items: List[Dict[str, Any]],
    max_tokens: Optional[int],
    request_timeout: float,
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
    previous_response_id: Optional[str] = None,
) -> Dict[str, Any]:
    headers = build_provider_headers(base_url, api_key)
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    if tools is not None:
        payload["tools"] = responses_function_tools_from_chat_tools(tools)
    _preflight_cost_guard(model=model, payload=payload, max_tokens=max_tokens)
    response = requests.post(
        responses_url(base_url),
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {response.status_code} from {responses_url(base_url)}: {response.text}"
        ) from exc
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"Unexpected API response body: {body!r}")
    _record_actual_cost(model=model, body=body, request_kind="responses")
    return body


def extract_first_message(response_body: Dict[str, Any]) -> Dict[str, Any]:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Unexpected API response schema: {response_body}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Unexpected API response schema: {response_body}")
    return message


def _content_tool_fallback_enabled(model: Optional[str]) -> bool:
    return isinstance(model, str) and model in CONTENT_TOOL_FALLBACK_MODELS


def _coerce_message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "\n".join(part for part in text_parts if part)
    return ""


def _extract_synthetic_tool_call_from_content(
    content: str,
    *,
    available_actions: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    text = content.strip()
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    bare_text = text.strip()
    if len(bare_text) >= 2 and bare_text.startswith("`") and bare_text.endswith("`"):
        bare_text = bare_text[1:-1].strip()
    search_match = re.search(r"""(?s)\b(search_tool|search|search_action)\s*\((.*?)\)""", text)
    if search_match:
        inner = search_match.group(2)
        search_args = [
            match.group(2)
            for match in re.finditer(r"""(?s)(['"])(.*?)\1""", inner)
            if isinstance(match.group(2), str) and match.group(2).strip()
        ]
        if search_args:
            return {
                "id": "synthetic_content_parse_0",
                "type": "function",
                "function": {
                    "name": "search_action",
                    "arguments": json.dumps({"keywords": " ".join(search_args)}),
                },
                "_synthetic_from_content": True,
            }

    click_match = re.search(r"""(?s)\b(click|click_action)\s*\(\s*(['"])(.*?)\2\s*\)""", text)
    if click_match:
        return {
            "id": "synthetic_content_parse_0",
            "type": "function",
            "function": {
                "name": "click_action",
                "arguments": json.dumps({"value": click_match.group(3)}),
                },
                "_synthetic_from_content": True,
            }

    if isinstance(available_actions, dict):
        clickables = available_actions.get("clickables")
        if isinstance(clickables, list):
            normalized_bare = bare_text.lower()
            normalized_clickables = {
                clickable.lower(): clickable
                for clickable in clickables
                if isinstance(clickable, str) and clickable.strip()
            }
            search_clickable = normalized_clickables.get("search")
            if (
                normalized_bare in {"search", "search_action", "search_tool"}
                and search_clickable is not None
                and available_actions.get("has_search_bar", False)
            ):
                return {
                    "id": "synthetic_content_parse_0",
                    "type": "function",
                    "function": {
                        "name": "click_action",
                        "arguments": json.dumps({"value": search_clickable}),
                    },
                    "_synthetic_from_content": True,
                }
            if normalized_bare in {"click", "click_action"} and len(normalized_clickables) == 1:
                lone_clickable = next(iter(normalized_clickables.values()))
                return {
                    "id": "synthetic_content_parse_0",
                    "type": "function",
                    "function": {
                        "name": "click_action",
                        "arguments": json.dumps({"value": lone_clickable}),
                    },
                    "_synthetic_from_content": True,
                }
            for clickable in clickables:
                if not isinstance(clickable, str):
                    continue
                if normalized_bare == clickable.lower():
                    return {
                        "id": "synthetic_content_parse_0",
                        "type": "function",
                        "function": {
                            "name": "click_action",
                            "arguments": json.dumps({"value": clickable}),
                        },
                        "_synthetic_from_content": True,
                    }

    if re.search(r"""(?s)\babstain_action\b(?:\s*\(\s*\))?""", text):
        return {
            "id": "synthetic_content_parse_0",
            "type": "function",
            "function": {
                "name": "abstain_action",
                "arguments": "{}",
            },
            "_synthetic_from_content": True,
        }

    return None


def extract_tool_calls_from_message(
    message: Dict[str, Any],
    *,
    model: Optional[str] = None,
    available_actions: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    if not _content_tool_fallback_enabled(model):
        return []
    content = _coerce_message_content_to_text(message.get("content"))
    synthetic_tool_call = _extract_synthetic_tool_call_from_content(
        content,
        available_actions=available_actions,
    )
    if synthetic_tool_call is None:
        return []
    return [synthetic_tool_call]


def extract_tool_calls_from_responses_output(response_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = response_body.get("output") or []
    if not isinstance(output, list):
        return []
    tool_calls: List[Dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                },
            }
        )
    return tool_calls


def assistant_message_from_response(
    message: Dict[str, Any],
    *,
    model: Optional[str] = None,
    available_actions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_calls = extract_tool_calls_from_message(
        message,
        model=model,
        available_actions=available_actions,
    )
    content = message.get("content") or ""
    if tool_calls and any(call.get("_synthetic_from_content") for call in tool_calls):
        content = ""
    assistant_message = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


def stable_shuffle(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def load_bucket_plan(path: Path) -> Dict[str, Any]:
    plan = load_json(path)
    if not isinstance(plan, dict):
        raise ValueError(f"Bucket plan must be a JSON object: {path}")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(f"Bucket plan must contain an assignments list: {path}")
    return plan


def load_rewrite_output_map(path: Path, expected_category: str) -> Dict[int, Dict[str, Any]]:
    records = load_jsonl(path)
    rewrite_map: Dict[int, Dict[str, Any]] = {}
    for record in records:
        dataset_index = int(record["dataset_index"])
        category = record.get("category")
        if category != expected_category:
            raise ValueError(f"Unexpected category {category!r} in {path}; expected {expected_category!r}")
        if dataset_index in rewrite_map:
            raise ValueError(f"Duplicate dataset_index {dataset_index} found in {path}")
        rewrite_map[dataset_index] = record
    return rewrite_map


def retry_call_with_validation(
    *,
    source_record: Dict[str, Any],
    category: str,
    prompt_version: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    sleep_seconds: float,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    messages = build_rewrite_messages(source_record, category, prompt_version=prompt_version)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response_body = call_openai_compatible_chat(
                model=model,
                base_url=base_url,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                reasoning_effort=reasoning_effort,
            )
            message = extract_first_message(response_body)
            raw_text = message.get("content")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("API returned an empty response content.")
            payload = parse_model_json(raw_text)
            validated = validate_rewrite_output(source_record["instruction"], payload)
            return {
                "dataset_index": int(source_record["dataset_index"]),
                "source_id": source_record["source_id"],
                "asin": source_record["asin"],
                "instruction_idx_within_asin": int(source_record["instruction_idx_within_asin"]),
                "source_instruction": source_record["instruction"],
                "rewritten_instruction": validated["rewritten_instruction"],
                "short_reason": validated["short_reason"],
                "category": category,
                "should_abstain": True,
                "source_split": source_record["source_split"],
                "instruction_attributes": source_record["instruction_attributes"],
                "instruction_options": source_record["instruction_options"],
                "model": model,
                "base_url": base_url,
                "prompt_version": prompt_version,
                "created_at": utc_now_iso(),
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed to rewrite dataset_index={source_record['dataset_index']} after {max_retries} attempts: {last_error}"
    )


def get_api_key(api_key_env: str) -> str:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is empty or not set.")
    return api_key


def _validate_search_arguments(arguments: Any, available_actions: Dict[str, Any]) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if set(arguments.keys()) != {"keywords"}:
        return "unexpected_argument_keys"
    keywords = arguments.get("keywords")
    if not available_actions.get("has_search_bar", False):
        return "search_unavailable"
    if not isinstance(keywords, str) or not keywords.strip():
        return "invalid_keywords"
    return None


def _validate_click_arguments(arguments: Any, available_actions: Dict[str, Any]) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if set(arguments.keys()) != {"value"}:
        return "unexpected_argument_keys"
    value = arguments.get("value")
    if not isinstance(value, str):
        return "invalid_value"
    normalized_clickables = {str(item).lower() for item in available_actions.get("clickables", [])}
    if value.lower() not in normalized_clickables:
        return "value_not_available"
    return None


def _validate_abstain_arguments(arguments: Any) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if arguments:
        return "unexpected_argument_keys"
    return None


def classify_first_tool_call(tool_calls: List[Dict[str, Any]], available_actions: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "predicted_class": "invalid",
        "predicted_tool_name": None,
        "predicted_arguments": None,
        "invalid": True,
        "invalid_reason": None,
        "invalid_arguments": False,
        "invalid_arguments_reason": None,
    }

    if not tool_calls:
        result["invalid_reason"] = "no_tool_call"
        return result

    tool_call = tool_calls[0]
    function = tool_call.get("function") or {}
    function_name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    result["predicted_tool_name"] = function_name

    if function_name not in VALID_FIRST_TURN_TOOLS:
        result["invalid_reason"] = "unknown_tool"
        return result

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except Exception:
        result["invalid_reason"] = "arguments_parse_error"
        return result

    result["predicted_class"] = VALID_FIRST_TURN_TOOLS[function_name]
    result["predicted_arguments"] = arguments
    result["invalid"] = False
    result["invalid_reason"] = None

    if function_name == "search_action":
        invalid_argument_reason = _validate_search_arguments(arguments, available_actions)
    elif function_name == "click_action":
        invalid_argument_reason = _validate_click_arguments(arguments, available_actions)
    else:
        invalid_argument_reason = _validate_abstain_arguments(arguments)

    if invalid_argument_reason is not None:
        result["invalid_arguments"] = True
        result["invalid_arguments_reason"] = invalid_argument_reason

    return result


def normalize_action_from_classification(classification: Dict[str, Any]) -> Optional[str]:
    if classification.get("invalid"):
        return None

    predicted_class = classification.get("predicted_class")
    arguments = classification.get("predicted_arguments")
    if predicted_class == "search":
        if isinstance(arguments, dict):
            keywords = arguments.get("keywords")
            if isinstance(keywords, str):
                return f"search[{keywords}]"
        return None
    if predicted_class == "click":
        if isinstance(arguments, dict):
            value = arguments.get("value")
            if isinstance(value, str):
                return f"click[{value}]"
        return None
    if predicted_class == "abstain":
        return "abstain"
    return None


def compute_first_turn_metrics(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    prediction_counts = {
        "search": 0,
        "click": 0,
        "abstain": 0,
        "invalid": 0,
    }
    invalid_samples = []
    invalid_argument_samples = []

    n_positive = 0
    n_negative = 0

    for result in sample_results:
        predicted_class = result.get("predicted_class", "invalid")
        prediction_counts[predicted_class] = prediction_counts.get(predicted_class, 0) + 1

        if result.get("should_abstain"):
            n_positive += 1
        else:
            n_negative += 1

        if result.get("invalid"):
            invalid_samples.append(
                {
                    "index": result.get("index"),
                    "source_id": result.get("source_id"),
                    "reason": result.get("invalid_reason"),
                    "predicted_tool_name": result.get("predicted_tool_name"),
                }
            )

        if result.get("invalid_arguments"):
            invalid_argument_samples.append(
                {
                    "index": result.get("index"),
                    "source_id": result.get("source_id"),
                    "predicted_class": result.get("predicted_class"),
                    "predicted_tool_name": result.get("predicted_tool_name"),
                    "reason": result.get("invalid_arguments_reason"),
                    "predicted_arguments": result.get("predicted_arguments"),
                }
            )

    evaluable_results = [result for result in sample_results if not result.get("invalid")]
    evaluable_positive = [result for result in evaluable_results if result.get("should_abstain")]
    evaluable_negative = [result for result in evaluable_results if not result.get("should_abstain")]

    tp = sum(1 for result in evaluable_positive if result.get("predicted_class") == "abstain")
    fn = sum(1 for result in evaluable_positive if result.get("predicted_class") in {"search", "click"})
    fp = sum(1 for result in evaluable_negative if result.get("predicted_class") == "abstain")
    tn = sum(1 for result in evaluable_negative if result.get("predicted_class") in {"search", "click"})

    abstain_precision = _safe_div(tp, tp + fp)
    abstain_recall = _safe_div(tp, tp + fn)
    abstain_f1 = _safe_div(2 * abstain_precision * abstain_recall, abstain_precision + abstain_recall)
    utr_abs = _safe_div(fn, n_positive)

    return {
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "abstain_f1": abstain_f1,
        "utr_abs": utr_abs,
        "n_total": len(sample_results),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_evaluable": len(evaluable_results),
        "n_positive_evaluable": len(evaluable_positive),
        "n_negative_evaluable": len(evaluable_negative),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prediction_counts": prediction_counts,
        "invalid_rate": _safe_div(len(invalid_samples), len(sample_results)),
        "invalid_samples": invalid_samples,
        "invalid_argument_rate": _safe_div(len(invalid_argument_samples), len(sample_results)),
        "invalid_argument_samples": invalid_argument_samples,
    }


def compute_multiturn_metrics(
    sample_results: List[Dict[str, Any]],
    *,
    gamma: float = MULTITURN_GAMMA,
    penalty_lambda: float = MULTITURN_LAMBDA,
    max_delta: int = DEFAULT_PASS_AT_MAX_DELTA,
) -> Dict[str, Any]:
    first_round_inputs: List[Dict[str, Any]] = []
    s_scores: List[float] = []
    pass_counts = {delta: 0 for delta in range(max_delta + 1)}
    n_positive = 0
    n_negative = 0
    eventual_tp = 0
    eventual_fp = 0
    eventual_fn = 0
    eventual_tn = 0

    for result in sample_results:
        if result.get("should_abstain"):
            n_positive += 1
        else:
            n_negative += 1

        first_round = result.get("first_round") or {}
        first_round_inputs.append(
            {
                "index": result.get("index"),
                "source_id": result.get("source_id"),
                "should_abstain": result.get("should_abstain"),
                "predicted_class": first_round.get("predicted_class", "invalid"),
                "predicted_tool_name": first_round.get("predicted_tool_name"),
                "predicted_arguments": first_round.get("predicted_arguments"),
                "invalid": first_round.get("invalid", True),
                "invalid_reason": first_round.get("invalid_reason"),
                "invalid_arguments": first_round.get("invalid_arguments", False),
                "invalid_arguments_reason": first_round.get("invalid_arguments_reason"),
            }
        )

        eventual_abstain = bool(result.get("eventual_abstain"))
        should_abstain = bool(result.get("should_abstain"))
        if should_abstain and eventual_abstain:
            eventual_tp += 1
        elif should_abstain and not eventual_abstain:
            eventual_fn += 1
        elif not should_abstain and eventual_abstain:
            eventual_fp += 1
        else:
            eventual_tn += 1

        if not should_abstain:
            continue

        if not eventual_abstain:
            s_scores.append(0.0)
            continue

        first_abstain_step = result.get("first_abstain_step")
        if not isinstance(first_abstain_step, int) or first_abstain_step < 1:
            s_scores.append(0.0)
            continue

        k_before_first_abstain = result.get("k_before_first_abstain")
        if not isinstance(k_before_first_abstain, int) or k_before_first_abstain < 0:
            k_before_first_abstain = 0

        delay = first_abstain_step - 1
        s_score = (gamma ** delay) * math.exp(-penalty_lambda * k_before_first_abstain)
        s_scores.append(s_score)
        for delta in range(max_delta + 1):
            if delay <= delta:
                pass_counts[delta] += 1

    first_round_metrics = compute_first_turn_metrics(first_round_inputs)
    first_round_metrics = {
        f"first_round_{key}": value
        for key, value in first_round_metrics.items()
    }

    eventual_precision = _safe_div(eventual_tp, eventual_tp + eventual_fp)
    eventual_recall = _safe_div(eventual_tp, eventual_tp + eventual_fn)
    eventual_f1 = _safe_div(2 * eventual_precision * eventual_recall, eventual_precision + eventual_recall)

    summary: Dict[str, Any] = {
        "n_total": len(sample_results),
        "n_positive": n_positive,
        "n_negative": n_negative,
        **first_round_metrics,
        "eventual_precision": eventual_precision,
        "eventual_recall": eventual_recall,
        "eventual_f1": eventual_f1,
        "eventual_tp": eventual_tp,
        "eventual_fp": eventual_fp,
        "eventual_fn": eventual_fn,
        "eventual_tn": eventual_tn,
        "s_mean_positive": _safe_div(sum(s_scores), len(s_scores)),
    }
    for delta in range(max_delta + 1):
        summary[f"pass_at_{delta}"] = _safe_div(pass_counts[delta], n_positive)
    return summary


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
