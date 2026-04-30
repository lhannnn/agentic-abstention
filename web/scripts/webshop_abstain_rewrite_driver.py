#!/usr/bin/env python3
"""Rewrite WebShop instructions for one abstain category using an OpenAI-compatible API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from webshop_abstain_common import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BUCKET_PLAN_PATH,
    DEFAULT_MODEL,
    DEFAULT_REWRITE_OUTPUTS,
    PROMPT_VERSION_V1,
    POSITIVE_CATEGORIES,
    SUBJECTIVE,
    UNDERSPECIFIED_INTENT,
    FALSE_PREMISES,
    EXTERNAL_INFORMATION_REQUIRED,
    append_jsonl,
    get_api_key,
    load_bucket_plan,
    load_jsonl,
    retry_call_with_validation,
)


def parse_args(
    default_category: str | None = None,
    *,
    default_bucket_plan: Path | str = DEFAULT_BUCKET_PLAN_PATH,
    default_outputs: Dict[str, Path] = DEFAULT_REWRITE_OUTPUTS,
    allowed_categories: List[str] = POSITIVE_CATEGORIES,
    default_prompt_version: str = PROMPT_VERSION_V1,
    default_model: str = DEFAULT_MODEL,
    default_base_url: str = DEFAULT_BASE_URL,
    default_api_key_env: str = DEFAULT_API_KEY_ENV,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if default_category is None:
        parser.add_argument("--category", required=True, choices=allowed_categories)
    else:
        parser.add_argument("--category", default=default_category, choices=allowed_categories)
    parser.add_argument("--bucket-plan", default=str(default_bucket_plan))
    parser.add_argument("--output", help="Output JSONL path. Defaults to the category-specific file.")
    parser.add_argument("--prompt-version", default=default_prompt_version)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--api-key-env", default=default_api_key_env)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Optional OpenAI GPT-5 reasoning effort override.",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--count", type=int, help="Optional cap on how many records to rewrite.")
    parser.add_argument("--resume", action="store_true", help="Append and skip already-finished dataset_index values.")
    args = parser.parse_args()
    args.default_outputs = default_outputs
    return args


def completed_dataset_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(record["dataset_index"]) for record in load_jsonl(path)}


def rewrite_assignment(
    assignment: Dict[str, object],
    *,
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
    reasoning_effort: str | None,
) -> Dict[str, object]:
    return retry_call_with_validation(
        source_record=assignment,
        category=category,
        prompt_version=prompt_version,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
        reasoning_effort=reasoning_effort,
    )


def main(
    default_category: str | None = None,
    *,
    default_bucket_plan: Path | str = DEFAULT_BUCKET_PLAN_PATH,
    default_outputs: Dict[str, Path] = DEFAULT_REWRITE_OUTPUTS,
    allowed_categories: List[str] = POSITIVE_CATEGORIES,
    default_prompt_version: str = PROMPT_VERSION_V1,
    default_model: str = DEFAULT_MODEL,
    default_base_url: str = DEFAULT_BASE_URL,
    default_api_key_env: str = DEFAULT_API_KEY_ENV,
) -> int:
    args = parse_args(
        default_category=default_category,
        default_bucket_plan=default_bucket_plan,
        default_outputs=default_outputs,
        allowed_categories=allowed_categories,
        default_prompt_version=default_prompt_version,
        default_model=default_model,
        default_base_url=default_base_url,
        default_api_key_env=default_api_key_env,
    )
    category = args.category
    plan = load_bucket_plan(Path(args.bucket_plan).expanduser().resolve())

    assignments = [row for row in plan["assignments"] if row["category"] == category]
    if args.count is not None:
        assignments = assignments[: args.count]

    output_path = Path(args.output).expanduser().resolve() if args.output else args.default_outputs[category]
    completed = completed_dataset_indices(output_path) if args.resume else set()

    if output_path.exists() and not args.resume:
        raise RuntimeError(
            f"Output file already exists: {output_path}. Re-run with --resume or choose a different --output path."
        )

    pending = [row for row in assignments if int(row["dataset_index"]) not in completed]
    if not pending:
        print(f"No pending records for category={category}. Output is up to date: {output_path}")
        return 0

    api_key = get_api_key(args.api_key_env)

    successes: List[Dict[str, object]] = []
    failures: List[str] = []
    max_workers = max(1, args.concurrency)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset_index = {
            executor.submit(
                rewrite_assignment,
                assignment,
                category=category,
                prompt_version=args.prompt_version,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
                reasoning_effort=args.reasoning_effort,
            ): int(assignment["dataset_index"])
            for assignment in pending
        }
        for future in as_completed(future_to_dataset_index):
            dataset_index = future_to_dataset_index[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                failures.append(f"dataset_index={dataset_index}: {exc}")

    successes.sort(key=lambda record: int(record["dataset_index"]))
    append_jsonl(output_path, successes)
    print(f"Wrote {len(successes)} rewritten records for category={category} to {output_path}")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
