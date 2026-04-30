#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observe_act_turns import (
    actual_decision_from_result,
    compute_trial_turn_metrics,
    load_result_json,
    summarize_turn_metric_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute observe-act turns from existing Harbor Codex traces "
            "or from the local case-study bundle."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-dir", type=Path, default=None)
    group.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the sidecar JSON. Defaults to <input>/observe_act_turns.json",
    )
    return parser.parse_args()


def discover_job_trials(job_dir: Path) -> list[Path]:
    trials: list[Path] = []
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "result.json").is_file():
            trials.append(child)
    return trials


def discover_bundle_trials(bundle_dir: Path) -> list[Path]:
    return sorted(path.parent for path in bundle_dir.glob("**/result.json"))


def build_row(trial_dir: Path) -> dict[str, object]:
    result = load_result_json(trial_dir)
    turn_metrics = compute_trial_turn_metrics(trial_dir, result=result)
    return {
        "task_name": result.get("task_name"),
        "trial_name": result.get("trial_name"),
        "trial_dir": str(trial_dir.resolve()),
        "agent_name": (result.get("agent_info") or {}).get("name"),
        "actual_decision": actual_decision_from_result(result),
        **turn_metrics,
    }


def main() -> int:
    args = parse_args()
    input_root = (
        args.job_dir.expanduser().resolve()
        if args.job_dir is not None
        else args.bundle_dir.expanduser().resolve()
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else input_root / "observe_act_turns.json"
    )

    if args.job_dir is not None:
        trials = discover_job_trials(input_root)
        input_kind = "job_dir"
    else:
        trials = discover_bundle_trials(input_root)
        input_kind = "bundle_dir"

    if not trials:
        raise RuntimeError(f"No trial result.json files found under {input_root}")

    rows = [build_row(trial_dir) for trial_dir in trials]
    payload = {
        "input_kind": input_kind,
        "input_root": str(input_root),
        **summarize_turn_metric_rows(rows),
        "per_trial": rows,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Observe-act turn summary written to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
