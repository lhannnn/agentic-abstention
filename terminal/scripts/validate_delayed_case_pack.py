#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "data" / "delayed"
REQUIRED_FIELDS = {
    "task_name",
    "source_task_name",
    "variant",
    "abstention_timing",
    "trigger_style",
    "trigger_evidence",
    "expected_abstain",
    "expected_decision",
    "earliest_abstain_turn",
    "timely_grace_turns",
    "task_dir",
}
SPEC_REQUIRED_FIELDS = {
    "case_id",
    "source_task_name",
    "trigger_style",
    "design_goal",
    "original_instruction_summary",
    "rewrite_intent",
    "instruction_edits",
    "environment_edits",
    "trigger_evidence",
    "earliest_abstain_turn",
    "timely_grace_window",
    "why_original_remains_solvable",
    "why_delayed_rewrite_is_initially_plausible",
    "why_task_becomes_objectively_unresolvable_after_trigger",
    "failure_modes_to_avoid",
    "final_reviewer_verdicts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the delayed-abstention case pack metadata.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--require-task-dirs",
        action="store_true",
        help="Also require materialized task directories. Off by default because release data intentionally omits them.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"Blank line in manifest at line {line_no}")
        row = json.loads(line)
        if set(row) != REQUIRED_FIELDS:
            raise RuntimeError(f"Unexpected manifest schema at line {line_no}: {sorted(row)}")
        if row["task_name"] in seen:
            raise RuntimeError(f"Duplicate task_name in manifest: {row['task_name']}")
        seen.add(row["task_name"])
        rows.append(row)
    return rows


def validate_specs(dataset_dir: Path, delayed_rows: list[dict[str, Any]]) -> None:
    specs_dir = dataset_dir / "specs"
    if not specs_dir.is_dir():
        raise RuntimeError(f"Specs directory not found: {specs_dir}")
    for row in delayed_rows:
        spec_path = specs_dir / f"{row['task_name']}.json"
        if not spec_path.is_file():
            raise RuntimeError(f"Missing spec file for delayed case: {spec_path}")
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        if not SPEC_REQUIRED_FIELDS.issubset(payload):
            raise RuntimeError(f"Spec missing required fields for {spec_path.name}: {sorted(payload)}")
        if payload["case_id"] != row["task_name"]:
            raise RuntimeError(f"case_id mismatch in {spec_path}")
        if payload["source_task_name"] != row["source_task_name"]:
            raise RuntimeError(f"source_task_name mismatch in {spec_path}")
        if payload["trigger_style"] != row["trigger_style"]:
            raise RuntimeError(f"trigger_style mismatch in {spec_path}")
        final_verdicts = payload["final_reviewer_verdicts"]
        if not isinstance(final_verdicts, list) or len(final_verdicts) != 2:
            raise RuntimeError(f"Expected exactly 2 reviewer verdicts in {spec_path}")


def validate_rewrite_only_reviews(dataset_dir: Path, delayed_rows: list[dict[str, Any]]) -> tuple[int, int]:
    reviews_dir = dataset_dir / "reviews"
    policy_path = reviews_dir / "rewrite_only_policy.json"
    consensus_path = reviews_dir / "rewrite_only_consensus.json"
    if not policy_path.is_file():
        raise RuntimeError(f"Missing rewrite-only policy file: {policy_path}")
    if not consensus_path.is_file():
        raise RuntimeError(f"Missing rewrite-only consensus file: {consensus_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("acceptance_focus") != "rewrite_correctness_only":
        raise RuntimeError("rewrite_only_policy.json must use acceptance_focus=rewrite_correctness_only")
    if policy.get("ignore_visible_verifier_permissiveness") is not True:
        raise RuntimeError("rewrite_only_policy.json must ignore visible-verifier permissiveness")
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    accepted = consensus.get("accepted")
    backlog = consensus.get("backlog")
    if not isinstance(accepted, list) or not isinstance(backlog, list):
        raise RuntimeError("rewrite_only_consensus.json accepted/backlog must be lists")
    delayed_names = {row["task_name"] for row in delayed_rows}
    unknown = (set(accepted) | set(backlog)) - delayed_names
    if unknown:
        raise RuntimeError(f"rewrite_only_consensus.json references unknown cases: {sorted(unknown)}")
    if set(accepted) & set(backlog):
        raise RuntimeError("rewrite_only_consensus.json has overlap between accepted and backlog")
    return len(accepted), len(backlog)


def validate_task_dirs(dataset_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        task_dir = dataset_dir.parent.parent / row["task_dir"]
        if not task_dir.is_dir():
            raise RuntimeError(f"Missing task_dir for manifest row: {task_dir}")
        if not (task_dir / "instruction.md").is_file():
            raise RuntimeError(f"Missing instruction.md in {task_dir}")
        if not (task_dir / "task.toml").is_file():
            raise RuntimeError(f"Missing task.toml in {task_dir}")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest_rows = load_manifest(dataset_dir / "manifest.jsonl")
    delayed_rows = [row for row in manifest_rows if row["abstention_timing"] == "delayed"]
    control_rows = [row for row in manifest_rows if row["abstention_timing"] == "control"]
    for row in delayed_rows:
        if row["trigger_style"] == "none":
            raise RuntimeError(f"Delayed row has trigger_style=none: {row['task_name']}")
        if row["trigger_evidence"] in (None, ""):
            raise RuntimeError(f"Delayed row missing trigger_evidence: {row['task_name']}")
        if not isinstance(row["earliest_abstain_turn"], int):
            raise RuntimeError(f"Delayed row missing earliest_abstain_turn: {row['task_name']}")
    for row in control_rows:
        if row["trigger_style"] != "none":
            raise RuntimeError(f"Control row has non-none trigger_style: {row['task_name']}")
    validate_specs(dataset_dir, delayed_rows)
    accepted_count, backlog_count = validate_rewrite_only_reviews(dataset_dir, delayed_rows)
    if args.require_task_dirs:
        validate_task_dirs(dataset_dir, manifest_rows)
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "row_count": len(manifest_rows),
                "control_count": len(control_rows),
                "delayed_count": len(delayed_rows),
                "rewrite_only_accepted_count": accepted_count,
                "rewrite_only_backlog_count": backlog_count,
                "task_dirs_checked": bool(args.require_task_dirs),
                "status": "ok",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
