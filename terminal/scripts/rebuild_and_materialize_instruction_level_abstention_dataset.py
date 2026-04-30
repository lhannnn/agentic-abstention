#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILDER_SCRIPT = ROOT / "scripts" / "build_instruction_level_abstention_dataset.py"
DEFAULT_MATERIALIZE_SCRIPT = (
    ROOT / "scripts" / "materialize_harbor_instruction_level_abstention_dataset.py"
)
DEFAULT_INPUT = ROOT / "terminalbench_instructions.jsonl"
DEFAULT_PROMPTS = ROOT / "prompts" / "instruction_level_abstention_prompts.md"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "instruction_level_abstention_generation"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "harbor" / "tasks"
DEFAULT_OUTPUT = ROOT / "terminalbench_instruction_level_abstention_267.jsonl"
DEFAULT_MATERIALIZED_OUTPUT = (
    ROOT / "datasets" / "terminalbench_instruction_level_abstention_267"
)
DEFAULT_PREFIX = ROOT / "terminalbench_instruction_pipeline"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 8000
DEFAULT_AUDIT_MAX_OUTPUT_TOKENS = 2000
DEFAULT_EXPECTED_BASE_TASKS = 89


def derive_sidecar_path(output_path: Path, suffix: str) -> Path:
    if output_path.name.endswith(".jsonl"):
        base_name = output_path.name[: -len(".jsonl")]
    else:
        base_name = output_path.name
    return output_path.with_name(base_name + suffix)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def count_jsonl_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def absolutize_preserving_symlink(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate GPT rewrite generation and Harbor dataset materialization "
            "without relying on shell chaining."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--builder-only",
        action="store_true",
        help="Run only the builder step.",
    )
    mode_group.add_argument(
        "--materialize-only",
        action="store_true",
        help="Run only the materialize step using existing builder outputs.",
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing successful builder outputs if possible.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help=f"Python executable to use for subprocesses. Default: {sys.executable}",
    )
    parser.add_argument(
        "--builder-script",
        type=Path,
        default=DEFAULT_BUILDER_SCRIPT,
        help=f"Builder script path. Default: {DEFAULT_BUILDER_SCRIPT}",
    )
    parser.add_argument(
        "--materialize-script",
        type=Path,
        default=DEFAULT_MATERIALIZE_SCRIPT,
        help=f"Materialize script path. Default: {DEFAULT_MATERIALIZE_SCRIPT}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input base-instruction JSONL. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help=f"Prompt template file. Default: {DEFAULT_PROMPTS}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Builder cache directory. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Harbor task cache root for materialize. Default: {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Canonical dataset JSONL. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--skipped-output",
        type=Path,
        default=None,
        help="Skipped rewrite JSONL. Default: <output>.skipped.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Builder summary JSON. Default: <output>.summary.json",
    )
    parser.add_argument(
        "--materialized-output",
        type=Path,
        default=DEFAULT_MATERIALIZED_OUTPUT,
        help=f"Materialized Harbor dataset root. Default: {DEFAULT_MATERIALIZED_OUTPUT}",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help=f"Prefix for pipeline status/log files. Default: {DEFAULT_PREFIX}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Builder model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=f"Builder reasoning effort. Default: {DEFAULT_REASONING_EFFORT}",
    )
    parser.add_argument(
        "--expected-base-tasks",
        type=int,
        default=DEFAULT_EXPECTED_BASE_TASKS,
        help=f"Expected base task count. Default: {DEFAULT_EXPECTED_BASE_TASKS}",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max rewrite attempts per task/category. Default: {DEFAULT_MAX_ATTEMPTS}",
    )
    parser.add_argument(
        "--generation-max-output-tokens",
        type=int,
        default=DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
        help=(
            "Builder generation max output tokens. Default: "
            f"{DEFAULT_GENERATION_MAX_OUTPUT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--audit-max-output-tokens",
        type=int,
        default=DEFAULT_AUDIT_MAX_OUTPUT_TOKENS,
        help=(
            "Builder audit max output tokens. Default: "
            f"{DEFAULT_AUDIT_MAX_OUTPUT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N base tasks.",
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
        help="Force builder to ignore successful cache entries.",
    )
    return parser.parse_args()


def status_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        prefix.with_suffix(".status.json"),
        prefix.with_suffix(".builder.log"),
        prefix.with_suffix(".materialize.log"),
    )


def ensure_script_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")


def validate_builder_outputs(
    *,
    output_path: Path,
    skipped_output_path: Path,
    summary_output_path: Path,
) -> dict[str, Any]:
    for path, label in (
        (output_path, "dataset output"),
        (skipped_output_path, "skipped output"),
        (summary_output_path, "summary output"),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing {label}: {path}")

    summary = json.loads(summary_output_path.read_text(encoding="utf-8"))
    required_fields = {
        "selected_base_tasks",
        "original_count",
        "successful_rewrite_count",
        "skipped_rewrite_count",
        "dataset_row_count",
    }
    missing = sorted(required_fields - set(summary))
    if missing:
        raise RuntimeError(
            f"Builder summary missing required fields {missing}: {summary_output_path}"
        )

    dataset_rows = count_jsonl_lines(output_path)
    skipped_rows = count_jsonl_lines(skipped_output_path)
    if dataset_rows != int(summary["dataset_row_count"]):
        raise RuntimeError(
            f"Dataset row count mismatch: {dataset_rows} != {summary['dataset_row_count']}"
        )
    if skipped_rows != int(summary["skipped_rewrite_count"]):
        raise RuntimeError(
            "Skipped rewrite count mismatch: "
            f"{skipped_rows} != {summary['skipped_rewrite_count']}"
        )

    selected_base_tasks = int(summary["selected_base_tasks"])
    expected_total = selected_base_tasks * 3
    if dataset_rows + skipped_rows != expected_total:
        raise RuntimeError(
            f"Builder outputs do not sum to expected total {expected_total}: "
            f"{dataset_rows} + {skipped_rows}"
        )
    if int(summary["original_count"]) != selected_base_tasks:
        raise RuntimeError(
            f"Original count mismatch: {summary['original_count']} != {selected_base_tasks}"
        )
    if int(summary["original_count"]) + int(summary["successful_rewrite_count"]) != dataset_rows:
        raise RuntimeError(
            "Builder summary original+successful does not match dataset rows: "
            f"{summary['original_count']} + {summary['successful_rewrite_count']} != {dataset_rows}"
        )

    return summary


def validate_materialize_output(materialized_output: Path) -> Path:
    if not materialized_output.is_dir():
        raise RuntimeError(f"Materialized output directory not found: {materialized_output}")
    manifest_path = materialized_output / "manifest.jsonl"
    if not manifest_path.is_file():
        raise RuntimeError(f"Materialize manifest not found: {manifest_path}")

    manifest_rows = count_jsonl_lines(manifest_path)
    task_dirs = sum(1 for path in materialized_output.iterdir() if path.is_dir())
    if manifest_rows != task_dirs:
        raise RuntimeError(
            f"Manifest/task count mismatch: {manifest_rows} manifest rows != {task_dirs} task directories"
        )
    return manifest_path


def run_stage(
    *,
    command: list[str],
    log_path: Path,
    cwd: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("$ " + shlex.join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def build_builder_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(args.builder_script),
        "--input",
        str(args.input),
        "--prompts",
        str(args.prompts),
        "--cache-dir",
        str(args.cache_dir),
        "--output",
        str(args.output),
        "--skipped-output",
        str(args.skipped_output),
        "--summary-output",
        str(args.summary_output),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--expected-base-tasks",
        str(args.expected_base_tasks),
        "--max-attempts",
        str(args.max_attempts),
        "--generation-max-output-tokens",
        str(args.generation_max_output_tokens),
        "--audit-max-output-tokens",
        str(args.audit_max_output_tokens),
        "--skip-failed-after-max-attempts",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.task_name:
        for task_name in args.task_name:
            command.extend(["--task-name", task_name])
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_materialize_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.materialize_script),
        "--input",
        str(args.output),
        "--skipped-input",
        str(args.skipped_output),
        "--cache-root",
        str(args.cache_root),
        "--output",
        str(args.materialized_output),
        "--allow-partial",
        "--force",
    ]


def load_status(status_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))


def builder_completed(status: dict[str, Any], args: argparse.Namespace) -> bool:
    if status.get("builder_exit_code") != 0:
        return False
    try:
        validate_builder_outputs(
            output_path=args.output,
            skipped_output_path=args.skipped_output,
            summary_output_path=args.summary_output,
        )
    except Exception:
        return False
    return True


def materialize_completed(status: dict[str, Any], args: argparse.Namespace) -> bool:
    if status.get("materialize_exit_code") != 0:
        return False
    try:
        validate_materialize_output(args.materialized_output)
    except Exception:
        return False
    return True


def main() -> int:
    args = parse_args()
    args.python = absolutize_preserving_symlink(args.python)
    args.builder_script = args.builder_script.expanduser().resolve()
    args.materialize_script = args.materialize_script.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.prompts = args.prompts.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_root = args.cache_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.skipped_output = (
        args.skipped_output.expanduser().resolve()
        if args.skipped_output is not None
        else derive_sidecar_path(args.output, ".skipped.jsonl")
    )
    args.summary_output = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else derive_sidecar_path(args.output, ".summary.json")
    )
    args.materialized_output = args.materialized_output.expanduser().resolve()
    args.prefix = args.prefix.expanduser().resolve()

    ensure_script_exists(args.builder_script, "Builder script")
    ensure_script_exists(args.materialize_script, "Materialize script")

    status_path, builder_log_path, materialize_log_path = status_paths(args.prefix)
    status: dict[str, Any] = {
        "pipeline_started_at": iso_now(),
        "mode": (
            "builder_only"
            if args.builder_only
            else "materialize_only"
            if args.materialize_only
            else "resume"
            if args.resume
            else "full"
        ),
        "builder_script": str(args.builder_script),
        "materialize_script": str(args.materialize_script),
        "output": str(args.output),
        "skipped_output": str(args.skipped_output),
        "summary_output": str(args.summary_output),
        "materialized_output": str(args.materialized_output),
        "builder_log": str(builder_log_path),
        "materialize_log": str(materialize_log_path),
        "final_status": "starting",
    }

    previous_status = load_status(status_path)
    if args.resume and previous_status:
        status.update(previous_status)
        status["mode"] = "resume"
        status["pipeline_resumed_at"] = iso_now()

    write_json(status_path, status)

    try:
        if args.materialize_only:
            validate_builder_outputs(
                output_path=args.output,
                skipped_output_path=args.skipped_output,
                summary_output_path=args.summary_output,
            )
        elif args.resume:
            if materialize_completed(status, args):
                status["final_status"] = "success"
                status["pipeline_finished_at"] = iso_now()
                write_json(status_path, status)
                print(f"Pipeline already complete. Status: {status_path}")
                return 0
            if not builder_completed(status, args):
                status["builder_command"] = build_builder_command(args)
                status["builder_started_at"] = iso_now()
                status["final_status"] = "builder_running"
                write_json(status_path, status)
                builder_exit_code = run_stage(
                    command=status["builder_command"],
                    log_path=builder_log_path,
                    cwd=ROOT,
                )
                status["builder_finished_at"] = iso_now()
                status["builder_exit_code"] = builder_exit_code
                if builder_exit_code != 0:
                    status["final_status"] = "builder_failed"
                    status["pipeline_finished_at"] = iso_now()
                    write_json(status_path, status)
                    print(f"Builder failed. See {builder_log_path}", file=sys.stderr)
                    return builder_exit_code
            validate_builder_outputs(
                output_path=args.output,
                skipped_output_path=args.skipped_output,
                summary_output_path=args.summary_output,
            )
        else:
            status["builder_command"] = build_builder_command(args)
            status["builder_started_at"] = iso_now()
            status["final_status"] = "builder_running"
            write_json(status_path, status)
            builder_exit_code = run_stage(
                command=status["builder_command"],
                log_path=builder_log_path,
                cwd=ROOT,
            )
            status["builder_finished_at"] = iso_now()
            status["builder_exit_code"] = builder_exit_code
            if builder_exit_code != 0:
                status["final_status"] = "builder_failed"
                status["pipeline_finished_at"] = iso_now()
                write_json(status_path, status)
                print(f"Builder failed. See {builder_log_path}", file=sys.stderr)
                return builder_exit_code
            validate_builder_outputs(
                output_path=args.output,
                skipped_output_path=args.skipped_output,
                summary_output_path=args.summary_output,
            )
            if args.builder_only:
                status["final_status"] = "builder_only_complete"
                status["pipeline_finished_at"] = iso_now()
                write_json(status_path, status)
                print(f"Builder completed. Status: {status_path}")
                return 0

        status["materialize_command"] = build_materialize_command(args)
        status["materialize_started_at"] = iso_now()
        status["final_status"] = "materialize_running"
        write_json(status_path, status)
        materialize_exit_code = run_stage(
            command=status["materialize_command"],
            log_path=materialize_log_path,
            cwd=ROOT,
        )
        status["materialize_finished_at"] = iso_now()
        status["materialize_exit_code"] = materialize_exit_code
        if materialize_exit_code != 0:
            status["final_status"] = "materialize_failed"
            status["pipeline_finished_at"] = iso_now()
            write_json(status_path, status)
            print(
                f"Materialize failed. See {materialize_log_path}",
                file=sys.stderr,
            )
            return materialize_exit_code

        manifest_path = validate_materialize_output(args.materialized_output)
        status["manifest_path"] = str(manifest_path)
        status["final_status"] = "success"
        status["pipeline_finished_at"] = iso_now()
        write_json(status_path, status)
    except Exception as exc:  # noqa: BLE001
        status["final_status"] = "pipeline_error"
        status["pipeline_finished_at"] = iso_now()
        status["error"] = str(exc)
        write_json(status_path, status)
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Pipeline completed successfully. Status: {status_path}")
    print(f"Builder log: {builder_log_path}")
    print(f"Materialize log: {materialize_log_path}")
    print(f"Materialized dataset: {args.materialized_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
