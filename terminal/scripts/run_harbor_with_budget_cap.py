#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CostSnapshot:
    n_trials_with_result: int
    n_input_tokens: int
    n_cache_tokens: int
    n_output_tokens: int
    cost_usd_lower_bound: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Harbor job and stop it if the estimated lower-bound model cost exceeds a budget."
        )
    )
    parser.add_argument("--harbor-bin", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--analysis-script", type=Path, default=None)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--input-price-per-mtok", type=float, default=0.75)
    parser.add_argument("--cached-input-price-per-mtok", type=float, default=0.075)
    parser.add_argument("--output-price-per-mtok", type=float, default=4.5)
    parser.add_argument("--graceful-stop-sec", type=float, default=120.0)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_cost_snapshot(
    job_dir: Path,
    *,
    input_price_per_mtok: float,
    cached_input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> CostSnapshot:
    n_trials_with_result = 0
    n_input_tokens = 0
    n_cache_tokens = 0
    n_output_tokens = 0

    if job_dir.is_dir():
        for child in sorted(job_dir.iterdir()):
            if not child.is_dir():
                continue
            result = load_result(child / "result.json")
            if result is None:
                continue
            n_trials_with_result += 1
            agent_result = result.get("agent_result") or {}
            if not isinstance(agent_result, dict):
                continue
            n_input_tokens += int(agent_result.get("n_input_tokens") or 0)
            n_cache_tokens += int(agent_result.get("n_cache_tokens") or 0)
            n_output_tokens += int(agent_result.get("n_output_tokens") or 0)

    uncached_input_tokens = max(n_input_tokens - n_cache_tokens, 0)
    cost_usd_lower_bound = (
        uncached_input_tokens * input_price_per_mtok
        + n_cache_tokens * cached_input_price_per_mtok
        + n_output_tokens * output_price_per_mtok
    ) / 1_000_000.0

    return CostSnapshot(
        n_trials_with_result=n_trials_with_result,
        n_input_tokens=n_input_tokens,
        n_cache_tokens=n_cache_tokens,
        n_output_tokens=n_output_tokens,
        cost_usd_lower_bound=cost_usd_lower_bound,
    )


def job_trial_dir_names(job_dir: Path) -> set[str]:
    if not job_dir.is_dir():
        return set()
    return {child.name for child in job_dir.iterdir() if child.is_dir()}


def stop_matching_docker_containers(job_dir: Path) -> None:
    trial_names = job_trial_dir_names(job_dir)
    if not trial_names:
        return

    ps = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if ps.returncode != 0:
        return

    running = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    to_stop = [
        name for name in running if any(trial_name in name for trial_name in trial_names)
    ]
    for name in to_stop:
        subprocess.run(["docker", "stop", name], check=False, capture_output=True, text=True)


def terminate_process_tree(proc: subprocess.Popen[bytes], graceful_stop_sec: float) -> None:
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
    except ProcessLookupError:
        return

    deadline = time.time() + graceful_stop_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1.0)

    try:
        proc.kill()
    except ProcessLookupError:
        return


def maybe_run_analysis(args: argparse.Namespace) -> int | None:
    if args.analysis_script is None or args.manifest is None:
        return None
    cmd = [
        str(args.python_bin),
        str(args.analysis_script),
        "--manifest",
        str(args.manifest),
        "--job-dir",
        str(args.job_dir),
    ]
    print(f"[budget-runner] running analysis: {shlex.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    args.job_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.harbor_bin),
        "jobs",
        "start",
        "-c",
        str(args.config),
        "--env-file",
        str(args.env_file),
    ]
    print(f"[budget-runner] starting: {shlex.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
    )

    stopping = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        print(f"[budget-runner] received signal {signum}, stopping child", flush=True)
        terminate_process_tree(proc, args.graceful_stop_sec)
        stop_matching_docker_containers(args.job_dir)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    budget_triggered = False

    try:
        while proc.poll() is None and not stopping:
            snapshot = compute_cost_snapshot(
                args.job_dir,
                input_price_per_mtok=args.input_price_per_mtok,
                cached_input_price_per_mtok=args.cached_input_price_per_mtok,
                output_price_per_mtok=args.output_price_per_mtok,
            )
            print(
                "[budget-runner] "
                f"results={snapshot.n_trials_with_result} "
                f"input={snapshot.n_input_tokens} "
                f"cache={snapshot.n_cache_tokens} "
                f"output={snapshot.n_output_tokens} "
                f"cost_lower_bound=${snapshot.cost_usd_lower_bound:.4f}",
                flush=True,
            )
            if snapshot.cost_usd_lower_bound > args.budget_usd:
                budget_triggered = True
                print(
                    "[budget-runner] budget exceeded; "
                    f"lower_bound=${snapshot.cost_usd_lower_bound:.4f} > ${args.budget_usd:.2f}",
                    flush=True,
                )
                terminate_process_tree(proc, args.graceful_stop_sec)
                stop_matching_docker_containers(args.job_dir)
                break
            time.sleep(args.poll_sec)
    finally:
        if proc.poll() is None:
            terminate_process_tree(proc, args.graceful_stop_sec)

    exit_code = proc.wait()
    analysis_exit_code = maybe_run_analysis(args)

    if budget_triggered:
        return 0
    if exit_code != 0:
        return exit_code
    if analysis_exit_code not in (None, 0):
        return int(analysis_exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
