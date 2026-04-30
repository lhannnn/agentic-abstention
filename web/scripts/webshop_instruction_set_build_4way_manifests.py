#!/usr/bin/env python3
"""Build homogeneous 4-way shard manifests for the WebShop 1000-task instruction set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "webshop"
DEFAULT_INPUT = DATA_DIR / "webshop_instruction_set_1000_gpt54_high.jsonl"
DEFAULT_OUTPUT_DIR = DATA_DIR / "manifests"


SHARD_SPECS = [
    {
        "name": "shard00",
        "filename": "webshop_instruction_set_1000_gpt54_high__shard00_0_249.jsonl",
        "start": 0,
        "end": 249,
        "variant_type": "original",
        "env_variant": "full",
    },
    {
        "name": "shard01",
        "filename": "webshop_instruction_set_1000_gpt54_high__shard01_250_499.jsonl",
        "start": 250,
        "end": 499,
        "variant_type": "original",
        "env_variant": "full",
    },
    {
        "name": "shard02",
        "filename": "webshop_instruction_set_1000_gpt54_high__shard02_500_748.jsonl",
        "start": 500,
        "end": 748,
        "variant_type": "rewrite",
        "env_variant": "full",
    },
    {
        "name": "shard03",
        "filename": "webshop_instruction_set_1000_gpt54_high__shard03_749_999.jsonl",
        "start": 749,
        "end": 999,
        "variant_type": "missing_target",
        "env_variant": "pruned_missing_target_251",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 rows in {input_path}, found {len(rows)}")

    seen_indices: set[int] = set()
    for spec in SHARD_SPECS:
        subset = [
            row
            for row in rows
            if spec["start"] <= int(row["dataset_index"]) <= spec["end"]
        ]
        if len(subset) != (spec["end"] - spec["start"] + 1):
            raise RuntimeError(
                f"{spec['name']} expected {spec['end'] - spec['start'] + 1} rows, found {len(subset)}"
            )
        variant_counts = Counter(row["variant_type"] for row in subset)
        env_counts = Counter(row["env_variant"] for row in subset)
        if set(variant_counts) != {spec["variant_type"]}:
            raise RuntimeError(f"{spec['name']} variant mismatch: {variant_counts}")
        if set(env_counts) != {spec["env_variant"]}:
            raise RuntimeError(f"{spec['name']} env mismatch: {env_counts}")
        shard_indices = {int(row["dataset_index"]) for row in subset}
        if shard_indices & seen_indices:
            raise RuntimeError(f"{spec['name']} overlaps with previous shards: {sorted(shard_indices & seen_indices)}")
        seen_indices |= shard_indices

        output_path = output_dir / spec["filename"]
        with output_path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{spec['name']} rows={len(subset)} variants={dict(variant_counts)} envs={dict(env_counts)} output={output_path}"
        )

    expected = set(range(1000))
    if seen_indices != expected:
        raise RuntimeError(f"Shard coverage mismatch: missing={sorted(expected - seen_indices)} extra={sorted(seen_indices - expected)}")
    print(f"wrote_manifests={len(SHARD_SPECS)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
