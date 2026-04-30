#!/usr/bin/env python3
"""Split a WebShop instruction-set JSONL into round-robin shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem

    records = load_jsonl(input_path)
    for shard in range(args.num_shards):
        shard_records = records[shard:: args.num_shards]
        output_path = output_dir / f"{prefix}.shard{shard:02d}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in shard_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{output_path}\t{len(shard_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
