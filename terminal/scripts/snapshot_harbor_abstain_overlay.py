#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_SITE_PACKAGES = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
DEFAULT_OUTPUT_ROOT = ROOT / "harbor_overlay" / "site-packages"
OVERLAY_FILES = [
    "harbor/models/trial/config.py",
    "harbor/agents/factory.py",
    "harbor/agents/terminus_2/templates/terminus-json-plain.txt",
    "harbor/agents/terminus_2/terminus_json_plain_parser.py",
    "harbor/agents/terminus_2/terminus_2.py",
    "harbor/agents/installed/hermes.py",
    "harbor/job.py",
    "harbor/trial/trial.py",
    "harbor/models/job/result.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot the locally patched Harbor abstain files into harbor_overlay."
    )
    parser.add_argument(
        "--source-site-packages",
        type=Path,
        default=DEFAULT_SOURCE_SITE_PACKAGES,
        help=f"Source site-packages root. Default: {DEFAULT_SOURCE_SITE_PACKAGES}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Overlay output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_site_packages.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.is_dir():
        print(f"Source site-packages not found: {source_root}", file=sys.stderr)
        return 1

    copied = 0
    for relative_path in OVERLAY_FILES:
        source_path = source_root / relative_path
        if not source_path.is_file():
            print(f"Source overlay file not found: {source_path}", file=sys.stderr)
            return 1

        destination_path = output_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied += 1

    print(f"Snapshot complete: copied {copied} files into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
