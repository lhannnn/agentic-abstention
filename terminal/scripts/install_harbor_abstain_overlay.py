#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OVERLAY_ROOT = ROOT / "harbor_overlay" / "site-packages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the Harbor abstain overlay into a target Python environment."
    )
    parser.add_argument(
        "--overlay-root",
        type=Path,
        default=DEFAULT_OVERLAY_ROOT,
        help=f"Overlay root that mirrors site-packages. Default: {DEFAULT_OVERLAY_ROOT}",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=None,
        help="Virtualenv root whose lib/python*/site-packages directory should be patched.",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=None,
        help="Patch this site-packages directory directly instead of resolving from --venv.",
    )
    return parser.parse_args()


def resolve_site_packages(args: argparse.Namespace) -> Path:
    if bool(args.venv) == bool(args.site_packages):
        raise RuntimeError("Pass exactly one of --venv or --site-packages")

    if args.site_packages is not None:
        site_packages = args.site_packages.expanduser().resolve()
        if not site_packages.is_dir():
            raise RuntimeError(f"site-packages directory not found: {site_packages}")
        return site_packages

    venv_root = args.venv.expanduser().resolve()
    if not venv_root.is_dir():
        raise RuntimeError(f"Virtualenv root not found: {venv_root}")

    matches = sorted(venv_root.glob("lib/python*/site-packages"))
    if not matches:
        raise RuntimeError(f"Could not find site-packages under {venv_root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple site-packages directories found under {venv_root}: {matches}"
        )
    return matches[0]


def main() -> int:
    args = parse_args()
    overlay_root = args.overlay_root.expanduser().resolve()
    if not overlay_root.is_dir():
        print(f"Overlay root not found: {overlay_root}", file=sys.stderr)
        return 1

    try:
        site_packages = resolve_site_packages(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    copied = 0
    for source_path in sorted(overlay_root.rglob("*")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(overlay_root)
        destination_path = site_packages / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied += 1

    print(f"Installed {copied} overlay files into {site_packages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
