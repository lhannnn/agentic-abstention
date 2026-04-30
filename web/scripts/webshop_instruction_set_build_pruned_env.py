#!/usr/bin/env python3
"""Build the shared pruned WebShop environment for the 251 missing-target tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

from webshop_abstain_common import (
    DEFAULT_MISSING_TARGET251_MANIFEST_PATH,
    DEFAULT_PRUNED_DOCS_PATH,
    DEFAULT_PRUNED_ENV_METADATA_PATH,
    DEFAULT_PRUNED_ENV_ROOT,
    DEFAULT_PRUNED_INDEX_DIR,
    DEFAULT_PRUNED_ITEMS_PATH,
    default_full_attr_candidates,
    default_full_items_candidates,
    default_human_candidates,
    ensure_parent_dir,
    load_jsonl,
    resolve_existing_path,
    utc_now_iso,
    write_json,
)


def default_full_documents_candidates() -> list[Path]:
    return [
        Path(__file__).resolve().parents[1] / "external" / "WebShop" / "search_engine" / "resources" / "documents.jsonl",
        Path("/usr/src/webshop/search_engine/resources/documents.jsonl"),
    ]


def iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_array = False
        done = False

        while not done:
            chunk = handle.read(1 << 20)
            if chunk:
                buffer += chunk
            else:
                done = True

            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                if not in_array:
                    if stripped[0] != "[":
                        raise ValueError(f"{path} does not begin with a JSON array.")
                    in_array = True
                    buffer = stripped[1:]
                    continue

                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break

                if stripped[0] == "]":
                    return
                if stripped[0] == ",":
                    buffer = stripped[1:]
                    continue

                try:
                    record, consumed = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    if done:
                        raise
                    break
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object inside array in {path}")
                yield record
                buffer = stripped[consumed:]


def filter_items_json(items_path: Path, output_path: Path, removed_asins: set[str]) -> Dict[str, int]:
    ensure_parent_dir(output_path)
    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for row in iter_json_array(items_path):
            total_rows += 1
            asin = row.get("asin")
            if isinstance(asin, str) and asin in removed_asins:
                removed_rows += 1
                continue

            if not first:
                handle.write(",\n")
            handle.write(json.dumps(row, ensure_ascii=False))
            kept_rows += 1
            first = False
        handle.write("\n]\n")

    return {
        "source_item_count": total_rows,
        "kept_item_count": kept_rows,
        "removed_item_count": removed_rows,
    }


def filter_documents_jsonl(documents_path: Path, output_path: Path, removed_asins: set[str]) -> Dict[str, int]:
    ensure_parent_dir(output_path)
    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    with documents_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total_rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {documents_path}:{line_number}") from exc

            asin = row.get("id")
            if not isinstance(asin, str):
                product = row.get("product")
                if isinstance(product, dict):
                    asin = product.get("asin")
            if isinstance(asin, str) and asin in removed_asins:
                removed_rows += 1
                continue

            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")
            kept_rows += 1

    return {
        "source_document_count": total_rows,
        "kept_document_count": kept_rows,
        "removed_document_count": removed_rows,
    }


def build_lucene_index(
    *,
    python_bin: str,
    resources_dir: Path,
    index_dir: Path,
    threads: int,
) -> None:
    if index_dir.exists():
        if any(index_dir.iterdir()):
            raise RuntimeError(
                f"Index directory already exists and is not empty: {index_dir}. "
                "Remove it manually or choose a different --output-root."
            )
    else:
        index_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_bin,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(resources_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(max(1, threads)),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-target-manifest", default=str(DEFAULT_MISSING_TARGET251_MANIFEST_PATH))
    parser.add_argument("--items-path", help="Path to the full items_shuffle.json")
    parser.add_argument("--documents-path", help="Path to the full search_engine/resources/documents.jsonl")
    parser.add_argument("--attr-path", help="Path to the full items_ins_v2.json")
    parser.add_argument("--human-path", help="Path to the full items_human_ins.json")
    parser.add_argument("--output-root", default=str(DEFAULT_PRUNED_ENV_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--index-threads", type=int, default=1)
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Only build the pruned items/documents files and metadata; do not rebuild Lucene indexes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing_manifest_path = Path(args.missing_target_manifest).expanduser().resolve()
    missing_rows = load_jsonl(missing_manifest_path)
    removed_asins = sorted({str(row["target_asin"]) for row in missing_rows})
    if len(missing_rows) != 251:
        raise RuntimeError(f"Expected 251 missing-target rows, found {len(missing_rows)}")

    items_path = resolve_existing_path(
        args.items_path,
        default_full_items_candidates(),
        "Could not find full items_shuffle.json. Pass --items-path explicitly.",
    )
    documents_path = resolve_existing_path(
        args.documents_path,
        default_full_documents_candidates(),
        "Could not find full search_engine/resources/documents.jsonl. Pass --documents-path explicitly.",
    )
    attr_path = resolve_existing_path(
        args.attr_path,
        default_full_attr_candidates(),
        "Could not find full items_ins_v2.json. Pass --attr-path explicitly.",
    )
    human_path = resolve_existing_path(
        args.human_path,
        default_human_candidates(),
        "Could not find items_human_ins.json. Pass --human-path explicitly.",
    )

    output_root = Path(args.output_root).expanduser().resolve()
    pruned_items_path = output_root / DEFAULT_PRUNED_ITEMS_PATH.name
    pruned_docs_path = output_root / "search_engine" / "resources" / "documents.jsonl"
    pruned_index_dir = output_root / "search_engine" / "indexes"
    metadata_path = output_root / DEFAULT_PRUNED_ENV_METADATA_PATH.name

    output_root.mkdir(parents=True, exist_ok=True)

    item_stats = filter_items_json(items_path, pruned_items_path, set(removed_asins))
    document_stats = filter_documents_jsonl(documents_path, pruned_docs_path, set(removed_asins))

    index_built = False
    if not args.skip_indexes:
        build_lucene_index(
            python_bin=args.python_bin,
            resources_dir=pruned_docs_path.parent,
            index_dir=pruned_index_dir,
            threads=args.index_threads,
        )
        index_built = True

    metadata = {
        "created_at": utc_now_iso(),
        "env_variant": "pruned_missing_target_251",
        "missing_target_manifest_path": str(missing_manifest_path),
        "source_items_path": str(items_path),
        "source_documents_path": str(documents_path),
        "source_attr_path": str(attr_path),
        "source_human_path": str(human_path),
        "pruned_items_path": str(pruned_items_path),
        "pruned_documents_path": str(pruned_docs_path),
        "pruned_index_dir": str(pruned_index_dir),
        "bootstrap_goal_index": 0,
        "removed_target_asin_count": len(removed_asins),
        "removed_target_asins": removed_asins,
        "index_built": index_built,
        "item_stats": item_stats,
        "document_stats": document_stats,
    }
    write_json(metadata_path, metadata)

    print(f"Wrote pruned items file to {pruned_items_path}")
    print(f"Wrote pruned documents file to {pruned_docs_path}")
    if index_built:
        print(f"Built pruned Lucene index at {pruned_index_dir}")
    else:
        print(f"Skipped Lucene indexing. Metadata still written to {metadata_path}")
    print(f"Wrote pruned environment metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
