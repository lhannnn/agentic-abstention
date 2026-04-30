#!/usr/bin/env python3
"""Export real WebShop human-goal indices from the full product data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from webshop_abstain_common import (
    DEFAULT_FULL_GOAL_MANIFEST_PATH,
    default_full_items_candidates,
    default_human_candidates,
    ensure_parent_dir,
    load_json,
    resolve_existing_path,
    write_jsonl,
)


def default_documents_candidates():
    return [
        Path(__file__).resolve().parents[1] / "external" / "WebShop" / "search_engine" / "resources" / "documents.jsonl",
        Path("/usr/src/webshop/search_engine/resources/documents.jsonl"),
    ]


def build_real_goal_manifest_from_documents(documents_path: Path):
    records = []
    goal_index = 0
    with documents_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            document = json.loads(stripped)
            product = document.get("product")
            if not isinstance(product, dict):
                continue
            asin = product.get("asin")
            instructions = product.get("instructions")
            if not isinstance(instructions, list):
                continue
            for instruction_idx_within_asin, instruction_record in enumerate(instructions):
                if not isinstance(instruction_record, dict):
                    continue
                instruction_attributes = instruction_record.get("instruction_attributes") or []
                if not instruction_attributes:
                    continue

                records.append(
                    {
                        "goal_index": goal_index,
                        "source_id": f"{asin}:{instruction_idx_within_asin}",
                        "asin": asin,
                        "instruction_idx_within_asin": instruction_idx_within_asin,
                        "instruction": instruction_record.get("instruction"),
                        "instruction_attributes": instruction_attributes,
                        "instruction_options": instruction_record.get("instruction_options", []),
                        "query": product.get("query"),
                        "product_name": product.get("name") or product.get("Title"),
                        "product_category": product.get("product_category"),
                    }
                )
                goal_index += 1
    return records


def build_real_goal_manifest(items_path: Path, human_path: Path):
    items = load_json(items_path)
    human = load_json(human_path)
    if not isinstance(items, list):
        raise ValueError(f"Expected product list in {items_path}")
    if not isinstance(human, dict):
        raise ValueError(f"Expected human instruction dict in {human_path}")

    seen_asins = set()
    goal_index = 0
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        asin = item.get("asin")
        if not isinstance(asin, str) or asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        instructions = human.get(asin)
        if not isinstance(instructions, list):
            continue

        for instruction_idx_within_asin, instruction_record in enumerate(instructions):
            if not isinstance(instruction_record, dict):
                continue
            instruction_attributes = instruction_record.get("instruction_attributes") or []
            if not instruction_attributes:
                continue

            records.append(
                {
                    "goal_index": goal_index,
                    "source_id": f"{asin}:{instruction_idx_within_asin}",
                    "asin": asin,
                    "instruction_idx_within_asin": instruction_idx_within_asin,
                    "instruction": instruction_record.get("instruction"),
                    "instruction_attributes": instruction_attributes,
                    "instruction_options": instruction_record.get("instruction_options", []),
                    "query": item.get("query"),
                    "product_name": item.get("name"),
                    "product_category": item.get("product_category"),
                }
            )
            goal_index += 1
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-path", help="Path to WebShop search_engine/resources/documents.jsonl")
    parser.add_argument("--items-path", help="Path to full WebShop items_shuffle.json")
    parser.add_argument("--human-path", help="Path to full WebShop items_human_ins.json")
    parser.add_argument("--output", default=str(DEFAULT_FULL_GOAL_MANIFEST_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document_candidates = default_documents_candidates()
    documents_path = None
    if args.documents_path or any(path.exists() for path in document_candidates):
        documents_path = resolve_existing_path(
            args.documents_path,
            document_candidates,
            "Could not find WebShop search resources documents.jsonl. Pass --documents-path explicitly.",
        )

    if documents_path is not None:
        records = build_real_goal_manifest_from_documents(documents_path)
    else:
        items_path = resolve_existing_path(
            args.items_path,
            default_full_items_candidates(),
            "Could not find full items_shuffle.json. Pass --items-path pointing to the full WebShop product file.",
        )
        human_path = resolve_existing_path(
            args.human_path,
            default_human_candidates(),
            "Could not find items_human_ins.json. Pass --human-path pointing to the full WebShop human instructions file.",
        )
        records = build_real_goal_manifest(items_path, human_path)
    source_ids = {record["source_id"] for record in records}
    if len(source_ids) != len(records):
        raise RuntimeError("Duplicate source_id detected in real goal manifest.")
    goal_indices = {int(record["goal_index"]) for record in records}
    if len(goal_indices) != len(records):
        raise RuntimeError("Duplicate goal_index detected in real goal manifest.")

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} real goal records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
