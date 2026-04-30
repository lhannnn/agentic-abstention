from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_delayed_abstention_job import classify_delay_bucket, load_manifest
from scripts.validate_delayed_case_pack import validate_rewrite_only_reviews, validate_specs

DATASET_DIR = ROOT / "data" / "delayed"


def test_classify_delay_bucket() -> None:
    assert classify_delay_bucket("abstain", 1, 2, 1) == "early"
    assert classify_delay_bucket("abstain", 2, 2, 1) == "timely"
    assert classify_delay_bucket("abstain", 3, 2, 1) == "timely"
    assert classify_delay_bucket("abstain", 4, 2, 1) == "late"
    assert classify_delay_bucket("continue", None, 2, 1) == "never"
    assert classify_delay_bucket("no_decision", None, 2, 1) == "never"
    assert classify_delay_bucket("continue", None, None, None) is None


def test_delayed_case_pack_metadata_validates() -> None:
    rows = load_manifest(DATASET_DIR / "manifest.jsonl")
    delayed_rows = [row for row in rows if row["abstention_timing"] == "delayed"]
    assert len(rows) >= 20
    assert len(delayed_rows) >= 10
    validate_specs(DATASET_DIR, delayed_rows)
    validate_rewrite_only_reviews(DATASET_DIR, delayed_rows)


def test_rewrite_only_consensus_has_expected_accepted_count() -> None:
    payload = json.loads((DATASET_DIR / "reviews" / "rewrite_only_consensus.json").read_text(encoding="utf-8"))
    assert len(payload["accepted"]) == 21
    assert "gcode-to-text__delayed_observation" in set(payload["accepted"])
    assert "build-pov-ray__delayed_execution" in set(payload["backlog"])
