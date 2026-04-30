from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "analyze_paired_abstention_job.py"
SPEC = spec_from_file_location("analyze_paired_abstention_job", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_result(path: Path, task_name: str, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_name": task_name,
                "agent_result": {"metadata": metadata},
                "exception_info": None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class AnalyzePairedAbstentionJobTests(unittest.TestCase):
    def test_build_summary_computes_rewritten_metrics(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.jsonl"
            manifest_rows = []
            for idx in range(10):
                task_name = f"orig-{idx}"
                manifest_rows.append(
                    {
                        "task_name": task_name,
                        "source_task_name": task_name,
                        "variant": "original",
                        "instruction_level_category": "original",
                        "expected_abstain": False,
                        "expected_decision": "continue",
                        "task_dir": str(root / task_name),
                    }
                )
            for idx in range(19):
                source = f"orig-{idx % 10}"
                category = (
                    "false_premise_or_contradiction"
                    if idx < 10
                    else "underspecified_intent"
                )
                manifest_rows.append(
                    {
                        "task_name": f"rew-{idx}",
                        "source_task_name": source,
                        "variant": "rewritten",
                        "instruction_level_category": category,
                        "expected_abstain": True,
                        "expected_decision": "abstain",
                        "task_dir": str(root / f"rew-{idx}"),
                    }
                )
            write_jsonl(manifest_path, manifest_rows)

            job_dir = root / "job"
            for idx in range(10):
                write_result(
                    job_dir / f"orig-{idx}__trial" / "result.json",
                    f"orig-{idx}",
                    {
                        "decision": "continue",
                        "abstain": False,
                        "n_interaction_rounds": 3,
                        "interaction_limit_reached": False,
                    },
                )
            for idx in range(19):
                if idx < 3:
                    metadata = {
                        "decision": "abstain",
                        "abstain": True,
                        "abstain_category": "false_premise_or_contradiction",
                        "n_interaction_rounds": 1,
                        "interaction_limit_reached": False,
                    }
                elif idx < 5:
                    metadata = {
                        "decision": "abstain",
                        "abstain": True,
                        "abstain_category": "underspecified_intent",
                        "n_interaction_rounds": 2,
                        "interaction_limit_reached": False,
                    }
                else:
                    metadata = {
                        "decision": "continue",
                        "abstain": False,
                        "n_interaction_rounds": 20,
                        "interaction_limit_reached": idx == 5,
                    }
                write_result(
                    job_dir / f"rew-{idx}__trial" / "result.json",
                    f"rew-{idx}",
                    metadata,
                )

            loaded_manifest = MODULE.load_manifest(manifest_path)
            trial_results = MODULE.load_trial_results(job_dir)
            summary = MODULE.build_summary(loaded_manifest, trial_results)
            MODULE.validate_summary(summary)

            self.assertEqual(summary["rewritten_task_count"], 19)
            self.assertEqual(summary["original_task_count"], 10)
            self.assertAlmostEqual(summary["metrics"]["timely_recall"], 3 / 19)
            self.assertAlmostEqual(summary["metrics"]["overall_recall"], 5 / 19)
            self.assertAlmostEqual(summary["metrics"]["pass_at_k"]["1"], 3 / 19)
            self.assertAlmostEqual(summary["metrics"]["pass_at_k"]["2"], 5 / 19)
            self.assertAlmostEqual(summary["metrics"]["pass_at_k"]["10"], 5 / 19)
            expected_spl = ((3 * 1.0) + (2 * 0.5)) / 19
            self.assertAlmostEqual(summary["metrics"]["spl"], expected_spl)
            self.assertEqual(summary["counts"]["interaction_limit_reached_count"], 1)
            self.assertEqual(
                summary["turn_metrics"]["source_counts"], {"legacy_fallback": 29}
            )
            self.assertEqual(
                summary["rewritten_per_task"][0]["observe_act_turn_source"],
                "legacy_fallback",
            )


if __name__ == "__main__":
    unittest.main()
