from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "build_paired_abstention_subset.py"
SPEC = spec_from_file_location("build_paired_abstention_subset", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_task(root: Path, name: str, instruction: str) -> Path:
    task_dir = root / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    (task_dir / "task.toml").write_text("", encoding="utf-8")
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "environment").mkdir()
    return task_dir


class BuildPairedAbstentionSubsetTests(unittest.TestCase):
    def test_select_rows_uses_all_rewrites_and_unique_source_originals(self):
        rows = []
        for task_name in ("task-a", "task-b"):
            rows.append(
                {
                    "task_name": task_name,
                    "source_task_name": task_name,
                    "variant": "original",
                    "instruction_level_category": "original",
                    "expected_abstain": False,
                    "expected_decision": "continue",
                    "task_dir": f"/tmp/{task_name}",
                }
            )
            rows.append(
                {
                    "task_name": f"{task_name}__false_premise_or_contradiction",
                    "source_task_name": task_name,
                    "variant": "rewritten",
                    "instruction_level_category": "false_premise_or_contradiction",
                    "expected_abstain": True,
                    "expected_decision": "abstain",
                    "task_dir": f"/tmp/{task_name}__fpoc",
                }
            )
        original_rows, rewritten_rows = MODULE.select_rows(rows)
        self.assertEqual({row["task_name"] for row in original_rows}, {"task-a", "task-b"})
        self.assertEqual(
            {row["task_name"] for row in rewritten_rows},
            {
                "task-a__false_premise_or_contradiction",
                "task-b__false_premise_or_contradiction",
            },
        )

    def test_validate_subset_rejects_missing_original_coverage(self):
        jsonl_rows = [
            {
                "task_name": "task-a",
                "source_task_name": "task-a",
                "variant": "original",
                "instruction_level_category": "original",
                "instruction": "original",
            },
            {
                "task_name": "task-b__underspecified_intent",
                "source_task_name": "task-b",
                "variant": "rewritten",
                "instruction_level_category": "underspecified_intent",
                "instruction": "rewrite",
            },
        ]
        manifest_rows = [
            {
                "task_name": "task-a",
                "source_task_name": "task-a",
                "variant": "original",
                "instruction_level_category": "original",
                "expected_abstain": False,
                "expected_decision": "continue",
                "task_dir": "/tmp/task-a",
            },
            {
                "task_name": "task-b__underspecified_intent",
                "source_task_name": "task-b",
                "variant": "rewritten",
                "instruction_level_category": "underspecified_intent",
                "expected_abstain": True,
                "expected_decision": "abstain",
                "task_dir": "/tmp/task-b__underspecified_intent",
            },
        ]
        with self.assertRaises(RuntimeError):
            MODULE.validate_subset(jsonl_rows, manifest_rows)

    def test_materialize_subset_copies_selected_tasks(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_root = root / "out"
            task_a = make_task(source_root, "task-a", "original a")
            task_b = make_task(source_root, "task-b__underspecified_intent", "rewrite b")
            rows = [
                {
                    "task_name": "task-a",
                    "source_task_name": "task-a",
                    "variant": "original",
                    "instruction_level_category": "original",
                    "expected_abstain": False,
                    "expected_decision": "continue",
                    "task_dir": str(task_a),
                },
                {
                    "task_name": "task-b__underspecified_intent",
                    "source_task_name": "task-b",
                    "variant": "rewritten",
                    "instruction_level_category": "underspecified_intent",
                    "expected_abstain": True,
                    "expected_decision": "abstain",
                    "task_dir": str(task_b),
                },
            ]
            manifest_rows = MODULE.materialize_subset(rows, output_root)
            self.assertTrue((output_root / "task-a" / "instruction.md").is_file())
            self.assertTrue(
                (output_root / "task-b__underspecified_intent" / "instruction.md").is_file()
            )
            self.assertEqual(len(manifest_rows), 2)


if __name__ == "__main__":
    unittest.main()
