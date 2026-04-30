from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import toml  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("toml is not installed; run pip install -r requirements.txt") from exc


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = (
    ROOT / "scripts" / "materialize_harbor_instruction_level_abstention_dataset.py"
)
SPEC = spec_from_file_location("instruction_level_materialize", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_source_task(cache_root: Path, namespace: str, task_name: str) -> Path:
    task_dir = cache_root / namespace / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("original instruction", encoding="utf-8")
    (task_dir / "task.toml").write_text("", encoding="utf-8")
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "environment").mkdir()
    return task_dir


class InstructionLevelMaterializePartialTests(unittest.TestCase):
    def test_load_rows_allow_partial_skips_expected_count_check(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "partial.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "task_name": "task-a",
                        "source_task_name": "task-a",
                        "variant": "original",
                        "instruction_level_category": "original",
                        "instruction": "original a",
                    },
                    {
                        "task_name": "task-a__underspecified_intent",
                        "source_task_name": "task-a",
                        "variant": "rewritten",
                        "instruction_level_category": "underspecified_intent",
                        "instruction": "rewrite a",
                    },
                ],
            )

            rows = MODULE.load_rows(
                input_path,
                expected_count=267,
                allow_partial=True,
            )

            self.assertEqual(len(rows), 2)

    def test_materialize_partial_copies_skipped_report(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            output_root = root / "out"
            input_path = root / "partial.jsonl"
            skipped_input = root / "partial.skipped.jsonl"

            make_source_task(cache_root, "ns", "task-a")
            write_jsonl(
                input_path,
                [
                    {
                        "task_name": "task-a",
                        "source_task_name": "task-a",
                        "variant": "original",
                        "instruction_level_category": "original",
                        "instruction": "original a",
                    },
                    {
                        "task_name": "task-a__false_premise_or_contradiction",
                        "source_task_name": "task-a",
                        "variant": "rewritten",
                        "instruction_level_category": "false_premise_or_contradiction",
                        "instruction": "rewrite a",
                    },
                ],
            )
            write_jsonl(
                skipped_input,
                [
                    {
                        "task_name": "task-b__underspecified_intent",
                        "source_task_name": "task-b",
                        "instruction_level_category": "underspecified_intent",
                        "variant": "rewritten",
                        "status": "skipped",
                        "failed_stage": "validation",
                        "failed_attempt": 2,
                        "error": "failed",
                        "last_failure_codes": ["other"],
                        "last_rewrite_guidance": "guidance",
                        "cache_path": "/tmp/cache.json",
                    }
                ],
            )

            rows = MODULE.load_rows(
                input_path,
                expected_count=267,
                allow_partial=True,
            )
            source_index = MODULE.index_cache(cache_root)
            output_root.mkdir()
            manifest_rows = MODULE.materialize_rows(rows, source_index, output_root)
            MODULE.validate_output(output_root, manifest_rows, allow_partial=True)
            manifest_path = MODULE.write_manifest(output_root, manifest_rows)
            copied_path = MODULE.copy_skipped_report(skipped_input, output_root)

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(manifest_rows), 2)
            self.assertTrue((output_root / "task-a").is_dir())
            self.assertTrue(
                (output_root / "task-a__false_premise_or_contradiction").is_dir()
            )
            self.assertEqual(copied_path, output_root / "skipped_rewrites.jsonl")
            self.assertEqual(
                copied_path.read_text(encoding="utf-8"),
                skipped_input.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
