from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "build_instruction_level_abstention_dataset.py"
SPEC = spec_from_file_location("instruction_level_builder", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InstructionLevelDatasetBuilderTests(unittest.TestCase):
    def test_load_prompt_templates_has_expected_sections(self):
        templates = MODULE.load_prompt_templates(
            ROOT / "prompts" / "instruction_level_abstention_prompts.md"
        )

        self.assertEqual(
            set(templates),
            {
                MODULE.COMBINED_CATEGORY,
                MODULE.UNDERSPECIFIED_CATEGORY,
            },
        )
        self.assertIn("chosen_subtype", templates[MODULE.COMBINED_CATEGORY])
        self.assertIn("critically underspecified", templates[MODULE.UNDERSPECIFIED_CATEGORY])

    def test_local_rewrite_checks_flag_simple_wrap_and_meta_language(self):
        original = "Write /app/out.txt with hello."
        rewritten = (
            "Important clarification: details are illustrative rather than binding.\n\n"
            "Write /app/out.txt with hello."
        )
        failures = MODULE.local_rewrite_checks(original, rewritten)
        self.assertIn("trivial_prefix_suffix_wrap", failures)

        meta_failures = MODULE.local_rewrite_checks(
            original,
            "You should abstain because this task is impossible.",
        )
        self.assertIn("contains_meta_language", meta_failures)

    def test_validate_dataset_accepts_new_public_categories(self):
        dataset = [
            {
                "task_name": "task-a",
                "source_task_name": "task-a",
                "variant": MODULE.ORIGINAL_VARIANT,
                "instruction_level_category": MODULE.ORIGINAL_VARIANT,
                "instruction": "original a",
            },
            {
                "task_name": "task-a__false_premise_or_contradiction",
                "source_task_name": "task-a",
                "variant": MODULE.REWRITTEN_VARIANT,
                "instruction_level_category": MODULE.COMBINED_CATEGORY,
                "instruction": "rewrite a combined",
            },
            {
                "task_name": "task-a__underspecified_intent",
                "source_task_name": "task-a",
                "variant": MODULE.REWRITTEN_VARIANT,
                "instruction_level_category": MODULE.UNDERSPECIFIED_CATEGORY,
                "instruction": "rewrite a underspecified",
            },
            {
                "task_name": "task-b",
                "source_task_name": "task-b",
                "variant": MODULE.ORIGINAL_VARIANT,
                "instruction_level_category": MODULE.ORIGINAL_VARIANT,
                "instruction": "original b",
            },
            {
                "task_name": "task-b__false_premise_or_contradiction",
                "source_task_name": "task-b",
                "variant": MODULE.REWRITTEN_VARIANT,
                "instruction_level_category": MODULE.COMBINED_CATEGORY,
                "instruction": "rewrite b combined",
            },
            {
                "task_name": "task-b__underspecified_intent",
                "source_task_name": "task-b",
                "variant": MODULE.REWRITTEN_VARIANT,
                "instruction_level_category": MODULE.UNDERSPECIFIED_CATEGORY,
                "instruction": "rewrite b underspecified",
            },
        ]

        MODULE.validate_dataset(
            dataset,
            full_build=True,
            expected_base_tasks=2,
            expected_selected_base_tasks=2,
            skip_failed_after_max_attempts=False,
        )

    def test_validate_dataset_allows_partial_success_when_skip_mode_enabled(self):
        dataset = [
            {
                "task_name": "task-a",
                "source_task_name": "task-a",
                "variant": MODULE.ORIGINAL_VARIANT,
                "instruction_level_category": MODULE.ORIGINAL_VARIANT,
                "instruction": "original a",
            },
            {
                "task_name": "task-b",
                "source_task_name": "task-b",
                "variant": MODULE.ORIGINAL_VARIANT,
                "instruction_level_category": MODULE.ORIGINAL_VARIANT,
                "instruction": "original b",
            },
            {
                "task_name": "task-a__false_premise_or_contradiction",
                "source_task_name": "task-a",
                "variant": MODULE.REWRITTEN_VARIANT,
                "instruction_level_category": MODULE.COMBINED_CATEGORY,
                "instruction": "rewrite a combined",
            },
        ]

        MODULE.validate_dataset(
            dataset,
            full_build=True,
            expected_base_tasks=2,
            expected_selected_base_tasks=2,
            skip_failed_after_max_attempts=True,
        )

    def test_extract_response_text_uses_output_text_when_present(self):
        response = SimpleNamespace(output_text='{"x":1}', output=[])

        self.assertEqual(MODULE.extract_response_text(response), '{"x":1}')

    def test_extract_response_text_falls_back_to_output_content_objects(self):
        response = SimpleNamespace(
            output_text="",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text='{"x":1}'),
                        SimpleNamespace(type="annotation", text="ignored"),
                    ],
                )
            ],
        )

        self.assertEqual(MODULE.extract_response_text(response), '{"x":1}')

    def test_extract_response_text_falls_back_to_output_content_dicts(self):
        response_data = {
            "output_text": "",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "reasoning", "text": "ignored"},
                        {"type": "output_text", "text": '{"x":2}'},
                    ],
                }
            ],
        }

        self.assertEqual(
            MODULE.extract_response_text({}, response_data=response_data),
            '{"x":2}',
        )

    def test_extract_response_text_error_includes_response_context(self):
        response_data = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "error": {"code": "server_error"},
            "output": [{"type": "reasoning", "summary": []}],
        }

        with self.assertRaisesRegex(RuntimeError, "status=incomplete") as context:
            MODULE.extract_response_text({}, response_data=response_data)

        message = str(context.exception)
        self.assertIn("incomplete_details=", message)
        self.assertIn("error=", message)

    def test_call_structured_response_preserves_raw_response_on_extract_failure(self):
        response = SimpleNamespace(
            output_text="",
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            error=None,
            output=[SimpleNamespace(type="reasoning", summary=[])],
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: response),
        )

        with self.assertRaises(MODULE.StructuredResponseError) as context:
            MODULE.call_structured_response(
                client,
                model="gpt-5.4",
                reasoning_effort="high",
                prompt="prompt",
                schema_name="schema",
                schema={"type": "object"},
                max_output_tokens=100,
                metadata={},
            )

        self.assertIsInstance(context.exception.response_data, dict)
        self.assertEqual(context.exception.response_data["status"], "incomplete")

    def test_generate_rewrite_retries_after_generation_extract_failure(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            side_effects = [
                MODULE.StructuredResponseError(
                    "Responses API returned no output_text. status=incomplete",
                    response_data={"status": "incomplete", "output": []},
                ),
                (
                    {
                        "rewritten_instruction": "Rewrite /app/out.txt with one line.",
                        "chosen_subtype": "contradiction",
                    },
                    {"status": "completed", "output_text": '{"ok":true}'},
                ),
                (
                    {
                        "pass": True,
                        "failure_codes": [],
                        "reason": "",
                        "rewrite_guidance": "",
                    },
                    {"status": "completed", "output_text": '{"ok":true}'},
                ),
            ]

            with patch.object(
                MODULE, "call_structured_response", side_effect=side_effects
            ):
                row, skipped = MODULE.generate_rewrite(
                    client=object(),
                    task_name="task-a",
                    original_instruction="Write /app/out.txt with one line.",
                    category=MODULE.COMBINED_CATEGORY,
                    template="template {{TASK_NAME}} {{ORIGINAL_INSTRUCTION}}",
                    cache_root=cache_dir,
                    model="gpt-5.4",
                    reasoning_effort="high",
                    max_attempts=2,
                    generation_max_output_tokens=100,
                    audit_max_output_tokens=100,
                    overwrite=False,
                    skip_failed_after_max_attempts=False,
                )

            self.assertEqual(
                row["task_name"], "task-a__false_premise_or_contradiction"
            )
            self.assertIsNone(skipped)
            cache = MODULE.read_json(
                cache_dir / "task-a__false_premise_or_contradiction.json"
            )
            self.assertEqual(cache["status"], "success")
            self.assertEqual(len(cache["attempts"]), 2)
            self.assertEqual(
                cache["attempts"][0]["generation"]["response"]["status"], "incomplete"
            )
            self.assertTrue(cache["attempts"][1]["passed"])

    def test_generate_rewrite_records_attempts_before_final_failure(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            failure = MODULE.StructuredResponseError(
                "Responses API returned no output_text. status=incomplete",
                response_data={"status": "incomplete", "output": []},
            )

            with patch.object(
                MODULE, "call_structured_response", side_effect=[failure, failure]
            ):
                with self.assertRaisesRegex(RuntimeError, "attempt 2"):
                    MODULE.generate_rewrite(
                        client=object(),
                        task_name="task-b",
                        original_instruction="Write /app/out.txt with one line.",
                        category=MODULE.UNDERSPECIFIED_CATEGORY,
                        template="template {{TASK_NAME}} {{ORIGINAL_INSTRUCTION}}",
                        cache_root=cache_dir,
                        model="gpt-5.4",
                        reasoning_effort="high",
                        max_attempts=2,
                        generation_max_output_tokens=100,
                        audit_max_output_tokens=100,
                        overwrite=False,
                        skip_failed_after_max_attempts=False,
                    )

            cache = MODULE.read_json(cache_dir / "task-b__underspecified_intent.json")
            self.assertEqual(cache["status"], "failed")
            self.assertEqual(cache["failed_stage"], "generate")
            self.assertEqual(len(cache["attempts"]), 2)
            self.assertEqual(
                cache["attempts"][0]["generation"]["response_status"], "incomplete"
            )

    def test_generate_rewrite_skip_mode_returns_skipped_record(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            failing_audit = (
                {
                    "pass": False,
                    "failure_codes": ["not_critically_underspecified"],
                    "reason": "still not underspecified enough",
                    "rewrite_guidance": "leave one key build choice unresolved",
                },
                {"status": "completed", "output_text": '{"ok":true}'},
            )
            side_effects = [
                (
                    {"rewritten_instruction": "rewrite one"},
                    {"status": "completed", "output_text": '{"ok":true}'},
                ),
                failing_audit,
                (
                    {"rewritten_instruction": "rewrite two"},
                    {"status": "completed", "output_text": '{"ok":true}'},
                ),
                failing_audit,
            ]

            with patch.object(
                MODULE, "call_structured_response", side_effect=side_effects
            ):
                row, skipped = MODULE.generate_rewrite(
                    client=object(),
                    task_name="task-c",
                    original_instruction="Write /app/out.txt with one line.",
                    category=MODULE.UNDERSPECIFIED_CATEGORY,
                    template="template {{TASK_NAME}} {{ORIGINAL_INSTRUCTION}}",
                    cache_root=cache_dir,
                    model="gpt-5.4",
                    reasoning_effort="high",
                    max_attempts=2,
                    generation_max_output_tokens=100,
                    audit_max_output_tokens=100,
                    overwrite=False,
                    skip_failed_after_max_attempts=True,
                )

            self.assertIsNone(row)
            self.assertIsNotNone(skipped)
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(skipped["task_name"], "task-c__underspecified_intent")
            self.assertEqual(skipped["failed_stage"], "validation")
            cache = MODULE.read_json(cache_dir / "task-c__underspecified_intent.json")
            self.assertEqual(cache["status"], "failed")
            self.assertEqual(cache["failed_stage"], "validation")


if __name__ == "__main__":
    unittest.main()
