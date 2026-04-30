from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "observe_act_turns.py"
SPEC = spec_from_file_location("observe_act_turns", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_trial(
    trial_dir: Path,
    *,
    task_name: str,
    agent_name: str = "codex",
    metadata: dict[str, object] | None = None,
    events: list[dict[str, object]] | None = None,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "trial_name": f"{task_name}__trial",
                "agent_info": {"name": agent_name, "version": "0.120.0"},
                "agent_result": {"metadata": metadata or {}},
                "exception_info": None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if events is not None:
        session_dir = trial_dir / "agent" / "sessions" / "2026" / "04" / "15"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / "rollout.jsonl"
        session_file.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
            encoding="utf-8",
        )


def write_trajectory(trial_dir: Path, *, agent_name: str, steps: list[dict[str, object]]) -> None:
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "sess-1",
                "agent": {"name": agent_name, "version": "test", "model_name": "dummy"},
                "steps": steps,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ObserveActTurnTests(unittest.TestCase):
    def test_immediate_abstain_is_one_turn(self):
        with TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial"
            write_trial(
                trial_dir,
                task_name="case-1",
                metadata={"n_interaction_rounds": 1, "decision": "abstain"},
                events=[
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"text": "ABSTAIN underspecified_intent: missing target"}],
                        },
                    }
                ],
            )

            metrics = MODULE.compute_trial_turn_metrics(trial_dir)
            self.assertEqual(metrics["observe_act_turns"], 1)
            self.assertEqual(metrics["first_abstain_turn"], 1)
            self.assertEqual(metrics["observe_act_turn_source"], "codex_session_jsonl")

    def test_assistant_after_new_observation_counts_second_turn(self):
        with TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial"
            write_trial(
                trial_dir,
                task_name="case-2",
                metadata={"n_interaction_rounds": 1, "decision": "abstain"},
                events=[
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"text": "I will inspect the archive."}],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "shell",
                            "arguments": "{\"cmd\": \"ls\"}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": "{\"output\": \"file1\\nfile2\"}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "text": "ABSTAIN false_premise_or_contradiction: output conflicts with instruction"
                                }
                            ],
                        },
                    },
                ],
            )

            metrics = MODULE.compute_trial_turn_metrics(trial_dir)
            self.assertEqual(metrics["observe_act_turns"], 2)
            self.assertEqual(metrics["first_abstain_turn"], 2)

    def test_multiple_tool_calls_without_new_assistant_decision_stay_in_same_turn(self):
        with TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial"
            write_trial(
                trial_dir,
                task_name="case-3",
                metadata={"n_interaction_rounds": 1, "decision": "continue"},
                events=[
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"text": "I will check several files."}],
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "shell",
                            "arguments": "{\"cmd\": \"ls\"}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": "{\"output\": \"a\\nb\"}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call-2",
                            "name": "shell",
                            "arguments": "{\"cmd\": \"cat a\"}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-2",
                            "output": "{\"output\": \"contents\"}",
                        },
                    },
                ],
            )

            metrics = MODULE.compute_trial_turn_metrics(trial_dir)
            self.assertEqual(metrics["observe_act_turns"], 1)
            self.assertIsNone(metrics["first_abstain_turn"])

    def test_missing_trace_falls_back_to_legacy_rounds(self):
        with TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial"
            write_trial(
                trial_dir,
                task_name="case-4",
                metadata={"n_interaction_rounds": 7, "decision": "continue"},
                events=None,
            )

            metrics = MODULE.compute_trial_turn_metrics(trial_dir)
            self.assertEqual(metrics["observe_act_turns"], 7)
            self.assertEqual(metrics["observe_act_turn_source"], "legacy_fallback")
            self.assertIsNone(metrics["first_abstain_turn"])

    def test_terminus_2_trajectory_uses_atif_step_adapter(self):
        with TemporaryDirectory() as tmpdir:
            trial_dir = Path(tmpdir) / "trial"
            write_trial(
                trial_dir,
                task_name="case-5",
                agent_name="terminus-2",
                metadata={"n_interaction_rounds": 3, "decision": "abstain"},
                events=None,
            )
            write_trajectory(
                trial_dir,
                agent_name="terminus-2",
                steps=[
                    {
                        "step_id": 1,
                        "source": "user",
                        "message": "initial prompt",
                    },
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "Analysis: inspect files",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "bash_command",
                                "arguments": {"keystrokes": "ls\n", "duration": 0.1},
                            }
                        ],
                        "observation": {"results": [{"content": "file list"}]},
                    },
                    {
                        "step_id": 3,
                        "source": "agent",
                        "message": "Analysis: still checking",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-2",
                                "function_name": "bash_command",
                                "arguments": {"keystrokes": "cat x\n", "duration": 0.1},
                            }
                        ],
                        "observation": {"results": [{"content": "contents"}]},
                    },
                    {
                        "step_id": 4,
                        "source": "agent",
                        "message": "Analysis: the target metric is missing, so the task is underspecified.",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-3",
                                "function_name": "mark_task_abstained",
                                "arguments": {
                                    "abstain_category": "underspecified_intent",
                                    "abstain_reason": "missing target metric",
                                },
                            }
                        ],
                        "observation": {
                            "results": [{"content": "Agent abstained. Verification will be skipped."}]
                        },
                    },
                ],
            )

            metrics = MODULE.compute_trial_turn_metrics(trial_dir)
            self.assertEqual(metrics["observe_act_turns"], 3)
            self.assertEqual(metrics["first_abstain_turn"], 3)
            self.assertEqual(metrics["observe_act_turn_source"], "atif_step_adapter")


if __name__ == "__main__":
    unittest.main()
