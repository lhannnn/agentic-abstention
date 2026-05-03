import json

from recipe.abstention import Response, Responses
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.action_space import ACTION_ABSTAIN, ACTION_ANSWER, ACTION_INVALID, ACTION_TOOL_CALL
from recipe.wikimedia_evaluation import WikimediaEpisodeMetricsEvaluator
from recipe.wikimedia_exclude_invalid_metrics import build_episode_metrics_excluding_invalid
from recipe.wikimedia_json_actions import parse_wikimedia_json_response
from recipe.wikimedia_multiturn import EpisodeTrace


def _trace(request_ordinal, action, termination_reason, first_action=None):
    first_action = first_action or action
    return EpisodeTrace.model_validate(
        {
            "request_ordinal": request_ordinal,
            "original_question": f"Q{request_ordinal}",
            "initial_agent_prompt": "Prompt",
            "turns": [
                {
                    "turn_index": 1,
                    "agent_raw_response": json.dumps({"action": first_action}),
                    "parsed_action": action,
                }
            ],
            "termination_reason": termination_reason,
            "num_turns": 1,
            "final_action": action,
            "final_response": json.dumps({"action": action}),
            "final_success": None,
        }
    )


def test_wikimedia_json_parser_accepts_release_actions():
    assert parse_wikimedia_json_response('{"action":"ABSTAIN","reason":"missing"}').predicted_action == ACTION_ABSTAIN
    assert parse_wikimedia_json_response('{"action":"ANSWER","answer":"A"}').predicted_action == ACTION_ANSWER
    search = parse_wikimedia_json_response('{"action":"SEARCH","search_query":"entity"}')
    assert search.predicted_action == ACTION_TOOL_CALL
    assert search.search_query == "entity"


def test_episode_metrics_and_exclude_invalid_paths():
    abstain_prompt = Prompt(
        question="Q",
        should_abstain=True,
        reference_answers=None,
        metadata={},
    )
    responses = Responses(
        responses=[
            Response(
                prompt=abstain_prompt,
                response='{"action":"ABSTAIN","reason":"missing"}',
                response_or_abstention='{"action":"ABSTAIN","reason":"missing"}',
                predicted_action=ACTION_ABSTAIN,
            ),
            Response(
                prompt=abstain_prompt,
                response='not-json',
                response_or_abstention='not-json',
                predicted_action=ACTION_INVALID,
            ),
        ]
    )
    traces = [
        _trace(0, ACTION_ABSTAIN, "abstain"),
        _trace(1, ACTION_INVALID, "invalid_action"),
    ]

    evaluator = WikimediaEpisodeMetricsEvaluator(save_dir="/tmp")
    records = evaluator.build_episode_metrics(responses, traces)
    summary = evaluator.summarize(records)
    assert summary.num_abstain_samples == 2
    assert summary.pass_at_k["pass_at_0"] == 0.5
    assert summary.pass_at_k["pass_at_10"] == 0.5

    filtered, exclude_invalid_summary = build_episode_metrics_excluding_invalid(
        traces,
        responses,
    )
    assert len(filtered) == 1
    assert filtered[0].request_ordinal == 0
    assert exclude_invalid_summary.excluded_invalid_count == 1
