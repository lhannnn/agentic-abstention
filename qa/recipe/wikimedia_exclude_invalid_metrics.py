"""
Offline Wikimedia abstain-only metrics that exclude INVALID predictions from the
denominator for one-off experiments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from recipe.abstention import Responses
from recipe.action_space import ACTION_ABSTAIN, ACTION_INVALID
from recipe.inference import LoadSaveDataMixin
from recipe.utils import permissive_makedirs
from recipe.wikimedia_evaluation import EpisodeMetricsRecord, WikimediaEpisodeMetricsEvaluator
from recipe.wikimedia_multiturn import EpisodeTrace


class WikimediaEpisodeMetricsSummaryExcludeInvalid(BaseModel, LoadSaveDataMixin):
    original_num_abstain_samples: int
    excluded_invalid_count: int
    valid_num_abstain_samples: int
    num_abstain_samples: int
    early_recall: Optional[float]
    overall_recall: Optional[float]
    spl: Optional[float]
    average_search_calls: Optional[float]
    average_decision_turns: Optional[float]
    forced_stop_count: int
    forced_stop_rate: Optional[float]
    pass_at_k: dict[str, Optional[float]]
    search_call_distribution: dict[str, int]


def load_wikimedia_run_artifacts(
    result_dir: str | Path,
) -> tuple[Optional[Responses], list[EpisodeTrace]]:
    result_dir = Path(result_dir)
    trace_path = result_dir / "MultiTurnEpisodeTraces.jsonl"
    traces = [
        EpisodeTrace.model_validate_json(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    detector_path = result_dir / "WikimediaJsonAbstentionDetector.json"
    responses = None
    if detector_path.exists():
        responses = Responses.load(detector_path)
        if len(responses.responses) != len(traces):
            raise ValueError(
                "Response count does not match trace count for Wikimedia exclude-invalid summary."
            )
    return responses, traces


def build_episode_metrics_excluding_invalid(
    traces: list[EpisodeTrace],
    responses: Optional[Responses] = None,
    *,
    assume_all_should_abstain: bool = False,
) -> tuple[list[EpisodeMetricsRecord], WikimediaEpisodeMetricsSummaryExcludeInvalid]:
    if responses is None and not assume_all_should_abstain:
        raise ValueError(
            "Responses are required unless assume_all_should_abstain=True."
        )

    evaluator = WikimediaEpisodeMetricsEvaluator(save_dir=".")
    episode_metrics: list[EpisodeMetricsRecord] = []
    original_num_abstain_samples = 0
    excluded_invalid_count = 0

    iterator = zip(responses.responses, traces) if responses is not None else (
        (None, trace) for trace in traces
    )

    for response, trace in iterator:
        should_abstain = (
            response.prompt.should_abstain if response is not None else True
        )
        if not should_abstain:
            continue

        original_num_abstain_samples += 1
        predicted_action = (
            response.predicted_action if response is not None else None
        ) or trace.final_action
        if predicted_action == ACTION_INVALID:
            excluded_invalid_count += 1
            continue

        oracle_abstain_step = 1
        first_abstain_step = next(
            (
                turn.turn_index
                for turn in trace.turns
                if turn.parsed_action == ACTION_ABSTAIN
            ),
            None,
        )
        search_calls_used = max(
            [turn.search_call_index or 0 for turn in trace.turns],
            default=0,
        )
        decision_turns = trace.num_turns
        success = first_abstain_step is not None and first_abstain_step >= oracle_abstain_step
        spl_contribution = (
            oracle_abstain_step / max(first_abstain_step, oracle_abstain_step)
            if success and first_abstain_step is not None
            else 0.0
        )

        episode_metrics.append(
            EpisodeMetricsRecord(
                request_ordinal=trace.request_ordinal,
                should_abstain=True,
                oracle_abstain_step=oracle_abstain_step,
                first_abstain_step=first_abstain_step,
                search_calls_used=search_calls_used,
                decision_turns=decision_turns,
                success=success,
                termination_reason=trace.termination_reason,
                spl_contribution=spl_contribution,
            )
        )

    base_summary = evaluator.summarize(episode_metrics)
    summary = WikimediaEpisodeMetricsSummaryExcludeInvalid(
        original_num_abstain_samples=original_num_abstain_samples,
        excluded_invalid_count=excluded_invalid_count,
        valid_num_abstain_samples=len(episode_metrics),
        **base_summary.model_dump(mode="json"),
    )
    return episode_metrics, summary


def save_episode_metrics_excluding_invalid(
    result_dir: str | Path,
    episode_metrics: list[EpisodeMetricsRecord],
    summary: WikimediaEpisodeMetricsSummaryExcludeInvalid,
) -> None:
    result_dir = Path(result_dir)
    permissive_makedirs(result_dir)

    metrics_path = result_dir / "WikimediaEpisodeMetricsExcludeInvalid.json"
    metrics_path.write_text(
        json.dumps(
            [record.model_dump(mode="json") for record in episode_metrics],
            indent=2,
        )
    )

    summary.save(str(result_dir), "WikimediaEpisodeMetricsSummaryExcludeInvalid")

    flattened = summary.model_dump(mode="json")
    pass_at_k = flattened.pop("pass_at_k")
    search_call_distribution = flattened.pop("search_call_distribution")
    flattened.update(pass_at_k)
    for search_calls, count in search_call_distribution.items():
        flattened[f"search_call_count_{search_calls}"] = count

    csv_path = result_dir / "WikimediaEpisodeMetricsSummaryExcludeInvalid.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)
