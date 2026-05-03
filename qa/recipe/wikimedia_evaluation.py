"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from recipe.abstention import Response, Responses
from recipe.action_space import ACTION_ABSTAIN
from recipe.evaluation import AbstentionEvaluator
from recipe.inference import LoadSaveDataMixin
from recipe.utils import permissive_makedirs
from recipe.wikimedia_multiturn import EpisodeTrace


class WikimediaEpisodeMetricsSummary(BaseModel, LoadSaveDataMixin):
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


class EpisodeMetricsRecord(BaseModel):
    request_ordinal: int
    should_abstain: bool
    oracle_abstain_step: int
    first_abstain_step: Optional[int]
    search_calls_used: int
    decision_turns: int
    success: bool
    termination_reason: str
    spl_contribution: float


class WikimediaEpisodeMetricsEvaluator(AbstentionEvaluator):
    SUMMARY_FILE_NAME = "WikimediaEpisodeMetricsSummary"
    EPISODE_METRICS_FILE_NAME = "WikimediaEpisodeMetrics"
    TRACE_FILE_NAME = "MultiTurnEpisodeTraces.jsonl"

    def is_abstention_correct(self, response: Response) -> bool:
        return response.prompt.should_abstain and response.predicted_action == ACTION_ABSTAIN

    def run(self, responses: Responses) -> Responses:
        responses = super().run(responses)
        traces = self._load_episode_traces()
        if len(traces) != len(responses.responses):
            raise ValueError(
                "Episode trace count does not match evaluated responses count."
            )

        episode_metrics = self.build_episode_metrics(responses, traces)
        self._save_episode_metrics(episode_metrics)
        summary = self.summarize(episode_metrics)
        summary.save(self.save_dir, self.SUMMARY_FILE_NAME)
        self.save_summary_csv(summary)
        return responses

    def build_episode_metrics(
        self,
        responses: Responses,
        traces: list[EpisodeTrace],
    ) -> list[EpisodeMetricsRecord]:
        records = []
        for request_ordinal, (response, trace) in enumerate(zip(responses.responses, traces)):
            if trace.request_ordinal != request_ordinal:
                raise ValueError(
                    "Episode trace request_ordinal does not align with response order."
                )
            if not response.prompt.should_abstain:
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

            records.append(
                EpisodeMetricsRecord(
                    request_ordinal=request_ordinal,
                    should_abstain=response.prompt.should_abstain,
                    oracle_abstain_step=oracle_abstain_step,
                    first_abstain_step=first_abstain_step,
                    search_calls_used=search_calls_used,
                    decision_turns=decision_turns,
                    success=success,
                    termination_reason=trace.termination_reason,
                    spl_contribution=spl_contribution,
                )
            )
        return records

    def summarize(
        self,
        episode_metrics: list[EpisodeMetricsRecord],
    ) -> WikimediaEpisodeMetricsSummary:
        num_abstain_samples = len(episode_metrics)
        pass_at_k = {
            f"pass_at_{budget}": self._safe_rate(
                sum(
                    1
                    for record in episode_metrics
                    if record.success
                    and record.first_abstain_step is not None
                    and (record.first_abstain_step - record.oracle_abstain_step) <= budget
                ),
                num_abstain_samples,
            )
            for budget in range(0, 11)
        }
        search_call_distribution = Counter(
            record.search_calls_used for record in episode_metrics
        )

        return WikimediaEpisodeMetricsSummary(
            num_abstain_samples=num_abstain_samples,
            early_recall=self._safe_rate(
                sum(
                    1
                    for record in episode_metrics
                    if record.first_abstain_step == record.oracle_abstain_step
                ),
                num_abstain_samples,
            ),
            overall_recall=self._safe_rate(
                sum(1 for record in episode_metrics if record.success),
                num_abstain_samples,
            ),
            spl=self._safe_rate(
                sum(record.spl_contribution for record in episode_metrics),
                num_abstain_samples,
            ),
            average_search_calls=self._safe_rate(
                sum(record.search_calls_used for record in episode_metrics),
                num_abstain_samples,
            ),
            average_decision_turns=self._safe_rate(
                sum(record.decision_turns for record in episode_metrics),
                num_abstain_samples,
            ),
            forced_stop_count=sum(
                1
                for record in episode_metrics
                if record.termination_reason == "max_search_calls_reached"
            ),
            forced_stop_rate=self._safe_rate(
                sum(
                    1
                    for record in episode_metrics
                    if record.termination_reason == "max_search_calls_reached"
                ),
                num_abstain_samples,
            ),
            pass_at_k=pass_at_k,
            search_call_distribution={
                str(search_calls): count
                for search_calls, count in sorted(search_call_distribution.items())
            },
        )

    def save_summary_csv(self, summary: WikimediaEpisodeMetricsSummary) -> None:
        permissive_makedirs(self.save_dir)
        csv_path = Path(self.save_dir) / f"{self.SUMMARY_FILE_NAME}.csv"
        flattened = summary.model_dump(mode="json")
        pass_at_k = flattened.pop("pass_at_k")
        search_call_distribution = flattened.pop("search_call_distribution")
        flattened.update(pass_at_k)
        for search_calls, count in search_call_distribution.items():
            flattened[f"search_call_count_{search_calls}"] = count

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(flattened))
            writer.writeheader()
            writer.writerow(flattened)

    def _load_episode_traces(self) -> list[EpisodeTrace]:
        trace_path = Path(self.save_dir) / self.TRACE_FILE_NAME
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Wikimedia multi-turn trace file not found at {trace_path}."
            )
        with trace_path.open("r") as trace_file:
            return [
                EpisodeTrace.model_validate_json(line)
                for line in trace_file
                if line.strip()
            ]

    def _save_episode_metrics(self, episode_metrics: list[EpisodeMetricsRecord]) -> None:
        json_path = Path(self.save_dir) / f"{self.EPISODE_METRICS_FILE_NAME}.json"
        json_path.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in episode_metrics],
                indent=2,
            )
        )

    def _safe_rate(
        self,
        numerator: float,
        denominator: int,
    ) -> Optional[float]:
        if denominator == 0:
            return None
        return numerator / denominator
