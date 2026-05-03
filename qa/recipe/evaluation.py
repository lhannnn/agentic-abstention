"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
evaluates whether a response is an abstention 
and whether it's correct to abstain
"""

import csv
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from recipe.abstention import Response, Responses
from recipe.action_space import (
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_TOOL_CALL,
    gold_action_from_should_abstain,
    parse_action_space_response,
)
from recipe.abstention_keywords import ABSTENTION_KEYWORDS
from recipe.evaluation_judge_prompts import (
    LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT,
    LLM_JUDGE_CORRECTNESS_PROMPT,
    LLM_JUDGE_MATH_CORRECTNESS_PROMPT,
)
from recipe.inference import LoadSaveDataMixin
from recipe.models import InferenceModel
from recipe.utils import permissive_makedirs
from recipe.wikimedia_json_actions import parse_wikimedia_json_response
from pydantic import BaseModel


class AbstentionEvaluator(ABC):
    """Determines whether the model should abstain given the question.
    Really only meaningful for correctness;
    otherwise, ground-truth comes from the dataset
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def is_abstention_correct(self, response: Response) -> bool:
        pass

    def run(self, responses: Responses) -> Responses:
        responses_list = []
        for response in responses.responses:
            response.is_abstention_correct = self.is_abstention_correct(response)
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses


class GroundTruthAbstentionEvaluator(AbstentionEvaluator):
    def is_abstention_correct(self, response: Response) -> bool:
        return response.is_abstention == response.prompt.should_abstain


class ActionSpaceMetricsSummary(BaseModel, LoadSaveDataMixin):
    num_samples: int
    answerable_count: int
    abstain_warranted_count: int
    invalid_count: int
    invalid_rate: float
    binary_accuracy: float
    binary_precision: float
    binary_recall: float
    binary_f1: float
    exact_action_accuracy: float
    unnecessary_tool_call_rate_answerable: Optional[float]
    unnecessary_tool_call_rate_abstain_warranted: Optional[float]
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    predicted_answer_count: int
    predicted_abstain_count: int
    predicted_tool_call_count: int
    predicted_invalid_count: int


class ActionSpaceMetricsEvaluator(AbstentionEvaluator):
    SUMMARY_FILE_NAME = "ActionSpaceMetricsSummary"

    def is_abstention_correct(self, response: Response) -> bool:
        if response.predicted_action == ACTION_INVALID:
            return False
        return response.is_abstention == response.prompt.should_abstain

    def run(self, responses: Responses) -> Responses:
        responses = super().run(responses)
        summary = self.summarize(responses)
        summary.save(self.save_dir, self.SUMMARY_FILE_NAME)
        self.save_summary_csv(summary)
        return responses

    def summarize(self, responses: Responses) -> ActionSpaceMetricsSummary:
        tp = tn = fp = fn = 0
        exact_action_matches = 0
        answerable_count = 0
        abstain_warranted_count = 0
        tool_calls_on_answerable = 0
        tool_calls_on_abstain_warranted = 0
        action_counter = Counter()

        for response in responses.responses:
            predicted_action = response.predicted_action or ACTION_INVALID
            gold_action = gold_action_from_should_abstain(response.prompt.should_abstain)
            action_counter[predicted_action] += 1

            if predicted_action == gold_action:
                exact_action_matches += 1

            if gold_action == ACTION_ANSWER:
                answerable_count += 1
                if predicted_action == ACTION_TOOL_CALL:
                    tool_calls_on_answerable += 1
                if predicted_action in {ACTION_ANSWER, ACTION_TOOL_CALL}:
                    tn += 1
                else:
                    fp += 1
            elif gold_action == ACTION_ABSTAIN:
                abstain_warranted_count += 1
                if predicted_action == ACTION_TOOL_CALL:
                    tool_calls_on_abstain_warranted += 1
                if predicted_action == ACTION_ABSTAIN:
                    tp += 1
                else:
                    fn += 1

        num_samples = len(responses.responses)
        invalid_count = action_counter[ACTION_INVALID]
        precision = self._safe_rate(tp, tp + fp, default=0.0)
        recall = self._safe_rate(tp, tp + fn, default=0.0)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return ActionSpaceMetricsSummary(
            num_samples=num_samples,
            answerable_count=answerable_count,
            abstain_warranted_count=abstain_warranted_count,
            invalid_count=invalid_count,
            invalid_rate=self._safe_rate(invalid_count, num_samples, default=0.0),
            binary_accuracy=self._safe_rate(tp + tn, num_samples, default=0.0),
            binary_precision=precision,
            binary_recall=recall,
            binary_f1=f1,
            exact_action_accuracy=self._safe_rate(
                exact_action_matches, num_samples, default=0.0
            ),
            unnecessary_tool_call_rate_answerable=self._safe_rate(
                tool_calls_on_answerable, answerable_count
            ),
            unnecessary_tool_call_rate_abstain_warranted=self._safe_rate(
                tool_calls_on_abstain_warranted, abstain_warranted_count
            ),
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            predicted_answer_count=action_counter[ACTION_ANSWER],
            predicted_abstain_count=action_counter[ACTION_ABSTAIN],
            predicted_tool_call_count=action_counter[ACTION_TOOL_CALL],
            predicted_invalid_count=action_counter[ACTION_INVALID],
        )

    def save_summary_csv(self, summary: ActionSpaceMetricsSummary) -> None:
        permissive_makedirs(self.save_dir)
        csv_path = Path(self.save_dir) / f"{self.SUMMARY_FILE_NAME}.csv"
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=list(summary.model_dump(mode="json"))
            )
            writer.writeheader()
            writer.writerow(summary.model_dump(mode="json"))

    def _safe_rate(
        self, numerator: int, denominator: int, default: Optional[float] = None
    ) -> Optional[float]:
        if denominator == 0:
            return default
        return numerator / denominator


# This class is needed due to saving logic of AbstentionEvaluator being
# dependent on the class name.
class GroundTruthAbstentionEvaluatorWithReasoning(GroundTruthAbstentionEvaluator):
    pass


class AbstentionDetector(ABC):
    """Determines whether a response constitutes an abstention"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def detect_abstention(
        self, response: Response
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Detects abstention in model's response, returns is_abstention label
        which is bool or None (in case detector wasn't able to determine abstention label)
        and optionally a longer AbstentionDetector's response (relevant for LLM judge).
        """
        pass

    def run(self, responses: Responses) -> Responses:
        responses_list = []
        for response in responses.responses:
            response.is_abstention, response.full_judge_response = (
                self.detect_abstention(response)
            )
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses


class ContainsAbstentionKeywordAbstentionDetector(AbstentionDetector):

    def detect_abstention(self, response: Response) -> Tuple[bool, Optional[str]]:
        """search in response for keywords"""
        response_lower = response.response_or_abstention.lower()
        for keyword in ABSTENTION_KEYWORDS:
            if keyword.lower() in response_lower:
                # Second returned value corresponds to
                # full judge response, returning None
                # since this is not an LLM judge.
                return True, None

        return False, None


class ActionSpaceTagAbstentionDetector(AbstentionDetector):
    def detect_abstention(self, response: Response) -> Tuple[bool, Optional[str]]:
        parse_result = parse_action_space_response(response.response_or_abstention)
        response.predicted_action = parse_result.predicted_action
        response.action_parse_valid = parse_result.is_valid
        response.action_parse_error = parse_result.error
        return response.predicted_action == ACTION_ABSTAIN, None


class WikimediaJsonAbstentionDetector(AbstentionDetector):
    def detect_abstention(self, response: Response) -> Tuple[bool, Optional[str]]:
        parse_result = parse_wikimedia_json_response(response.response_or_abstention)
        response.predicted_action = parse_result.predicted_action
        response.action_parse_valid = parse_result.is_valid
        response.action_parse_error = parse_result.error
        return response.predicted_action == ACTION_ABSTAIN, None


class LLMJudgeAbstentionDetector(AbstentionDetector):
    USE_RESPONSE_WITH_REASONING = False

    def __init__(
        self, judge_model: InferenceModel, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._judge_model = judge_model

    def detect_abstention(self, response: Response) -> Tuple[Optional[bool], str]:
        # Reference answers are list or None
        if response.prompt.reference_answers:
            reference_answers = (
                "[" + "; ".join(response.prompt.reference_answers) + "]"
            )
        else:
            reference_answers = "[]"

        model_answer = (
            response.response_with_reasoning
            if self.USE_RESPONSE_WITH_REASONING
            else response.response_or_abstention
        )
        judge_prompt = LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT.format(
            question=response.prompt.question,
            ref_answer=reference_answers,
            abstention_label=response.prompt.should_abstain,
            model_answer=model_answer,
        )

        # Temporary workaround for batched inference
        judge_response = self._judge_model.respond([judge_prompt])[0]
        judge_response = judge_response.lower().strip(" .,\n")
        if judge_response not in ["yes", "no"]:
            logger.warning(
                f"\nUnexpected judge response:\n{judge_response}"
                f"\nJudge prompt:\n{judge_prompt}"
            )
            return None, judge_response
        return judge_response.lower() == "yes", judge_response


class LLMJudgeAbstentionDetectorWithReasoning(LLMJudgeAbstentionDetector):
    USE_RESPONSE_WITH_REASONING = True


class CorrectnessEvaluator(ABC):
    """Determines whether a response is correct, according to the dataset's reference answers"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def is_response_correct(
        self, response: Response
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Determine's whether the response is correct, returning is_response correct label
        which is bool or None (in case no reference answers are given), and optionally a
        longer AbstentionDetector's response (relevant for LLM judge).
        """
        pass

    def run(self, responses: Responses) -> Responses:

        responses_list = []
        for response in responses.responses:
            response.is_response_correct, response.full_correctness_judge_response = (
                self.is_response_correct(response)
            )
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)


class LLMJudgeCorrectnessEvaluator(CorrectnessEvaluator):

    def __init__(self, judge_model: InferenceModel, math_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._judge_model = judge_model
        self.math_mode = math_mode

    def is_response_correct(self, response: Response) -> Tuple[Optional[bool], str]:
        if response.prompt.reference_answers is None:
            return (None, "")

        if not self.math_mode:
            reference_answers = "\n".join(response.prompt.reference_answers)

            judge_prompt = LLM_JUDGE_CORRECTNESS_PROMPT.format(
                question=response.prompt.question,
                ref_answer=reference_answers,
                model_answer=response.response_or_abstention,
            )
        else:
            reference_answers = "; ".join(response.prompt.reference_answers)

            judge_prompt = LLM_JUDGE_MATH_CORRECTNESS_PROMPT.format(
                question=response.prompt.question,
                ref_answer=reference_answers.strip(" \n"),
                model_answer=response.response_or_abstention.strip(" \n"),
            )

        # Temporary workaround for batched inference
        judge_response = self._judge_model.respond([judge_prompt])[0]
        judge_response = judge_response.lower().strip(" .,\n")
        if judge_response not in ["correct", "incorrect"]:
            logger.warning(
                f"\nUnexpected judge response:\n{judge_response}"
                f"\nJudge prompt:\n{judge_prompt}"
            )
            return None, judge_response
        return judge_response.lower() == "correct", judge_response
