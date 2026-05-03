"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Generates and saves model responses for  questions
"""

import json
import logging
import os
from typing import Optional

from pydantic import BaseModel
from torch.utils.data import DataLoader, SubsetRandomSampler

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)
from recipe.models import InferenceModel, VLLMReasoningChatModelBase
from recipe.prompt_wrappers import IdentityPromptWrapper, PromptWrapper
from recipe.utils import permissive_makedirs

logger = logging.getLogger(__name__)


class LoadSaveDataMixin:
    """Load and save Pydantic classes for storing/loading results"""

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
        validated_data = cls.model_validate(data)
        return validated_data

    def save(self, save_dir: str, file_name: Optional[str] = None):
        permissive_makedirs(save_dir)
        data = self.model_dump(mode="json")
        file_name = file_name if file_name else self.__class__.__name__
        file_path = os.path.join(save_dir, f"{file_name}.json")
        logger.info(f"saving {file_path}")
        with open(file_path, "w") as file:
            json.dump(data, file)


class RawResponse(BaseModel):
    prompt: Prompt
    inference_prompt: Optional[str] = None
    response: str
    # for deepseek and other reasoning models
    reasoning: None | str = None
    response_with_reasoning: None | str = None
    predicted_action: Optional[str] = None
    action_parse_valid: bool = False
    action_parse_error: Optional[str] = None


class RawResponses(BaseModel, LoadSaveDataMixin):
    responses: list[RawResponse]

    def __len__(self):
        return len(self.responses)


def _collate_class_instances(batch):
    return batch

class InferencePipeline:
    """Uses torch datasets for batched inference"""

    def __init__(
        self,
        model: InferenceModel,
        dataset: list[AbstentionDataset],
        save_dir: str,
        batch_size: int = 1,
        prompt_wrapper: PromptWrapper | None = None,
    ):
        self.model = model
        self.dataset = dataset
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.prompt_wrapper = prompt_wrapper or IdentityPromptWrapper()

    def run(self, indices_subset=None) -> RawResponses:
        raw_responses_list = []

        # If a set of indices is passed, only iterate over those, instead of entire dataset
        if indices_subset is not None:
            sampler = SubsetRandomSampler(list(indices_subset))
        else:
            sampler = None

        batched_dataset = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=_collate_class_instances,
        )
        for prompt_batch in batched_dataset:
            questions = [
                self.prompt_wrapper.wrap_prompt(prompt) for prompt in prompt_batch
            ]

            if isinstance(self.model, VLLMReasoningChatModelBase):
                response_texts, reasoning_texts, response_with_reasoning_texts = self.model.respond(questions)
            else:
                response_texts = self.model.respond(questions)
                assert isinstance(
                    response_texts[0], str
                ), f"{response_texts=} should be a list of strings."
                reasoning_texts = None
                response_with_reasoning_texts = None

            for i, response_text in enumerate(response_texts):
                reasoning_text = reasoning_texts[i] if reasoning_texts else None
                response_with_reasoning_text = response_with_reasoning_texts[i] if response_with_reasoning_texts else None
                raw_response = RawResponse(
                    prompt=prompt_batch[i],
                    inference_prompt=questions[i],
                    response=response_text,
                    reasoning=reasoning_text,
                    response_with_reasoning=response_with_reasoning_text,
                )
                raw_responses_list.append(raw_response)

        raw_responses = RawResponses(responses=raw_responses_list)
        raw_responses.save(self.save_dir, self.__class__.__name__)

        return raw_responses
