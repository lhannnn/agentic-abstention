"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Implements abstention methods
"""

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

from pydantic import BaseModel

from recipe.inference import LoadSaveDataMixin, RawResponse, RawResponses


class Response(RawResponse):
    response_or_abstention: str
    full_judge_response: Optional[str] = None
    full_correctness_judge_response: Optional[str] = None
    is_abstention: Optional[bool] = None
    is_abstention_correct: Optional[bool] = None
    is_response_correct: Optional[bool] = None

    def to_flat_dict(self) -> dict:
        response_dict = self.model_dump()
        flat_response_dict = self._flatten(response_dict)
        return flat_response_dict

    def _flatten(self, dictionary, parent_key="", separator="_"):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(self._flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)


class Responses(BaseModel, LoadSaveDataMixin):
    responses: list[Response]

    def __len__(self):
        return len(self.responses)


class AbstentionMethod(ABC):
    abstention_string = "I don't know"

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def respond_or_abstain(raw_response: RawResponse) -> str:
        pass

    def run(self, raw_responses: RawResponses) -> Responses:
        all_responses = []
        for raw_response in raw_responses.responses:
            response_or_abstention = self.respond_or_abstain(raw_response)
            all_responses.append(
                Response(
                    **raw_response.model_dump(),
                    response_or_abstention=response_or_abstention,
                )
            )

        responses = Responses(responses=all_responses)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses


class DirectAbstention(AbstentionMethod):
    def respond_or_abstain(self, raw_response: RawResponse) -> str:
        return raw_response.response
