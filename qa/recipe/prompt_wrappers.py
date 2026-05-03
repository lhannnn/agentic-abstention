"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC, abstractmethod

from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.action_space import (
    build_action_space_prompt,
    build_action_space_search_prompt,
)


class PromptWrapper(ABC):
    @abstractmethod
    def wrap_prompt(self, prompt: Prompt) -> str:
        pass


class IdentityPromptWrapper(PromptWrapper):
    def wrap_prompt(self, prompt: Prompt) -> str:
        return prompt.question


class ActionSpacePromptWrapper(PromptWrapper):
    def wrap_prompt(self, prompt: Prompt) -> str:
        return build_action_space_prompt(prompt.question)


class ActionSpaceSearchPromptWrapper(PromptWrapper):
    def wrap_prompt(self, prompt: Prompt) -> str:
        return build_action_space_search_prompt(prompt.question)
