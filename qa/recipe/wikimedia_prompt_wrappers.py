"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.prompt_wrappers import PromptWrapper
from recipe.wikimedia_prompts import build_wikimedia_multiturn_agent_prompt


class WikimediaMultiTurnSearchPromptWrapper(PromptWrapper):
    def __init__(self, top_k: int = 3, max_search_calls: int = 10):
        self.top_k = top_k
        self.max_search_calls = max_search_calls

    def wrap_prompt(self, prompt: Prompt) -> str:
        return build_wikimedia_multiturn_agent_prompt(
            prompt.question,
            top_k=self.top_k,
            max_search_calls=self.max_search_calls,
        )
