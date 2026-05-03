"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.multiturn_alcuna_prompts import build_alcuna_multiturn_agent_prompt
from recipe.prompt_wrappers import PromptWrapper


class ALCUNAMultiTurnSearchPromptWrapper(PromptWrapper):
    def wrap_prompt(self, prompt: Prompt) -> str:
        return build_alcuna_multiturn_agent_prompt(prompt.question)
