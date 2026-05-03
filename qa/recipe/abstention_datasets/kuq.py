"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional

import datasets
import pandas as pd
from hydra.utils import to_absolute_path

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)
from recipe.agentic_dataset_policy import KUQ_ACTIVE_BENCHMARK_CATEGORIES


class KUQDataset(AbstentionDataset):

    _AVAILABLE_CATEGORIES = frozenset(
        [
            "ambiguous",
            "controversial",
            "false assumption",
            "counterfactual",
            "future unknown",
            "unsolved problem",
        ]
    )

    def __init__(
        self,
        categories: List[str] = None,
        max_num_samples=None,
        category_map_path: Optional[str] = None,
    ):
        super().__init__()

        self.dataset = datasets.load_dataset(
            "amayuelas/KUQ", data_files="knowns_unknowns.jsonl"
        )["train"]

        if category_map_path is not None:
            # Load the category data, and only keep rows where a category is assigned
            category_df = pd.read_csv(to_absolute_path(category_map_path)).dropna(
                subset="category"
            )
            category_map = dict(category_df[["id", "category"]].values)

            # Use the map to assign a category to each sample that doesn't already have one
            self.dataset = self.dataset.map(
                lambda sample, _id: {
                    "category": sample["category"] or category_map.get(_id, None)
                },
                with_indices=True,
            )

        self.categories = (
            set(categories)
            if categories is not None
            else set(KUQ_ACTIVE_BENCHMARK_CATEGORIES)
        )

        invalid_categories = self.categories - self._AVAILABLE_CATEGORIES
        if invalid_categories:
            raise ValueError(f"Invalid categories: {invalid_categories}")

        if self.categories:
            self.dataset = self.dataset.filter(
                lambda item: item["category"] in self.categories
            )

        self.max_num_samples = max_num_samples

    def __len__(self):
        return self.max_num_samples or len(self.dataset)

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError

        item = self.dataset[idx]

        question = item["question"]
        should_abstain = item["unknown"]
        reference_answers = item["answer"] if not should_abstain else None
        metadata = {
            "KUQ_source": item["source"],
            "KUQ_category": item["category"],
        }

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )


if __name__ == "__main__":
    kuq_dataset = KUQDataset()
    print(kuq_dataset[3])
