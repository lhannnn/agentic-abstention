"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
from pathlib import Path

import datasets
from recipe.abstention_datasets.abstract_abstention_dataset import AbstentionDataset
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt


class SituatedQAGeoDataset(AbstentionDataset):

    def __init__(self, data_path="data/situated_qa/geo.jsonl", max_num_samples=None):
        super().__init__()

        self.data_path = Path(data_path)
        self.dataset = self._load_local_dataset(self.data_path)

        # Construct the underspecified dataset (which needs to be deduplicated)
        visited_questions = set()
        deduplicated_rows = []
        for row in self.dataset:
            if row["question"] not in visited_questions:
                deduplicated_rows.append(row)
                visited_questions.add(row["question"])

        self.deduplicated_dataset = datasets.Dataset.from_list(list(deduplicated_rows))

        self.max_num_samples = max_num_samples

    def _load_local_dataset(self, data_path: Path) -> datasets.Dataset:
        if not data_path.exists():
            raise FileNotFoundError(
                "SituatedQA local cache not found. Expected a local JSONL, JSON, "
                f"Parquet, or datasets.load_from_disk directory at {data_path}."
            )

        if data_path.is_dir():
            return datasets.Dataset.load_from_disk(str(data_path))

        if data_path.suffix == ".parquet":
            return datasets.Dataset.from_parquet(str(data_path))

        if data_path.suffix in {".jsonl", ".json"}:
            rows = []
            with data_path.open("r") as source_file:
                if data_path.suffix == ".json":
                    parsed_json = json.load(source_file)
                    if isinstance(parsed_json, list):
                        rows = parsed_json
                    else:
                        raise ValueError(
                            "SituatedQA JSON cache must contain a list of objects."
                        )
                else:
                    for line in source_file:
                        if line.strip():
                            rows.append(json.loads(line))
            return datasets.Dataset.from_list(rows)

        raise ValueError(
            "Unsupported SituatedQA cache format. Use JSONL, JSON, Parquet, or a datasets directory."
        )

    def __len__(self):
        return self.max_num_samples or (
            len(self.dataset) + len(self.deduplicated_dataset)
        )

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        # Concatenate the deduplicated dataset (for underspecified questions) with
        # the original dataset (for fully specified questions)
        if idx < len(self.dataset):
            item = self.dataset[idx]
            question = item["edited_question"] + "?"
            reference_answers = item["any_answer"]
            should_abstain = False
        else:
            offset_idx = idx - len(self.dataset)
            item = self.deduplicated_dataset[offset_idx]
            question = item["question"] + "?"
            reference_answers = None
            should_abstain = True

        metadata = {
            "SituatedQA_id": item["id"],
            "SituatedQA_location": item["location"],
        }

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )
