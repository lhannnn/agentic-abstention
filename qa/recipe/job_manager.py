"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
python recipe/job_manager.py show_relaunch_commands_for_missing_jobs --sweep_dir=sweep-reasoning-2025-03-25
"""

import glob
import os
from collections import defaultdict
from itertools import product
from pathlib import Path

import fire
from omegaconf import OmegaConf

from recipe.agentic_dataset_policy import AGENTIC_BENCHMARK_DATASET_CONFIG_NAMES_SET


class JobManager:
    """
    Shows complete jobs, missing jobs and relaunch commands for missing jobs

    By default will run on all models (and datasets) unless list of models (or datasets) is specified

    Args:
        parent_results_dir: directory of the results
        sweep_dir: directory of the sweep
        final_result_file: name of the final result file to consider job complete
        mode: cluster_one_node_v100
        models: list of models based on config names to consider
        datasets: list of datasets based on config names to consider

    Methods:
        show_complete() -> dictionary of model -> complete jobs (datasets)
        show_missing() -> dictionary of model -> missing jobs (datasets)
        show_relaunch_command() -> suggested commands to relaunch missing jobs
    """

    def __init__(
        self,
        base_results_dir: str = "/large_experiments/robust_vlm/abstention-bench/results/",
        sweep_dir: str = "sweep-20240319",
        final_result_file: str = "GroundTruthAbstentionEvaluator.json",
        mode="cluster_one_node_v100",
        models: list[str] = [],
        datasets: list[str] = [],
    ):
        self.parent_results_dir = base_results_dir
        self.results_dir = os.path.join(base_results_dir, sweep_dir)
        self.sweep_dir = sweep_dir
        self.mode = mode
        self.final_result_file = final_result_file

        self.config_dir = Path(self.get_current_file_directory()).parent / "configs"

        self.dataset_to_dataset_class: dict[str, str] = (
            self.get_dataset_to_dataset_config()
        )
        self.datasets = (
            datasets if datasets else list(self.dataset_to_dataset_class.keys())
        )

        self.model_to_model_class: dict[str, str] = self.get_model_to_model_config()
        self.models: list[str] = (
            models if models else list(self.model_to_model_class.keys())
        )

    def get_dataset_to_dataset_config(
        self, exclude_dummy: bool = True
    ) -> dict[str, str]:
        """
        Creates a list of all dataset config names -> dataset class names
        """
        dataset_to_dataset_class = dict()
        allowed_dataset_config_names = set(AGENTIC_BENCHMARK_DATASET_CONFIG_NAMES_SET)
        if not exclude_dummy:
            allowed_dataset_config_names.add("dummy")

        for dataset_config_path in self.config_dir.glob("dataset/*.yaml"):
            dataset_config_name = dataset_config_path.stem
            if dataset_config_name not in allowed_dataset_config_names:
                continue
            dataset_config = OmegaConf.load(dataset_config_path)
            dataset_class = dataset_config.dataset_name
            dataset_to_dataset_class[dataset_config_name] = dataset_class
        if exclude_dummy:
            dataset_to_dataset_class.pop("dummy", None)
        return dataset_to_dataset_class

    def get_model_to_model_config(self, exclude_dummy: bool = True) -> dict[str, str]:
        """
        Creates a list of all model config names -> model class names
        """
        model_to_model_class = dict()

        for model_config_path in self.config_dir.glob("model/*.yaml"):
            model_config_name = model_config_path.stem
            model_config = OmegaConf.load(model_config_path)
            model_class = model_config.model_name
            model_to_model_class[model_config_name] = model_class
        if exclude_dummy:
            model_to_model_class.pop("dummy")
        return model_to_model_class

    def show_missing(self) -> dict[str, str]:
        """
        Returns:
            model: missing jobs

        Dictionary only contains entries with missing jobs
        """
        missing_jobs = defaultdict(list)
        for model, dataset in product(self.models, self.datasets):
            path = self.get_latest_result_path(model, dataset)
            if not path:
                missing_jobs[model].append(dataset)
        return missing_jobs

    def show_relaunch_commands(self) -> list[str]:
        """
        Create one sweep per model
        Given inference stage is quite fast, we relaunch the full job if any stage is missing.

        Returns: suggested command for relaunching missing jobs

        Example:
        python main.py -m mode=cluster_one_node_v100
        dataset=gpqa,mmlu_math,mmlu_history,gsm8k
        model=gemini_15_pro
        sweep_folder=sweep-reasoning-2025-03-25
        """
        missing_jobs = self.show_missing()
        commands = []
        for model, datasets in missing_jobs.items():
            dataset_config_names = ",".join(datasets)
            command = f"python main.py -m mode={self.mode} model={model} dataset={dataset_config_names} sweep_folder={self.sweep_dir}"
            commands.append(command)
        return commands

    def show_complete(self, show_relative_dir: bool = False) -> list[str]:
        """
        Returns a list of directories that have all results
        if a model has multiple subdirectories with all results, only the latest one is considered.
        """
        complete_paths = []

        for model, dataset in product(self.models, self.datasets):
            path = self.get_latest_result_path(model, dataset)
            if path:
                complete_paths.append(path)
        if show_relative_dir:
            base_results_dir = os.path.dirname(self.results_dir)
            relative_paths = [
                os.path.relpath(os.path.dirname(path), base_results_dir)
                for path in complete_paths
            ]
            return relative_paths
        return complete_paths

    def get_latest_result_path(self, model: str, dataset: str) -> str | None:
        """
        model and dataset are config names

        Will return latest path that has full results
        """
        dataset_class = self.dataset_to_dataset_class[dataset]
        model_class = self.model_to_model_class[model]
        paths = glob.glob(
            f"{self.results_dir}/{dataset_class}_{model_class}/*/{self.final_result_file}",
            recursive=True,
        )
        if not paths:
            return None
        sorted_paths = sorted(paths)
        # return latest given we use datetime labeling for our results
        return sorted_paths[-1]

    def get_current_file_directory(self) -> str:
        """
        Returns the directory of the current file.
        """
        return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    fire.Fire(JobManager)

    # job_manager = JobManager()
    # print(job_manager.results_dir)
    # print(job_manager.show_complete())

    # print()
    # print("missing jobs")
    # print(job_manager.show_missing())
    # print(job_manager.show_missing())
