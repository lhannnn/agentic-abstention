"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
import json
import logging
import os
import time
from pathlib import Path
from pprint import pformat
from typing import List, Optional

import git
import hydra
import submitit
import torch.distributed as dist
import transformers
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from submitit.helpers import clean_env

from recipe.abstention import Responses
from recipe.inference import InferencePipeline, RawResponses
from recipe.models import InferenceModel
from recipe.prompt_wrappers import IdentityPromptWrapper

logger = logging.getLogger(__name__)


MODELS_WITHOUT_GPU = [
    "DummyModel",
    "GPT4oAPI",
    "o1API",
    "Gemini15ProAPI",
    "TogetherAIAPI",
    "TogetherAIAPI_ACEFormat",
]

LLM_JUDGES_WITHOUT_GPU = [
    "ContainsAbstentionKeyword",
    "LLMJudgeGPT4o",
    "LLMJudgeTogetherAI",
]


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AbstentionWorkflow:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)

    def raise_warning_two_models_within_one_job(self):
        """VLLM fails to initialize a multi-gpu and single gpu model within a single job.
        https://x.com/wzhao_nlp/status/1868741915317039605
        """
        # Not relevant if we are only running inference, or running inference and eval in two jobs.
        if (
            self.config.only_run_inference
            or not self.config.run_single_job_for_inference_and_judge
        ):
            return
        if "judge_model" not in self.config.abstention_detector:
            return
        # We only need to process cases when both inference model and LLM judge require GPUs.
        if (
            self.config.judge_name in LLM_JUDGES_WITHOUT_GPU
            or self.config.model_name in MODELS_WITHOUT_GPU
        ):
            return

        inference_model_tensor_parallel = self.config.module.tensor_parallel_size
        judge_model_tensor_parallel = (
            self.config.abstention_detector.judge_model.tensor_parallel_size
        )

        if inference_model_tensor_parallel != judge_model_tensor_parallel:
            raise ValueError(
                "vLLM can't initialize two models with different tensor_parallel_size within one job."
            )

    def run(self):
        """Runs the full abstention workflow."""
        start_time_inference = time.time()
        self.raise_warning_two_models_within_one_job()

        # Stage 1: inference and abstention (currently direct abstention)
        if self.config.load_inference_logs:
            responses = self.load_inference_abstention_logs()
        else:
            raw_responses = self.run_inference()
            responses = self.run_abstention_method(raw_responses)

        # Since later stages modify the object itself, we want to keep the copy
        # for an independent judge.
        responses_copy = copy.deepcopy(responses)

        end_time_inference = time.time()
        elapsed_time_inference = end_time_inference - start_time_inference
        logger.info(f"Elapsed time inference: {format_time(elapsed_time_inference)}")

        if self.config.only_run_inference:
            self.final_cleanup()
            return

        # Stage 2: abstention detection and evaluation.
        if (
            self.config.load_inference_logs
            or self.config.run_single_job_for_inference_and_judge
        ):
            start_time_eval = time.time()
            logger.info("Running stage 2: abstention detection and evaluation.")

            # Temporarily set up a shared judge model to use for both abstention detection
            # and correctness evaluation (if using).
            shared_judge_model = self._set_up_shared_judge_model()

            responses = self.run_abstention_detection(
                responses, judge_model=shared_judge_model
            )

            responses = self.run_evaluation(responses)

            if self._should_run_judge_with_reasoning():
                responses_copy = self.run_abstention_detection_with_reasoning(
                    responses_copy, judge_model=shared_judge_model
                )
                responses_copy = self.run_evaluation_with_reasoning(responses_copy)

            if self._should_run_correctness_evaluator():
                logger.info("Running correctness evaluator")
                responses = self.run_correctness_evaluation(
                    responses, shared_judge_model
                )

            end_time_eval = time.time()
            elapsed_time_eval = end_time_eval - start_time_eval
            logger.info(f"Elapsed time eval: {format_time(elapsed_time_eval)}")
        # Launch abstention detection and evaluation in a separate process.
        else:
            logger.info("Launching a separate job for judge evaluations with submitit.")
            self.launch_evaluation_in_separate_job_with_submitit()

        logger.info("calling cleanup")
        self.final_cleanup()

    def launch_evaluation_in_separate_job_with_submitit(self):
        second_stage_config, second_stage_logs_dir = self.create_second_stage_config()
        executor = self.make_submitit_executor(
            second_stage_logs_dir, second_stage_config.abstention_detector_launcher
        )
        with clean_env():
            job = executor.submit(main, second_stage_config)
            logger.info(f"Launched eval job! job_id: {job.job_id}")
            # Wait for job completion
            output = job.result()
        logger.info("Eval job (stage 2) complete.")

    def create_second_stage_config(self):
        inference_logs_path = os.path.join(
            self.config.save_dir, "DirectAbstention.json"
        )
        logs_dir = os.path.join(self.config.logs_dir, "eval")
        overrides = OmegaConf.create(
            {
                "inference_logs_path": inference_logs_path,
                "load_inference_logs": True,
                "logs_dir": logs_dir,
            }
        )
        second_stage_config = OmegaConf.merge(self.config, overrides)
        # Resolve modifies config in place by replacing variables with their values
        OmegaConf.resolve(second_stage_config)
        logger.info(
            f"Second stage config (resolved): \n {OmegaConf.to_yaml(second_stage_config)}"
        )
        return second_stage_config, logs_dir

    def make_submitit_executor(self, logs_dir, slurm_launcher_configs: DictConfig):
        try:
            hydra_conf = HydraConfig.get()
        except ValueError:
            logger.info("Hydra launcher not set; launching evaluation locally")
            executor = submitit.LocalExecutor(folder=logs_dir)
            return executor

        logger.info(f"Existing hydra config: \n{pformat(hydra_conf)}")

        executor = submitit.AutoExecutor(folder=logs_dir)
        executor.update_parameters(**slurm_launcher_configs)
        return executor

    def load_inference_abstention_logs(self) -> Responses:
        """Loads logs saved after model inference and abstention method
        (e.g. from DirectAbstention.json) into responses object."""
        if self.config.inference_logs_path is None:
            raise ValueError(
                "inference_logs_path must be set if using load_inference_logs."
            )
        logger.info(f"Loading model responses from {self.config.inference_logs_path}")
        responses = Responses.load(self.config.inference_logs_path)
        return responses

    def run_inference(self) -> RawResponses:
        """Runs inference with an LLM specified in config."""
        logger.info(f"running inference on machine: {os.uname()}")
        model = instantiate(self.config.module)
        datamodule = instantiate(self.config.datamodule)
        prompt_wrapper = (
            instantiate(self.config.prompt_wrapper)
            if "prompt_wrapper" in self.config
            else IdentityPromptWrapper()
        )
        inference_pipeline = InferencePipeline(
            model,
            datamodule,
            self.config.save_dir,
            batch_size=self.config.inference_batch_size,
            prompt_wrapper=prompt_wrapper,
        )

        # Optionally load a set of sample indices to limit inference to
        indices_subset = self._get_indices_subset(datamodule.name)

        raw_responses = inference_pipeline.run(indices_subset=indices_subset)
        logger.info("inference finished!")
        return raw_responses

    def run_abstention_method(self, raw_responses: RawResponses) -> Responses:
        """Abstention method applied to raw responses.
        As of Jan 2025, this simply copies model's responses."""
        abstention_method = instantiate(self.config.abstention_method)
        responses = abstention_method.run(raw_responses)
        logger.info("abstention finished!")
        return responses

    def run_abstention_detection(
        self,
        responses: Responses,
        judge_model: InferenceModel = None,
    ) -> Responses:
        """Detecting abstention in inference model's responses.
        The default detection method is LLM judge."""
        logger.info(f"launching abstention detector on : {os.uname()}")
        if judge_model is None:
            abstention_detector = instantiate(self.config.abstention_detector)
        else:
            # Disable recursive instantiation to prevent the hydra from creating a nested judge
            # as part of the abstention detector, because we're explicitly passing one in.
            abstention_detector = instantiate(
                self.config.abstention_detector,
                _recursive_=False,
                judge_model=judge_model,
            )
        responses = abstention_detector.run(responses)
        return responses

    def run_abstention_detection_with_reasoning(
        self,
        responses: Responses,
        judge_model: InferenceModel = None,
    ) -> Responses:
        """Detecting abstention in inference model's responses with reasoning.
        The default detection method is LLM judge."""
        logger.info(f"launching abstention detector with reasoning on : {os.uname()}")
        # Disable recursive instantiation to prevent the hydra from creating a nested judge
        # as part of the abstention detector, because we're explicitly passing one in.
        abstention_detector_with_reasoning = instantiate(
            self.config.abstention_detector_with_reasoning,
            _recursive_=False,
            judge_model=judge_model,
        )
        responses = abstention_detector_with_reasoning.run(responses)
        return responses

    def run_correctness_evaluation(
        self,
        responses: Responses,
        judge_model: InferenceModel,
    ) -> Responses:
        """Detecting whether inference model's responses are correct.
        The default evaluation method is LLM judge."""
        # Disable recursive instantiation to prevent the hydra from creating a nested judge
        # as part of the correctness evaluator, because we're explicitly passing one in.
        correctness_evaluator = instantiate(
            self.config.correctness_evaluator,
            _recursive_=False,
            judge_model=judge_model,
        )
        responses = correctness_evaluator.run(responses)
        return responses

    def run_evaluation(self, responses: Responses) -> Responses:
        """Evaluates whether abstentions (as determined by the judge)
        were correct based on the gold labels from the dataset."""
        abstention_evaluator = instantiate(self.config.abstention_evaluator)
        responses = abstention_evaluator.run(responses)
        return responses

    def run_evaluation_with_reasoning(self, responses: Responses) -> Responses:
        """Evaluates whether abstentions (as determined by the judge)
        were correct based on the gold labels from the dataset."""
        abstention_evaluator_with_reasoning = instantiate(
            self.config.abstention_evaluator_with_reasoning
        )
        responses = abstention_evaluator_with_reasoning.run(responses)
        return responses

    def final_cleanup(self):
        if dist.is_available() and dist.is_initialized():
            logger.info("destroying process group")
            dist.destroy_process_group()

        logger.info("all finished!")

    def _should_run_correctness_evaluator(self):
        return (
            "correctness_evaluator" in self.config
            and self.config.correctness_evaluator is not None
        )

    def _should_run_judge_with_reasoning(self):
        if (
            "abstention_detector_with_reasoning" not in self.config
            or self.config.abstention_detector_with_reasoning is None
        ):
            return False

        if self.config.model_name not in ["S1_1_32B", "DeepSeek_R1_Distill_Llama_70B"]:
            raise ValueError("Cannot run a 'reasoning' judge on a non-reasoning model.")
        return (
            "abstention_detector_with_reasoning" in self.config
            and self.config.abstention_detector_with_reasoning is not None
        )

    def _set_up_shared_judge_model(self):
        if self._should_run_correctness_evaluator():
            if "judge_model" not in self.config.abstention_detector:
                raise ValueError(
                    "Cannot run correctness evaluation without LLM-as-judge abstention detector"
                )

            if (
                self.config.abstention_detector.judge_model
                != self.config.correctness_evaluator.judge_model
            ):
                raise ValueError(
                    "Cannot use a different judge model for correctness and abstention"
                )

        if self._should_run_judge_with_reasoning():
            if "judge_model" not in self.config.abstention_detector:
                raise ValueError(
                    "Cannot run LLM judge with reasoning trace without LLM-as-judge abstention detector"
                )

            if (
                self.config.abstention_detector.judge_model
                != self.config.abstention_detector_with_reasoning.judge_model
            ):
                raise ValueError(
                    "Cannot use a different judge model for abstention with and without reasoning."
                )

        if "judge_model" in self.config.abstention_detector:
            shared_judge_model = instantiate(
                self.config.abstention_detector.judge_model
            )
        else:
            shared_judge_model = None

        return shared_judge_model

    def _get_indices_subset(self, dataset_name: str) -> Optional[List[int]]:
        """Load a subset of sample indices to run inference over.

        Can bet with either dataset_indices_subset (to manually specify a list, e.g. for testing)
        or with dataset_indices_path (to specify a JSON file containing a map of dataset names to
        indices lists), but not both.

        Returns a list of indices, or None if no subsampling required.
        """
        indices = None

        if (
            self.config.dataset_indices_subset is not None
            and self.config.dataset_indices_path is not None
        ):
            raise ValueError(
                "Can't set both dataset_indices_subset and dataset_indices_path"
            )

        if self.config.dataset_indices_path is not None:
            with open(to_absolute_path(self.config.dataset_indices_path), "r") as f:
                dataset_name_to_indices = json.load(f)
                if dataset_name in dataset_name_to_indices:
                    logger.info(
                        f"Found matching subset indices for {dataset_name} in {self.config.dataset_indices_path}"
                    )
                    indices = dataset_name_to_indices[dataset_name]

        elif self.config.dataset_indices_subset is not None:
            indices = self.config.dataset_indices_subset

        return indices


@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="default_pipeline.yaml",
)
def main(config: DictConfig) -> None:
    save_config(config)
    abstention_workflow = AbstentionWorkflow(config)
    abstention_workflow.run()


def get_git_hash() -> Optional[str]:
    try:
        repo = git.Repo(Path(__file__).parent.resolve())
        sha = repo.head.object.hexsha
        return sha
    except Exception as e:
        logger.warning("not able to find git hash")
        logger.error(e)


def create_log_and_save_dirs(config, hydra_dir: str | None = None):
    """
    Create log, save directories, hydra_dir if provided and set permissions
    777 for directories and 2 levels up of parent directories
    """
    logger.info(f"saving logs to {config.logs_dir}")
    logger.info(f"saving results to {config.save_dir}")

    # Attempt to change the permissions of the Hydra-created log dirs
    Path(config.logs_dir).mkdir(exist_ok=True, parents=True)
    Path(config.save_dir).mkdir(exist_ok=True, parents=True)
    if hydra_dir is not None:
        try:
            dir_paths = [Path(config.logs_dir), Path(config.save_dir), Path(hydra_dir)]
        except TypeError:
            dir_paths = [Path(config.logs_dir), Path(config.save_dir)]
    else:
        dir_paths = [Path(config.logs_dir), Path(config.save_dir)]
    dir_paths = _add_parents(dir_paths, levels=2)
    for dir_path in dir_paths:
        try:
            os.chmod(dir_path, 0o777)
        except PermissionError:
            logger.warning(f"Failed to chmod {dir_path} directory")


def _add_parents(dir_paths: list[Path], levels=2) -> list[Path]:
    """
    Returns dir_paths
    + all parent directories up to levels deep
    """
    dir_path_parents = [dir_path.parent for dir_path in dir_paths]
    dir_paths.extend(dir_path_parents)
    if levels > 1:
        dir_paths = _add_parents(dir_paths, levels=levels - 1)
    return dir_paths


def save_config(config: DictConfig) -> None:
    try:
        hydra_dir = HydraConfig.get().run.dir
    except:
        logger.warning("Hydra dir not found")
        hydra_dir = None
    create_log_and_save_dirs(config, hydra_dir=hydra_dir)

    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    git_hash = get_git_hash()
    config_dict["git_hash"] = git_hash

    logger.info(pformat(config_dict))
    with open(Path(config.logs_dir) / "config.json", "w") as fp:
        json.dump(config_dict, fp)

    with open(Path(config.save_dir) / "config.json", "w") as fp:
        json.dump(config_dict, fp)


def set_seed(seed):
    """Set global random seeds, including Python random, NumPy and PyTorch."""
    transformers.set_seed(seed)


if __name__ == "__main__":
    main()
