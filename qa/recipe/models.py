"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Our favorite collection of models
"""

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Tuple

try:
    import google.generativeai as genai
except ImportError:
    genai = None
import openai
import torch
from openai import AsyncAzureOpenAI, AzureOpenAI
try:
    from transformers import AutoTokenizer, pipeline
except ImportError:
    AutoTokenizer = None
    pipeline = None
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from recipe.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class InferenceModel(ABC):
    """Can be API or model from vLLM"""

    def respond(self, questions: list[str]) -> list[str]:
        pass

    @property
    def name(self):
        return self.__class__.__name__


class DummyModel(InferenceModel):
    "A model that always abstains"

    def respond(self, questions: list[str]) -> list[str]:
        response = "I don't know"
        return len(questions) * [response]


class TinyLlamaChat(InferenceModel):
    def __init__(self, **kwargs):
        if pipeline is None:
            raise ImportError(
                "transformers is required for TinyLlamaChat. Install it with pip install transformers."
            )
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def respond(self, questions: list[str]) -> list[str]:
        """Hack to emulate batched inference"""
        responses = []
        for question in questions:
            response = self.respond_single_question(question)
            responses.append(response)
        return responses

    def respond_single_question(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": question,
            },
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            # defaults from HuggingFace,
            # we can alter these later if we want
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

        full_chat = outputs[0]["generated_text"]
        response = full_chat[full_chat.find("<|assistant|>") + 13 :]
        return response


class VLLMChatModelBase(InferenceModel):
    """
    Uses offline inference from VLLM
    Note this needs V100 or more modern GPUs to work

    outputs = self.llm.generate(prompts, sampling_params)
    returns
    Each output in outputs is a RequestOutput that looks like:
    RequestOutput(request_id=0, prompt='What color is the sky?',
    prompt_token_ids=[128000, 3923, 1933, 374, 279, 13180, 30],
    encoder_prompt=None,
    encoder_prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[
        CompletionOutput(index=0, text=' It’s blue! But have you ever stopped to think about why it is blue',
            token_ids=(1102, 753, 6437, 0, 2030, 617, 499, 3596, 10717, 311, 1781, 922, 3249, 433, 374, 6437),
            cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)],
            finished=True,
            metrics=RequestMetrics(arrival_time=1731704452.5715117,
            last_token_time=1731704452.5715117,
            first_scheduled_time=1731704452.57341,
            first_token_time=1731704452.6467705,
            time_in_queue=0.0018982887268066406,
            finished_time=1731704453.022214,
            scheduler_time=0.0017652076203376055,
            model_forward_time=None,
            model_execute_time=None),
            lora_request=None)
        ]
    To get the response text: output.outputs[0].text

    """

    def __init__(
        self,
        model_path: str,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        max_model_len=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=False,
        revision=None,
        tokenizer_revision=None,
        enforce_eager=False,
        skip_special_tokens=True,
        use_system_prompt=False,
    ):
        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vllm is required for local model inference. Install vllm to use VLLMChatModelBase models."
            )
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_path
        self.max_tokens = max_tokens
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            skip_special_tokens=skip_special_tokens,
        )
        self.use_system_prompt = use_system_prompt

        # empty any unused GPU memory
        torch.cuda.empty_cache()

        logger.info("Creating a vllm.LLM instance!")
        self.llm = LLM(
            model=model_path,
            # bfloat is support in V100
            dtype=torch.float16,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,  # Context window
            # Note that context window size does not appear to have a significant effect on inference times:
            # On QASPER Llama 70B with 8k context took 1h06, vs. 1h03 for a 32k context.
            max_num_batched_tokens=max_model_len,  # Max number of tokens to be processed in a batch - should be >= context window to avoid effectively truncating context
            # number of gpus over which to split model inference
            tensor_parallel_size=tensor_parallel_size,
            # Introduced because of DeepSeek model
            trust_remote_code=trust_remote_code,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            # These parameters might help in the future
            # with memory optimization and/or multi-node inference
            enforce_eager=enforce_eager,
            distributed_executor_backend=None,
        )
        logger.info("Finished creating a vllm.LLM instance!")

        if convert_prompt_to_chat:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required for chat prompt conversion. Install it with pip install transformers."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = None

    def question_to_chat_format(self, question: str) -> str:
        # Tokenization and chat format inspired by the recipe
        # from https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
        if self.use_system_prompt:
            prompt = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": question}]
        else:
            prompt = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    def respond(self, questions: list[str]) -> list[str]:
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]
        if self.tokenizer:
            formatted_prompts = [
                self.question_to_chat_format(question) for question in questions
            ]
        else:
            formatted_prompts = questions
        outputs = self.llm.generate(
            formatted_prompts, self.sampling_params, use_tqdm=False
        )
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text)

        logger.info(responses[:3])
        return responses


class Llama3_1_8B_Instruct(VLLMChatModelBase):
    def __init__(
        self,
        local_model_path="/large_experiments/robust_vlm/abstention-bench/huggingface/Llama-3.1-8B-Instruct",
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=1,
        max_model_len=32768,
        use_system_prompt=False,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        model_path = (
            local_model_path if local_model_path is not None else _VLLM_MODEL_NAME
        )
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            use_system_prompt=use_system_prompt,
        )


class Llama3_1_70B_Instruct(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class Llama3_1_405B_Instruct_FP8(VLLMChatModelBase):
    def __init__(
        self,
        local_model_path="/large_experiments/robust_vlm/abstention-bench/huggingface/Llama-3.1-405B-Instruct-FP8",
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.1-405B-FP8"
        model_path = (
            local_model_path if local_model_path is not None else _VLLM_MODEL_NAME
        )
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class Llama3_3_70B_Instruct(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class OLMo_7B_0724_Instruct(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        max_model_len=4096,
        tensor_parallel_size=1,
    ):
        _VLLM_MODEL_NAME = "allenai/OLMo-7B-0724-Instruct-hf"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )


class Mistral_7B_Instruct_v0_3(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=1,
        max_model_len=32768,
    ):
        _VLLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


class Qwen2_5_32B_Instruct(VLLMChatModelBase):
    def __init__(
        self,
        local_model_path="/large_experiments/robust_vlm/abstention-bench/huggingface/Qwen2.5-32B-Instruct",
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    ):
        _VLLM_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
        model_path = (
            local_model_path if local_model_path is not None else _VLLM_MODEL_NAME
        )
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )


### REASONING MODELS ###


class VLLMReasoningChatModelBase(VLLMChatModelBase):

    def __init__(self, max_reasoning_tokens=2000, *args, **kwargs):
        self.MAX_REASONING_TOKENS = max_reasoning_tokens
        # This will setup vLLM instance self.llm
        # and self.tokenizer.
        super().__init__(*args, **kwargs)

        # Make sure we have a defined tokenizer.
        assert self.tokenizer

        # Separately define sampling parameters for
        # reasoning chain sampling and final answer sampling.
        # Key difference between two sampling params are
        # max_tokens and stop_token_ids.
        self.sampling_params_reasoning = SamplingParams(
            max_tokens=self.MAX_REASONING_TOKENS,
            stop_token_ids=self.tokenizer(self.STOP_TOKENS_FOR_REASONING)["input_ids"],
            temperature=self.temperature,
            top_p=self.top_p,
            skip_special_tokens=False,
        )
        self.sampling_params_answer = SamplingParams(
            max_tokens=self.max_tokens,
            stop_token_ids=self.tokenizer(self.STOP_TOKENS)["input_ids"],
            temperature=self.temperature,
            top_p=self.top_p,
            skip_special_tokens=False,
        )

    def respond(
        self, questions: list[str],
    ) -> list[str]:

        formatted_prompts = [
            self.question_to_chat_format(question) for question in questions
        ]

        outputs_reasoning = self.generate_reasoning(formatted_prompts)
        reasoning_list, output_final_answer = self.generate_response(formatted_prompts, outputs_reasoning)

        reasonings = []
        responses = []
        responses_with_reasonings = []
        for reasoning, answer in zip(reasoning_list, output_final_answer):
            reasoning_text, answer_text, response_with_reasoning_text = self.post_process_responses_with_reasoning(
                reasoning, answer
            )
            reasonings.append(reasoning_text)
            responses.append(answer_text)
            responses_with_reasonings.append(response_with_reasoning_text)

        logger.info(f"Responses with reasonings: {responses_with_reasonings[0]}")
        return responses, reasonings, responses_with_reasonings
    
    def generate_reasoning(self, formatted_prompts):
        # This forward pass will only generate reasoning chains and
        # will NOT generate the final answers.
        outputs_reasoning = self.llm.generate(
            formatted_prompts,
            self.sampling_params_reasoning,
            use_tqdm=False
        )
        logger.info(f"Reasoning: {outputs_reasoning[0].outputs[0].text}")
        return outputs_reasoning

    def generate_response(self, formatted_prompts, outputs_reasoning):
        # Updating the prompt to include generated reasoning chains and
        # and start-of-final-answer token to trigger final response generation.
        reasoning_list = []
        prompts_with_reasoning_chains = []
        for i in range(len(formatted_prompts)):
            prompt_chain = formatted_prompts[i] + outputs_reasoning[i].outputs[0].text
            prompt_chain += self.START_OF_FINAL_RESPONSE_TOKEN
            prompts_with_reasoning_chains.append(prompt_chain)
            reasoning_list.append(outputs_reasoning[i].outputs[0].text)

        # This forward pass will only generate the final answers,
        # based on request and reasoning chain.
        output_final_answer = self.llm.generate(
            prompts_with_reasoning_chains,
            sampling_params=self.sampling_params_answer,
            use_tqdm=False,
        )
        logger.info(f"Final answer: {output_final_answer[0].outputs[0].text}")
        return reasoning_list, output_final_answer

    def post_process_responses_with_reasoning(self, reasoning_text, answer):
        answer_text = "Final Answer: " + answer.outputs[0].text

        # DeepSeek is stubborn and may continue thinking after the first forced </think>.
        if "</think>" in answer_text:
            response_parts = answer_text.split("</think>")
            for j in range(len(response_parts)-1):
                reasoning_text += response_parts[j]
            answer_text = response_parts[-1]

        response_with_reasoning_text = "Reasoning Chain:\n" + reasoning_text + "\n" + "Final Answer:\n" + answer_text
        return reasoning_text, answer_text, response_with_reasoning_text


class DeepSeek_R1_Distill_Llama_70B(VLLMReasoningChatModelBase):
    """In this version of DeepSeek tokenizer template forces the model
    to start generation from thinking token <think>."""

    def __init__(
        self,
        local_model_path="/large_experiments/robust_vlm/abstention-bench/huggingface/DeepSeek-R1-Distill-Llama-70B",
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        revision="b1c0b44b4369b597ad119a196caf79a9c40e141e",
        tokenizer_revision="b1c0b44b4369b597ad119a196caf79a9c40e141e",
        skip_special_tokens=True,
        enforce_eager=True,
        max_reasoning_tokens=4000,
        math_budget_forcing=False,
        use_system_prompt=False,
    ):
        _VLLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        model_path = (
            local_model_path if local_model_path is not None else _VLLM_MODEL_NAME
        )
        self.math_budget_forcing = math_budget_forcing
        self.STOP_TOKENS_FOR_REASONING = "</think><｜end▁of▁sentence｜>"
        self.STOP_TOKENS = "<｜end▁of▁sentence｜>"
        # This string is strongly correlated with the end of reasoning chain.
        self.ANSWER_TRIGGER_TOKEN = "\n\n**Final Answer**\n\\boxed{"
        self.START_OF_FINAL_RESPONSE_TOKEN = "\n</think>\n"
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            skip_special_tokens=skip_special_tokens,
            enforce_eager=enforce_eager,
            max_reasoning_tokens=max_reasoning_tokens,
            use_system_prompt=use_system_prompt,
        )

    def generate_reasoning(self, formatted_prompts):
        if not self.math_budget_forcing:
            return super().generate_reasoning(formatted_prompts)
        else:
            return self.generate_reasoning_math(formatted_prompts)

    def generate_response(self, formatted_prompts, outputs_reasoning):
        if not self.math_budget_forcing:
            return super().generate_response(formatted_prompts, outputs_reasoning)
        else:
            return self.generate_response_math(formatted_prompts, outputs_reasoning)

    def generate_reasoning_math(self, formatted_prompts):
        # This forward pass will only generate a reasoning chain until one of the following happens:
        # 1. We hit </think>
        # 2. We hit "**Final Answer**\n\\boxed{" variation.
        # 3. We hit MAX_REASONING_TOKENS tokens.
        self.sampling_params_reasoning = SamplingParams(
            max_tokens=self.MAX_REASONING_TOKENS,
            stop_token_ids=self.tokenizer(self.STOP_TOKENS_FOR_REASONING)["input_ids"],
            stop=[
                "**Final Answer**\n\\boxed{",
                "**Final Answer**\n\n\\boxed{",
                "**Final Answer** \\boxed{",
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            skip_special_tokens=False,
        )

        outputs_reasoning = self.llm.generate(
            formatted_prompts,
            self.sampling_params_reasoning,
            use_tqdm=False,
        )
        logger.info(f"Reasoning: {outputs_reasoning[0].outputs[0].text}")
        return outputs_reasoning

    def generate_response_math(self, formatted_prompts, outputs_reasoning):
        # Updating the prompt to include generated reasoning chains and
        # and a string to trigger final response generation in math "**Final Answer**\n\\boxed{".
        prompts_with_reasoning_chains = []
        for i in range(len(formatted_prompts)):
            prompt_chain = formatted_prompts[i] + outputs_reasoning[i].outputs[0].text + self.ANSWER_TRIGGER_TOKEN
            prompts_with_reasoning_chains.append(prompt_chain)

        sampling_params_boxed_final_answer = SamplingParams(
            max_tokens=100,
            stop=["}"],
            include_stop_str_in_output=True,
            temperature=self.temperature,
            top_p=self.top_p,
            skip_special_tokens=False,
        )

        # This forward pass will only generate the final answer
        # in the "\boxed" brackets to conclude the reasoning chain.
        boxed_final_answer = self.llm.generate(
            prompts_with_reasoning_chains,
            sampling_params=sampling_params_boxed_final_answer,
            use_tqdm=False,
        )

        logging.info(f"Boxed answer: {boxed_final_answer[0].outputs[0].text}")

        # Updating prompts with boxed answers and START_OF_FINAL_RESPONSE_TOKEN.
        final_reasoning_list = []
        prompts_with_reasoning_chains_and_boxed_ans = []
        for i in range(len(prompts_with_reasoning_chains)):
            final_reasoning = outputs_reasoning[i].outputs[0].text + self.ANSWER_TRIGGER_TOKEN + boxed_final_answer[i].outputs[0].text
            prompt_chain = (
                prompts_with_reasoning_chains[i] + \
                boxed_final_answer[i].outputs[0].text + \
                self.START_OF_FINAL_RESPONSE_TOKEN + "\nFinal answer: "
            )
            final_reasoning_list.append(final_reasoning)
            prompts_with_reasoning_chains_and_boxed_ans.append(prompt_chain)

        # This forward pass will *hopefully* only generate the final answers.
        output_final_answer = self.llm.generate(
            prompts_with_reasoning_chains_and_boxed_ans,
            sampling_params=self.sampling_params_answer,
            use_tqdm=False,
        )
        logger.info(f"Final answer: {output_final_answer[0].outputs[0].text}")
        return final_reasoning_list, output_final_answer


class S1_1_32B(VLLMReasoningChatModelBase):
    """
    Implements S1 reasoning model
    This model is based on Qwen/Qwen2.5-32B-Instruct
    """

    def __init__(
        self,
        local_model_path="/large_experiments/robust_vlm/abstention-bench/huggingface/s1.1-32B",
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        skip_special_tokens=False,
        max_reasoning_tokens=4000,
        use_system_prompt=False,
    ):
        _VLLM_MODEL_NAME = "simplescaling/s1.1-32B"
        model_path = (
            local_model_path if local_model_path is not None else _VLLM_MODEL_NAME
        )
        self.STOP_TOKENS_FOR_REASONING = "<|im_start|><|im_end|>"
        self.STOP_TOKENS = "<|im_end|>"
        self.START_OF_FINAL_RESPONSE_TOKEN = "\n<|im_start|>answer\nFinal Answer: "
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            skip_special_tokens=skip_special_tokens,
            max_reasoning_tokens=max_reasoning_tokens,
            use_system_prompt=use_system_prompt,
        )

    def question_to_chat_format(self, question: str) -> str:
        if self.use_system_prompt:
            prompt = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": question}]
        else:
            prompt = [{"role": "user", "content": question}]
        prompt_chat = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        # Overriding template to start from thinking chain.
        # This is already included in DeepSeek-R1 tokenizer config.
        prompt_chat += "<|im_start|>think\n"
        return prompt_chat


### TULU POST-TRAINING MODELS ###


class Llama3_1_8B_Base(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=False,
        tensor_parallel_size=1,
        max_model_len=32768,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


class Llama3_1_8B_Tulu_3_SFT(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=1,
        max_model_len=32678,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-SFT"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


class Llama3_1_8B_Tulu_3_DPO(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=1,
        max_model_len=32678,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-DPO"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


class Llama3_1_8B_Tulu_3_PPO_RLVF(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=1,
        max_model_len=32678,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )


class Llama3_1_70B_Base(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=False,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "meta-llama/Llama-3.1-70B"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class Llama3_1_70B_Tulu_3_SFT(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-70B-SFT"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class Llama3_1_70B_Tulu_3_DPO(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-70B-DPO"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


class Llama3_1_70B_Tulu_3_PPO_RLVF(VLLMChatModelBase):
    def __init__(
        self,
        temperature=0.8,
        top_p=0.95,
        max_tokens=None,
        convert_prompt_to_chat=True,
        tensor_parallel_size=8,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
    ):
        _VLLM_MODEL_NAME = "allenai/Llama-3.1-Tulu-3-70B"
        super().__init__(
            model_path=_VLLM_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            convert_prompt_to_chat=convert_prompt_to_chat,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )


### API MODELS ###


# From https://github.com/fairinternal/VisualWebArena-Privacy/blob/18c5daf6ab26ca207c91adf1b2483e138055db8d/llms/providers/openai_utils.py
def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def retry_with_exponential_backoff_async(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
    max_delay: int = 30,  # Don't delay more than 60s
    errors=(
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    async def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return await func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                delay = min(delay, max_delay)

                logger.info(f"Request failed - retrying after {delay:.2f}s")

                # Sleep for the delay
                await asyncio.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


class AsyncOpenAIModel(InferenceModel):

    def __init__(
        self,
        host,
        api_key,
        model_name,
        api_version,
        max_completion_tokens=None,
        temperature=0.8,
        seed=None,
    ):
        self.api_version = api_version
        self.api_key = api_key
        self.host = host

        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.seed = seed

    def respond(self, questions: list[str]) -> list[str]:
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]

        # Set up a fresh client for each batch because calling gather appears to terminate
        # the event loop, causing all subsequent batches to fail without a fresh client.
        self.client = AsyncAzureOpenAI(
            api_version=self.api_version,
            # needs to be set up in bash
            api_key=self.api_key,
            azure_endpoint=f"https://{self.host}",
            max_retries=0,  # Disable built-in retries so we can customize timing
        )

        responses = asyncio.run(self._respond_async(questions))
        return responses

    async def _respond_async(self, questions: list[str]) -> list[str]:
        """Call the API for every question in this batch concurrently, and return when all requests are completed."""
        tasks = [
            self._call_api_and_catch_exceptions(question) for question in questions
        ]
        return await asyncio.gather(*tasks)

    async def _call_api_and_catch_exceptions(self, question: str) -> str:
        """Call the API and handle errors so as not to interrupt other concurrent requests."""
        try:
            response = await self._call_api(question)
            return response
        except Exception as e:
            logger.error(
                f"After successive retries, an error occurred when calling the OpenAI API: {str(e)}"
            )
            return ''

    @retry_with_exponential_backoff_async
    async def _call_api(self, question: str) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )

            generated_text = completion.choices[0].message.content

            return generated_text
        except Exception as e:
            logger.error(f"Exception for prompt: {question}")
            raise e


class OpenAIModel(InferenceModel):

    def __init__(
        self,
        host,
        api_key,
        model_name,
        api_version,
        max_completion_tokens=None,
        temperature=0.8,
        seed=None,
    ):

        self.client = AzureOpenAI(
            api_version=api_version,
            # needs to be set up in bash
            api_key=api_key,
            azure_endpoint=f"https://{host}",
        )
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.seed = seed

    @retry_with_exponential_backoff
    def respond(self, questions: list[str]) -> list[str]:
        """TODO: check if there's a batch API.
        Right now making iterative calls as a hack"""
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]
        responses = []
        for question in questions:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                seed=self.seed,
            )
            generated_text = completion.choices[0].message.content
            logger.info(generated_text)
            responses.append(generated_text)
        return responses


class GPT4oAPI(AsyncOpenAIModel):
    def __init__(
        self,
        max_tokens=None,
        temperature=0.8,
        seed=None,
    ):
        MODEL_NAME = "gpt-4o"
        API_VERSION = "2024-10-21"
        HOST = "azure-services-fair-openai1-northcentralus.azure-api.net"
        if "AZURE_GPT4O_API_KEY" not in os.environ:
            raise ValueError(
                "AZURE_GPT4O_API_KEY environment variable must be set when using OpenAI API. "
                "See README.md on instructions how to obtain it and set it up."
            )
        api_key = os.environ["AZURE_GPT4O_API_KEY"]
        super().__init__(
            host=HOST,
            api_key=api_key,
            model_name=MODEL_NAME,
            api_version=API_VERSION,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )


class o1PreviewAPI(AsyncOpenAIModel):
    def __init__(
        self,
        max_tokens=None,
        seed=None,
    ):
        MODEL_NAME = "o1-preview"
        API_VERSION = "2024-12-01-preview"
        HOST = "azure-services-fair-openai1-southcentralusn2.azure-api.net"
        if "AZURE_O1_PREVIEW_API_KEY" not in os.environ:
            raise ValueError(
                "AZURE_O1_PREVIEW_API_KEY environment variable must be set when using OpenAI API. "
                "See README.md on instructions how to obtain it and set it up."
            )
        api_key = os.environ["AZURE_O1_PREVIEW_API_KEY"]
        # Important note on temperature with o1:
        # setting it to anything except for 1 produces
        # "Unsupported value: 'temperature' does not support 0.8 with this model.
        # Only the default (1) value is supported."
        super().__init__(
            host=HOST,
            api_key=api_key,
            model_name=MODEL_NAME,
            api_version=API_VERSION,
            max_completion_tokens=max_tokens,
            temperature=1,
            seed=seed,
        )


class o1API(AsyncOpenAIModel):
    def __init__(
        self,
        max_tokens=None,
        seed=None,
    ):
        MODEL_NAME = "o1"
        API_VERSION = "2024-12-01-preview"
        HOST = "azure-services-fair-openai2-eastus2n2.azure-api.net"
        if "AZURE_O1_API_KEY" not in os.environ:
            raise ValueError(
                "AZURE_O1_API_KEY environment variable must be set when using OpenAI API. "
                "See README.md on instructions how to obtain it and set it up."
            )
        api_key = os.environ["AZURE_O1_API_KEY"]
        # Important note on temperature with o1:
        # setting it to anything except for 1 produces
        # "Unsupported value: 'temperature' does not support 0.8 with this model.
        # Only the default (1) value is supported."
        super().__init__(
            host=HOST,
            api_key=api_key,
            model_name=MODEL_NAME,
            api_version=API_VERSION,
            max_completion_tokens=max_tokens,
            temperature=1,
            seed=seed,
        )


class TogetherAIAPI(InferenceModel):
    def __init__(
        self,
        together_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=None,
        temperature=0.8,
        seed=None,
        request_timeout_seconds=180,
    ):
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError(
                "TOGETHER_API_KEY environment variable must be set when using Together API."
            )
        api_key = os.environ["TOGETHER_API_KEY"]
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
            timeout=request_timeout_seconds,
        )
        self.model_name = together_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_timeout_seconds = request_timeout_seconds

    @retry_with_exponential_backoff
    def respond(self, questions: list[str]) -> list[str]:
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]

        responses = []
        for question in questions:
            request_kwargs = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
            }
            if self.max_tokens is not None:
                request_kwargs["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.seed is not None:
                request_kwargs["seed"] = self.seed

            completion = self.client.chat.completions.create(**request_kwargs)
            generated_text = completion.choices[0].message.content
            logger.info(generated_text)
            responses.append(generated_text)
        return responses


ACE_GENERATOR_PROMPT = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list
- final_answer: your concise final answer


**Playbook:**
## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS

**Reflection:**
(empty)

**Question:**
{}

**Context:**
{}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer here]"
}}

---
"""

ACE_CONTEXT_MAP = {
    "umwp": (
        "You are answering a math word problem. "
        "If the question lacks sufficient information to produce a definitive answer, "
        "you must say so clearly instead of guessing. "
        "Only provide a numeric answer when the problem is fully solvable."
    ),
    "selfaware": (
        "You are answering a knowledge question. "
        "If you are unsure about the answer or if the answer is genuinely unknown, "
        "say so clearly instead of making something up."
    ),
    "falseqa": (
        "You are answering a question. "
        "If the question contains a false or incorrect assumption, "
        "point out the false premise instead of answering the question as-is."
    ),
}


class TogetherAIAPI_ACEFormat(InferenceModel):
    """Same as TogetherAIAPI but wraps questions in ACE's generator prompt format."""

    def __init__(
        self,
        together_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=None,
        temperature=0.8,
        seed=None,
        request_timeout_seconds=180,
        ace_context_key="umwp",
    ):
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError(
                "TOGETHER_API_KEY environment variable must be set when using Together API."
            )
        api_key = os.environ["TOGETHER_API_KEY"]
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
            timeout=request_timeout_seconds,
        )
        self.model_name = together_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_timeout_seconds = request_timeout_seconds
        self.context = ACE_CONTEXT_MAP.get(ace_context_key, "Answer the following question accurately.")

    @retry_with_exponential_backoff
    def respond(self, questions: list[str]) -> list[str]:
        if isinstance(questions, str):
            questions = [questions]

        responses = []
        for question in questions:
            ace_prompt = ACE_GENERATOR_PROMPT.format(question, self.context)

            request_kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": ace_prompt}],
            }
            if self.max_tokens is not None:
                request_kwargs["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.seed is not None:
                request_kwargs["seed"] = self.seed

            completion = self.client.chat.completions.create(**request_kwargs)
            generated_text = completion.choices[0].message.content
            logger.info(generated_text)
            responses.append(generated_text)
        return responses


class GoogleAPIModel(InferenceModel):

    def __init__(
        self,
        api_key,
        model_name,
        max_output_tokens=None,
        temperature=0.8,
    ):
        if genai is None:
            raise ImportError(
                "google-generativeai is required for Gemini API models. Install it with pip install google-generativeai."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @retry_with_exponential_backoff
    def respond(self, questions: list[str]) -> list[str]:
        """TODO: check if there's a batch API.
        Right now making iterative calls as a hack"""
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]

        responses = []
        for question in questions:

            # TODO: there is no seed parameter in this version of API.
            # https://github.com/google-gemini/generative-ai-python/issues/605
            response = self.model.generate_content(
                question,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
                # Don't block anything for evaluation.
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "block_none",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
                    "HARM_CATEGORY_HATE_SPEECH": "block_none",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                },
            )
            if response.parts:
                generated_text = response.text
            else:
                generated_text = "No response generated."
            logger.info(generated_text)
            responses.append(generated_text)

        return responses


class Gemini15ProAPI(GoogleAPIModel):
    def __init__(
        self,
        max_tokens=None,
        temperature=0.8,
    ):
        MODEL_NAME = "gemini-1.5-pro"
        if "GOOGLE_GENAI_API_KEY" not in os.environ:
            raise ValueError(
                "GOOGLE_GENAI_API_KEY environment variable must be set when using Gemini API. "
            )
        api_key = os.environ["GOOGLE_GENAI_API_KEY"]
        super().__init__(
            api_key=api_key,
            model_name=MODEL_NAME,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )


if __name__ == "__main__":
    model = S1_1_32B()
    print(model.respond(["what color is the sky?"]))
    print(
        model.respond(
            [
                "I saw a grandson and their grandfather last week "
                "outside the Walmart trying to book a cab on Uber. "
                "Who was not comfortable using the phone?"
            ]
        )
    )
