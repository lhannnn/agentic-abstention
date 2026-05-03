"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

ALCUNA_MULTITURN_AGENT_PROMPT_TEMPLATE = """I will ask you a biological question. You are solving it in a multi-turn setting.

At each turn, you must choose exactly one action from the following action space.

1. ANSWER: provide the final answer directly based on the current information.
2. ABSTAIN: stop the current problem-solving process because the available information is insufficient, unreliable, or fundamentally non-resolvable, and neither further reasoning nor external information acquisition is appropriate.
3. SEARCH: obtain additional information by performing an external search.

Important:
- If you choose SEARCH, the environment will return search results and you will be asked to decide again.
- You may interact with the environment for at most 10 turns in total.
- If the information remains insufficient, you should choose ABSTAIN instead of searching forever.

Output format:
Line 1: exactly one of [ANSWER], [ABSTAIN], [SEARCH]

Line 2:
- if [ANSWER], provide the final answer only
- if [ABSTAIN], provide a brief reason for abstaining
- if [SEARCH], give the keywords or question to search in the format: Search [keywords or question]

Question: {question}"""

ALCUNA_RETRIEVABILITY_CHECKER_PROMPT_TEMPLATE = """You are a retrieval-grounding checker.

An agent is solving a question from the ALCUNA dataset and has issued a search query.

Your task is to determine whether this query refers to a known, retrievable biological species or concept, or instead refers to an unknown, fictional, unsupported, or ungrounded entity for which no reliable external information should be available.

Return valid JSON only:
{{
  "retrievability": "KNOWN_RETRIEVABLE" or "UNKNOWN_OR_UNGROUNDED",
  "explanation": "<brief explanation>"
}}

[Original Question]
{original_question}

[Search Query]
{search_query}"""

ALCUNA_RESPONSE_GENERATOR_PROMPT_TEMPLATE = """You are a biological search environment for the ALCUNA dataset.

An agent is answering a biology question and requested an external search.
You are given the original question, the search query, and the retrievability check result.

Your job is to generate the plain-text search result that the environment should return to the agent.

Rules:
- Return plain text only. Do not return JSON.
- Do not output action labels such as ANSWER, ABSTAIN, or SEARCH.
- If retrievability is KNOWN_RETRIEVABLE, return a concise search-result style snippet with grounded biological information that is relevant to the query and the original question.
- If retrievability is UNKNOWN_OR_UNGROUNDED, return that no reliable grounded external information was found, and briefly explain why the entity or concept appears fictional, unsupported, or not retrievable.
- Do not pretend you executed a real search engine. Write the content as a simulated environment response.
- Prefer concise, factual text over narration.

[Original Question]
{original_question}

[Search Query]
{search_query}

[Retrievability Checker Result]
{checker_result_json}"""

ALCUNA_MULTITURN_FOLLOW_UP_PROMPT_TEMPLATE = """Continue solving the same original question.

[Original Question]
{original_question}

[Your Previous Search Query]
{search_query}

[Environment Search Result]
{environment_response}

Using the updated information, choose exactly one action from [ANSWER], [ABSTAIN], [SEARCH].

Output format:
Line 1: exactly one of [ANSWER], [ABSTAIN], [SEARCH]

Line 2:
- if [ANSWER], provide the final answer only
- if [ABSTAIN], provide a brief reason for abstaining
- if [SEARCH], give the keywords or question to search in the format: Search [keywords or question]"""


def build_alcuna_multiturn_agent_prompt(question: str) -> str:
    return ALCUNA_MULTITURN_AGENT_PROMPT_TEMPLATE.format(question=question)


def build_alcuna_retrievability_checker_prompt(
    original_question: str,
    search_query: str,
) -> str:
    return ALCUNA_RETRIEVABILITY_CHECKER_PROMPT_TEMPLATE.format(
        original_question=original_question,
        search_query=search_query,
    )


def build_alcuna_response_generator_prompt(
    original_question: str,
    search_query: str,
    checker_result_json: str,
) -> str:
    return ALCUNA_RESPONSE_GENERATOR_PROMPT_TEMPLATE.format(
        original_question=original_question,
        search_query=search_query,
        checker_result_json=checker_result_json,
    )


def build_alcuna_multiturn_follow_up_prompt(
    original_question: str,
    search_query: str,
    environment_response: str,
) -> str:
    return ALCUNA_MULTITURN_FOLLOW_UP_PROMPT_TEMPLATE.format(
        original_question=original_question,
        search_query=search_query,
        environment_response=environment_response,
    )


def extract_original_question_from_multiturn_prompt(
    multiturn_prompt: str,
) -> str:
    marker = "\nQuestion: "
    if marker not in multiturn_prompt:
        return multiturn_prompt
    return multiturn_prompt.split(marker)[-1]
