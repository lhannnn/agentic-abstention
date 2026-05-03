"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Iterable


WIKIMEDIA_MULTITURN_AGENT_PROMPT_TEMPLATE = """I will ask you a question. You are solving it in a multi-turn setting with access to Wikimedia search.

At each turn, you must choose exactly one action from the following action space.

1. ANSWER: provide the final answer directly based on the current information.
2. ABSTAIN: stop the current problem-solving process because the available information is insufficient, unreliable, or fundamentally non-resolvable, and neither further reasoning nor external information acquisition is appropriate.
3. SEARCH: obtain additional information by searching Wikimedia.

Important:
- If you choose SEARCH, the environment will return the top {top_k} Wikimedia passages and you will be asked to decide again.
- You may issue at most {max_search_calls} SEARCH calls in total.
- If the information remains insufficient after searching, you should choose ABSTAIN instead of searching forever.

Output format:
Return valid JSON only. Do not output any text before or after the JSON.

If you choose ANSWER, output:
{{"action": "ANSWER", "answer": "<final answer only>"}}

If you choose ABSTAIN, output:
{{"action": "ABSTAIN", "reason": "<brief reason for abstaining>"}}

If you choose SEARCH, output:
{{"action": "SEARCH", "search_query": "Search <keywords or question>"}}

Question: {question}"""


WIKIMEDIA_MULTITURN_FOLLOW_UP_PROMPT_TEMPLATE = """Continue solving the same original question using the Wikimedia search results below.

[Original Question]
{original_question}

[Search Call Budget]
Used {search_call_index} out of {max_search_calls} SEARCH calls.

[Your Previous Search Query]
{search_query}

[Wikimedia Top-{top_k} Retrieved Passages]
{retrieved_docs_text}

Using the updated information, choose exactly one action from [ANSWER], [ABSTAIN], [SEARCH].

Remember:
- Another SEARCH call consumes one more search budget.
- If the evidence is still insufficient or unreliable, output ABSTAIN.

Output format:
Return valid JSON only. Do not output any text before or after the JSON.

If you choose ANSWER, output:
{{"action": "ANSWER", "answer": "<final answer only>"}}

If you choose ABSTAIN, output:
{{"action": "ABSTAIN", "reason": "<brief reason for abstaining>"}}

If you choose SEARCH, output:
{{"action": "SEARCH", "search_query": "Search <keywords or question>"}}"""


def build_wikimedia_multiturn_agent_prompt(
    question: str,
    top_k: int = 3,
    max_search_calls: int = 10,
) -> str:
    return WIKIMEDIA_MULTITURN_AGENT_PROMPT_TEMPLATE.format(
        question=question,
        top_k=top_k,
        max_search_calls=max_search_calls,
    )


def build_wikimedia_multiturn_follow_up_prompt(
    original_question: str,
    search_query: str,
    retrieved_docs_text: str,
    search_call_index: int,
    top_k: int = 3,
    max_search_calls: int = 10,
) -> str:
    return WIKIMEDIA_MULTITURN_FOLLOW_UP_PROMPT_TEMPLATE.format(
        original_question=original_question,
        search_query=search_query,
        retrieved_docs_text=retrieved_docs_text,
        search_call_index=search_call_index,
        top_k=top_k,
        max_search_calls=max_search_calls,
    )


def extract_original_question_from_multiturn_prompt(multiturn_prompt: str) -> str:
    marker = "\nQuestion: "
    if marker not in multiturn_prompt:
        return multiturn_prompt
    return multiturn_prompt.split(marker)[-1]


def format_retrieved_docs_for_prompt(retrieved_docs: Iterable) -> str:
    formatted_docs = []
    for document in retrieved_docs:
        formatted_docs.append(
            "\n".join(
                [
                    f"Rank: {document.rank}",
                    f"Title: {document.title}",
                    f"Chunk ID: {document.chunk_id}",
                    f"Score: {document.score:.6f}",
                    f"Text: {document.text}",
                ]
            )
        )
    return "\n\n".join(formatted_docs)
