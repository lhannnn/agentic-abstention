#!/usr/bin/env python3
"""Rewrite source-500 rows into the false-premises category with GPT-5-mini."""

from webshop_abstain_common import (
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REWRITE249_OUTPUTS,
    DEFAULT_REWRITE249_PLAN_PATH,
    POSITIVE_CATEGORIES_V2,
    PROMPT_VERSION_V2,
)
from webshop_abstain_rewrite_driver import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_category="false_premises",
            default_bucket_plan=DEFAULT_REWRITE249_PLAN_PATH,
            default_outputs=DEFAULT_REWRITE249_OUTPUTS,
            allowed_categories=POSITIVE_CATEGORIES_V2,
            default_prompt_version=PROMPT_VERSION_V2,
            default_model=DEFAULT_OPENAI_MODEL,
            default_base_url=DEFAULT_OPENAI_BASE_URL,
            default_api_key_env=DEFAULT_OPENAI_API_KEY_ENV,
        )
    )
