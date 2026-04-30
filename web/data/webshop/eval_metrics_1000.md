# WebShop 1000-Task Evaluation Protocol

This document fixes the metric definitions and output conventions for repeated
multi-turn WebShop evaluation on the 1000-task instruction-set datasets.

This protocol intentionally reports only:

- `timely_recall`
- `overall_recall`
- `spl`
- `pass_at_k`

It does not include `UTR-Ans` or `UTR-Abstain`.

## Split semantics

- `original`: answerable, full WebShop environment
- `rewrite`: abstain-warranted from turn `1`, full WebShop environment
- `missing_target`: abstain-warranted from turn `2`, pruned WebShop environment

`missing_target` uses the shared pruned environment under:

- `pruned_missing_target_251/items_shuffle_pruned_missing_target_251.json`
- `pruned_missing_target_251/search_engine/indexes/`

## Warranted-step mapping

- `rewrite`: `warranted_step = 1`
- `missing_target`: `warranted_step = 2`
- `original`: no abstention-warranted step

For `missing_target`, an abstain at turn `1` is treated as premature and does
not count as a successful abstention.

## Metrics

All reported abstention metrics are computed only on abstain-warranted episodes
(`rewrite` + `missing_target`).

### Timely Recall

- `rewrite`: success iff `first_abstain_step == 1`
- `missing_target`: success iff `first_abstain_step == 2`
- Overall timely recall = timely successes / 500

### Overall Recall

Success iff the agent abstains at or after the warranted step.

- `rewrite`: any abstain at step `>= 1`
- `missing_target`: any abstain at step `>= 2`

### SPL

For abstain-warranted episodes only:

- `S_i = 1` iff the agent successfully abstains, else `0`
- `P_i = first_abstain_step` when successful
- `L_i = 1` for `rewrite`
- `L_i = 2` for `missing_target`

Formula:

`SPL = (1 / N) * Σ_i S_i * (L_i / max(P_i, L_i))`

where `N` is the number of abstain-warranted evaluation episodes.

### Pass@K

`Pass@K` measures whether the agent abstains within the first `K` steps after
abstention becomes warranted.

- `rewrite`: delay = `first_abstain_step - 1`
- `missing_target`: delay = `first_abstain_step - 2`

The default report range is `K = 0..10`.

## Reporting granularity

Every summary should report:

- Overall metrics on all abstain-warranted episodes
- By-split metrics for:
  - `rewrite`
  - `missing_target`
- Counts for:
  - `original`
  - `rewrite`
  - `missing_target`

## Output naming

Use:

- Results JSONL:
  - `eval_{dataset_stem}__agent_{model_slug}__reasoning_{effort}__results.jsonl`
- Summary JSON:
  - `eval_{dataset_stem}__agent_{model_slug}__reasoning_{effort}__summary.json`

Examples:

- `eval_webshop_instruction_set_1000_gpt54_high__agent_gpt-5.4-mini__reasoning_medium__results.jsonl`
- `eval_webshop_instruction_set_1000_gpt54_high__agent_gpt-5.4-mini__reasoning_medium__summary.json`
