# Metrics

Metrics are computed over abstain-warranted samples unless otherwise stated.

## Timely Recall

Timely Recall is the fraction of abstain-warranted samples where the agent
abstains at the earliest oracle decision point.

For the Q&A Wikimedia setup, the oracle abstention step is 1, so Timely Recall
is equivalent to `pass@0`.

## Overall Recall

Overall Recall is the fraction of abstain-warranted samples where the agent
eventually produces a valid abstention within the allowed interaction budget.

For the Q&A Wikimedia setup with `max_search_calls = 10`, Overall Recall is
equivalent to `pass@10`.

## SPL

SPL rewards earlier abstention:

```text
SPL contribution = oracle_step / max(first_abstain_step, oracle_step)
```

If the agent never abstains, the contribution is 0.

## pass@k

`pass@k` is the fraction of abstain-warranted samples where the first valid
abstention happens no more than `k` steps after the oracle abstention step.

```text
pass@k = count(first_abstain_step - oracle_step <= k) / num_abstain_samples
```

This release reports `pass@0` through `pass@10` for Q&A episodes.

## Invalid Outputs

Some appendix analyses exclude parser-invalid outputs from the denominator.
Those outputs are clearly labeled as exclude-invalid results and do not replace
canonical summaries unless explicitly stated.
