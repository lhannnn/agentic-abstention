# Metrics

Canonical metrics use all abstain-warranted episodes as the denominator. Let
`o_i` be the oracle abstention step and `a_i` the first valid abstention step.

## Timely Recall

Timely Recall is the fraction of episodes with `a_i = o_i`.

Terminal delayed tasks instead report `timely_delayed_recall`, which counts
abstentions from `earliest_abstain_turn` through
`earliest_abstain_turn + timely_grace_turns`.

## Overall Recall

Overall Recall is the fraction of episodes with a valid abstention at or after
the oracle step and within the interaction budget.

## SPL

SPL is the mean per-episode contribution:

```text
contribution_i = o_i / max(a_i, o_i)
```

The contribution is 0 if no valid abstention occurs or if the agent abstains
before the oracle step.

## pass@k

Web and Q&A define `pass@k` as the fraction of episodes satisfying
`0 <= a_i - o_i <= k`. Q&A reports `pass@0` through `pass@10`; its
`early_recall` output field is Timely Recall.

Terminal immediate tasks use absolute observe-act turns and report Timely
Recall as Pass@1. Terminal delayed summaries use the timing-window metrics above
and do not report SPL or pass@k.

## Invalid outputs

Canonical summaries include parser-invalid outputs. Q&A files named
`*ExcludeInvalid*` report the alternate valid-only denominator.
