# Benchmark Protocol

Agentic abstention tests whether an agent can decide when not to answer while
operating in an interactive environment.

## Environments

- Web: WebShop navigation with rewritten goals and missing-target cases.
- Q&A: AbstentionBench-style questions with a Wikimedia `SEARCH` tool.
- Terminal: TerminalBench tasks where the agent must inspect or modify a local
  environment before deciding whether the request is solvable.

## Action Contract

All environments share the same decision semantics:

- `ANSWER`: the agent commits to a final answer or completed task.
- `ABSTAIN`: the agent stops because the task should not be answered or cannot
  be completed from the available information.
- Environment actions: the agent gathers evidence before making a final
  decision. Examples include `SEARCH` in Q&A, browser actions in WebShop, and
  shell commands in TerminalBench.

The Q&A release uses strict JSON actions for Wikimedia episodes:

```json
{"action": "ANSWER", "answer": "..."}
{"action": "ABSTAIN", "reason": "..."}
{"action": "SEARCH", "search_query": "..."}
```

## Scenario Families

The release reports five abstention scenarios:

- Answer Unknown
- False Premise
- Subjective
- Underspecified Context
- Underspecified Intent

Some source datasets map entirely to one scenario. Mixed datasets, such as
KUQ, CoCoNot, and UMWP in the Q&A environment, are split by per-example
metadata before aggregation.

## Reproducibility Boundary

This repository is a protocol and code release, not a full experiment dump.
Large raw assets, indexes, model outputs, remote logs, and debug traces are
excluded. Users rebuild or download those assets using the documented links and
commands.
