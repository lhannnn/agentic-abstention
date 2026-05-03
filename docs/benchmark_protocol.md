# Benchmark Protocol

Agentic abstention tests whether an agent can decide when not to answer while
operating in an interactive environment.

## Environments

- Web: WebShop navigation with rewritten goals and missing-target cases.
- Q&A: AbstentionBench-style questions with a Wikimedia `SEARCH` tool.
- Terminal: TerminalBench tasks where the agent must inspect or modify a local
  environment before deciding whether the request is solvable.

## Environment-Specific Interfaces

The concrete action interface is not shared across environments. Q&A exposes a
Wikimedia search tool, WebShop exposes browser/navigation actions, and
TerminalBench exposes command-line interaction. The common evaluation target is
the decision point: whether the agent should keep acting, produce a final
answer/task completion, or abstain.

Environment-specific README files and configs document the exact action format
used by each benchmark environment.

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
