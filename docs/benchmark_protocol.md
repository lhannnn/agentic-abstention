# Benchmark Protocol

Agentic Abstention evaluates whether an interactive agent continues, completes,
or abstains at the appropriate decision point.

## Environments

| Environment | Interaction | Abstention timing | Guide |
| --- | --- | --- | --- |
| WebShop | Browser search and navigation | Step 1 for instruction rewrites; step 2 for missing-target tasks | [Web](../web/README.md) |
| Q&A | Wikimedia `SEARCH`, `ANSWER`, and `ABSTAIN` actions | Oracle step 1 for abstain-warranted questions | [Q&A](../qa/README.md) |
| TerminalBench | Shell interaction through Harbor | Turn 1 for immediate tasks; manifest-defined earliest turn and grace window for delayed tasks | [Terminal](../terminal/README.md) |

Answerable controls have no oracle abstention step. Recall, SPL, and pass@k are
computed only on abstain-warranted episodes; see [Metrics](metrics.md).

## Scenario families

Scenario families are environment-specific:

- **Web:** Subjective Preference, Underspecified Intent, False Premise or
  Contradiction, and Missing Target.
- **Terminal:** False Premise or Contradiction, Underspecified Intent, and
  Missing Prerequisite.
- **Q&A:** Answer Unknown, False Premise, Subjective, Underspecified Context,
  and Underspecified Intent.

Mixed Q&A datasets, including KUQ, CoCoNot, and UMWP, are partitioned by
per-example metadata before aggregation.
