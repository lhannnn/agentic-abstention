# Agentic Abstention

Agentic Abstention benchmarks whether interactive agents recognize when a task
is unresolvable and abstain before taking unsupported actions.

[Paper](https://arxiv.org/abs/2606.28733) · [Project page](https://lhannnn.github.io/agentic-abstention/)

## Environments

- [Web](web/README.md): WebShop navigation with rewritten goals and missing-target cases.
- [Q&A](qa/README.md): AbstentionBench-style questions with multi-turn Wikimedia search.
- [Terminal](terminal/README.md): TerminalBench tasks with immediate and delayed abstention points.

## Quick start

```bash
git clone https://github.com/lhannnn/agentic-abstention.git
cd agentic-abstention
```

Follow the Web, Q&A, or Terminal guide for environment-specific setup, data
preparation, and evaluation commands.

## Evaluation

The benchmark measures whether an agent continues, completes the task, or
abstains at the correct decision point. Reported metrics include Timely Recall,
Overall Recall, SPL, and pass@k.

See the [benchmark protocol](docs/benchmark_protocol.md) and
[metric definitions](docs/metrics.md).

## Data

Large datasets, retrieval indexes, and generated outputs are prepared
separately. See [data sources](docs/data_sources.md) and the download guide for
each environment.
