# Agentic Abstention

This repository is a research artifact release for the agentic abstention benchmark. It
contains the code and lightweight artifacts needed to reproduce the benchmark
protocol across three agent environments:

- `web/`: WebShop-based web navigation tasks.
- `qa/`: Q&A tasks with Wikimedia multi-turn search.
- `terminal/`: TerminalBench immediate and delayed abstention tasks.

Raw datasets, search indexes, model outputs, debug traces, and cluster job
artifacts are intentionally not tracked. Each environment includes download or
materialization instructions for external assets.

## Repository Layout

```text
web/        WebShop instruction rewriting, missing-target construction, and evaluation
qa/         AbstentionBench-style Q&A datasets with Wikimedia SEARCH episodes
terminal/   TerminalBench task construction, Harbor configs, and analysis tools
docs/       Benchmark protocol, metric definitions, and data-source notes
```

## Quick Start

Clone the repository and enter the environment you want to reproduce:

```bash
git clone https://github.com/lhannnn/agentic-abstention.git
cd agentic-abstention
```

For WebShop:

```bash
cd web
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

For Q&A:

```bash
cd qa
conda env create -f environment.yml
conda activate abstention-bench
pip install -e .
```

For TerminalBench:

```bash
cd terminal
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

See the environment-specific README files for data downloads and run commands.

## Benchmark Protocol

Agentic abstention evaluates whether an agent knows when to stop and abstain
instead of continuing with unsupported actions. The concrete action interface is
environment-specific: Q&A uses Wikimedia search episodes, Web uses browser
navigation, and Terminal uses command-line interaction.

The main metrics are Timely Recall, Overall Recall, SPL, and pass@k. See
`docs/metrics.md` for the metric definitions and `docs/benchmark_protocol.md`
for the environment-level protocol summary.

## Data Policy

This repository tracks only small code, prompt, config, manifest, and test
files. It does not include:

- raw WebShop product/search assets,
- raw or materialized TerminalBench task directories,
- HuggingFace dataset caches,
- Wikimedia dumps or retrieval indexes,
- API keys or `.env` files,
- model outputs, logs, plots, and debug traces.

See `docs/data_sources.md` and each environment's `download/README.md`.
