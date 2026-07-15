# Q&A Agentic Abstention

Q&A benchmark implementation for agentic abstention with multi-turn Wikimedia search.

## Active Dataset Roster

The Q&A benchmark uses 16 active datasets:

```text
alcuna, bbq, big_bench_disambiguate, big_bench_known_unknowns,
coconot, falseqa, gpqa, gsm8k, kuq, mediq, mmlu_math,
moralchoice, qaqa, situated_qa, umwp, worldsense
```

The roster spans the five Q&A
[scenario families](../docs/benchmark_protocol.md#scenario-families).

## Setup

```bash
conda env create -f environment.yml
conda activate abstention-bench
pip install -e .
```

Copy `.env.example` to `.env` or export the variables directly:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
export TOGETHER_API_KEY=...
export HF_TOKEN=...
```

Only set keys for the provider and gated datasets you actually use.

## Build the Wikimedia Index

Prepare the Wikimedia dump and retrieval index as described in the [data guide](download/README.md).

## Run a Smoke Evaluation

OpenRouter example:

```bash
python main.py -m \
  mode=local \
  dataset=alcuna \
  model=wikimedia_multiturn_search_openrouter \
  module.openrouter_model_name=meta-llama/llama-3.3-70b-instruct \
  common_dir=$(pwd) \
  +experiment=wikimedia_multiturn_search_openrouter \
  dataset_indices_subset=[0] \
  dataset_indices_path=null
```

OpenAI example:

```bash
python main.py -m \
  mode=local \
  dataset=alcuna \
  model=wikimedia_multiturn_search_openai_gpt54mini \
  common_dir=$(pwd) \
  +experiment=wikimedia_multiturn_search_openai \
  dataset_indices_subset=[0] \
  dataset_indices_path=null
```

Together example:

```bash
python main.py -m \
  mode=local \
  dataset=alcuna \
  model=wikimedia_multiturn_search_together \
  common_dir=$(pwd) \
  +experiment=wikimedia_multiturn_search \
  dataset_indices_subset=[0] \
  dataset_indices_path=null
```

## Outputs

Each run writes results under:

```text
results/<sweep_folder>/<Dataset>_<Model>/<timestamp>/
```

The core Q&A artifacts are:

- `InferencePipeline.json`
- `DirectAbstention.json`
- `WikimediaJsonAbstentionDetector.json`
- `WikimediaEpisodeMetrics.json`
- `WikimediaEpisodeMetricsSummary.json`
- `MultiTurnEpisodeTraces.jsonl`

## Metrics

The evaluator reports Timely Recall, Overall Recall, SPL, pass@0 through
pass@10, average search calls, and forced-stop rate. See `../docs/metrics.md`.
