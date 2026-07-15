# WebShop Agentic Abstention

WebShop data and evaluation tools for the 1,000-task agentic abstention benchmark.

## Benchmark

| Dataset indices | Variant | Environment | Abstention warranted |
| --- | --- | --- | --- |
| 0–499 | `original` | full | No |
| 500–748 | `rewrite` | full | Step 1 |
| 749–999 | `missing_target` | pruned | Step 2 |

The evaluator defaults to
[`data/webshop/webshop_instruction_set_1000_gpt54_high.jsonl`](data/webshop/webshop_instruction_set_1000_gpt54_high.jsonl).
See the [artifact guide](data/webshop/README.md) for dataset lineage and the
[evaluation protocol](data/webshop/eval_metrics_1000.md) for metric definitions.

## Setup

Create and activate the Python environment described by upstream
[WebShop](https://github.com/princeton-nlp/WebShop). Run its full-data setup to
download the source assets, then install this repository's requirements:

```bash
cd web
git clone https://github.com/princeton-nlp/WebShop external/WebShop
cd external/WebShop
./setup.sh -d all
cd ../..
python -m pip install -r requirements.txt
cp .env.example .env
```

The upstream setup installs JDK 11, while this repository's `pyserini>=0.40`
requires JDK 21. Install JDK 21, point `JAVA_HOME` to it, and rebuild the full
Lucene index as described in the [asset guide](download/README.md). Set
`OPENAI_API_KEY` in `.env` or the shell.

## Prepare the environments

Build the pruned environment used by missing-target tasks:

```bash
python scripts/webshop_instruction_set_build_pruned_env.py
```

## Evaluate

Run the complete benchmark:

```bash
python scripts/webshop_instruction_set_evaluate_multiturn.py \
  --results-output outputs/webshop/results.jsonl \
  --summary-output outputs/webshop/summary.json
```

Use `--count 1` for a basic smoke test and `--resume` to continue an interrupted
run. Four homogeneous manifests are available in `data/webshop/manifests/` for
parallel evaluation.

## Dataset construction

The versioned manifests can be evaluated directly. Dataset lineage and
construction scripts are documented in the [artifact guide](data/webshop/README.md).
