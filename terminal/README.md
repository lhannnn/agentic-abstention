# TerminalBench Agentic Abstention

TerminalBench tasks for evaluating abstention in terminal agents.

Task families:
- `immediate`: rewritten instructions that are unresolvable from the prompt alone.
- `delayed`: tasks that look solvable initially and become unresolvable after local environment interaction.

## Contents

```text
data/immediate/   immediate rewrite data and category definitions
data/delayed/     delayed manifests, accepted subset, specs, and review policy
scripts/          build, materialize, validate, run, and analyze utilities
configs/          Harbor config templates
prompts/          rewrite prompts for immediate tasks
```

## Setup

```bash
cd terminal
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Install Harbor separately. If needed, apply the abstention overlay:

```bash
python scripts/install_harbor_abstain_overlay.py --venv /path/to/harbor/.venv
```

## Validate Metadata

```bash
python scripts/validate_immediate_release.py
python scripts/validate_delayed_case_pack.py
```

## Immediate Tasks

```bash
python scripts/materialize_harbor_instruction_level_abstention_dataset.py \
  --input data/immediate/immediate_rewrites_267.jsonl \
  --cache-root ~/.cache/harbor/tasks \
  --output datasets/terminalbench_instruction_level_abstention_267 \
  --allow-partial
```

## Delayed Tasks

Core files:

```text
data/delayed/manifest.jsonl
data/delayed/manifest.accepted_delayed_21.jsonl
data/delayed/specs/
data/delayed/reviews/rewrite_only_policy.json
data/delayed/reviews/rewrite_only_consensus.json
```

After materializing delayed task directories:

```bash
python scripts/validate_delayed_case_pack.py --require-task-dirs
```

## Run

Update paths in `configs/` if your benchmark root differs from `/workspace/terminalbench`.

```bash
harbor run -y --env-file .env.openai \
  --config configs/codex_gpt54mini_medium_instruction_level_abstention_256_p4.json

harbor run -y --env-file .env.openrouter \
  --config configs/gpt54mini_medium_openrouter_terminalbench_delayed_abstention_21_p4.json
```

## Analyze

```bash
python scripts/analyze_paired_abstention_job.py \
  --job-dir <job-dir> \
  --manifest <immediate-manifest.jsonl>

python scripts/analyze_delayed_abstention_job.py \
  --job-dir <job-dir> \
  --manifest data/delayed/manifest.accepted_delayed_21.jsonl
```
