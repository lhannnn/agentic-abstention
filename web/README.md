# WebShop Agentic Abstention Release

This repository is a cleaned WebShop-only release for the `agentic abstention`
part of the project. It does not mirror the full ACE workspace. It only
contains the core assets needed to:

- rebuild the rewritten WebShop instruction set,
- generate the missing-target split and pruned environment,
- run the 1000-task multi-turn evaluation, and
- merge shard results into benchmark summaries.

## Included

- core instruction-building and evaluation scripts in `scripts/`
- lightweight, repo-tracked derived artifacts in `data/webshop/`
- evaluation protocol in `data/webshop/eval_metrics_1000.md`
- download guidance in `download/README.md`

## Not included

- raw WebShop product files
- raw search resources and Lucene indexes
- local experiment outputs, plots, and debug traces
- AutoDL cluster configs and remote recovery scripts
- local environments or caches

## Raw assets

Check out the upstream WebShop repository under `external/WebShop/`, then place the raw assets under:

```text
external/WebShop/
  data/
    items_shuffle.json
    items_ins_v2.json
    items_human_ins.json
  search_engine/
    resources/
      documents.jsonl
```

See `download/README.md` for the expected files and source links you should provide.

## Minimal setup

1. Create a Python environment and install `requirements.txt`.
2. Add your API keys to a local `.env` file based on `.env.example`.
3. Clone the upstream WebShop codebase into `external/WebShop/`.
4. Download the raw WebShop assets into `external/WebShop/`.
5. Install the upstream WebShop runtime dependencies needed by `web_agent_site`.

## Runtime assumptions

- `scripts/webshop_instruction_set_evaluate_multiturn.py` imports `web_agent_site` from the upstream WebShop checkout.
- `scripts/webshop_instruction_set_build_pruned_env.py` uses `pyserini` and requires a working Java installation.
- The default paths are repo-local and assume `external/WebShop/` exists.

## Shortest reproduction path

1. Export the real-goal manifest:
   ```bash
   python scripts/webshop_abstain_export_real_goal_manifest.py
   ```
2. Export `source500` and build the rewrite plan:
   ```bash
   python scripts/webshop_instruction_set_export_source500.py
   python scripts/webshop_instruction_set_plan_rewrite249.py
   ```
3. Regenerate the three abstention rewrite files:
   ```bash
   python scripts/webshop_instruction_set_rewrite_subjective_gpt5mini.py
   python scripts/webshop_instruction_set_rewrite_underspecified_intent_gpt5mini.py
   python scripts/webshop_instruction_set_rewrite_false_premises_gpt5mini.py
   python scripts/webshop_instruction_set_merge_rewrite249.py
   ```
4. Build the missing-target split and pruned environment:
   ```bash
   python scripts/webshop_instruction_set_build_missing_target251.py
   python scripts/webshop_instruction_set_build_pruned_env.py
   python scripts/webshop_instruction_set_merge_1000.py
   python scripts/webshop_instruction_set_build_4way_manifests.py
   ```
5. Run at least one smoke evaluation:
   ```bash
   python scripts/webshop_instruction_set_evaluate_multiturn.py \
     --input data/webshop/manifests/webshop_instruction_set_1000_gpt54_high__shard00_0_249.jsonl \
     --count 1
   ```
6. Merge shard outputs after a full run:
   ```bash
   python scripts/webshop_instruction_set_merge_results.py \
     --input-dir <results-dir> \
     --pattern 'results_shard*.jsonl' \
     --merged-output <merged-results.jsonl> \
     --summary-output <merged-summary.json> \
     --diagnostic-output <diagnostic-summary.json> \
     --model <model-id> \
     --reasoning-effort <effort> \
     --input-path data/webshop/webshop_instruction_set_1000_gpt54_high.jsonl
   ```

## Notes

- The repo tracks the key derived manifests and rewrite outputs so users do not need to regenerate them from scratch unless they want to audit the pipeline.
- The raw WebShop files and any Lucene index outputs stay out of git.
- This release is scoped to the WebShop environment only; Q&A and terminal are intentionally excluded.
