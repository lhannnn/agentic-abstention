# WebShop release artifacts

This directory contains the lightweight, repo-tracked artifacts for the
WebShop `agentic abstention` release.

Included here:
- `webshop_real_goal_manifest.jsonl`
- `webshop_source500_manifest.jsonl`
- `webshop_rewrite249_plan.json`
- the three GPT-5-mini rewrite JSONL files
- `webshop_rewrite249_gpt5mini.jsonl`
- `webshop_missing_target251_manifest.jsonl`
- `webshop_instruction_set_1000.jsonl`
- `webshop_instruction_set_1000_gpt54_high.jsonl`
- homogeneous shard manifests in `manifests/`
- `eval_metrics_1000.md`
- `pruned_missing_target_251/removed_target_asins_251.json`

Not included here:
- raw WebShop product files
- raw search resources or Lucene indexes
- generated results, plots, or debug traces

Expected raw asset layout:

- `external/WebShop/data/items_shuffle.json`
- `external/WebShop/data/items_ins_v2.json`
- `external/WebShop/data/items_human_ins.json`
- `external/WebShop/search_engine/resources/documents.jsonl`

Typical workflow:

1. Export or regenerate the real-goal manifest:
   `python scripts/webshop_abstain_export_real_goal_manifest.py`
2. Export `source500`:
   `python scripts/webshop_instruction_set_export_source500.py`
3. Plan and generate the 249 rewrites:
   `python scripts/webshop_instruction_set_plan_rewrite249.py`
   `python scripts/webshop_instruction_set_rewrite_subjective_gpt5mini.py`
   `python scripts/webshop_instruction_set_rewrite_underspecified_intent_gpt5mini.py`
   `python scripts/webshop_instruction_set_rewrite_false_premises_gpt5mini.py`
4. Merge rewrites and build the missing-target split:
   `python scripts/webshop_instruction_set_merge_rewrite249.py`
   `python scripts/webshop_instruction_set_build_missing_target251.py`
5. Build the pruned environment and final 1000-task set:
   `python scripts/webshop_instruction_set_build_pruned_env.py`
   `python scripts/webshop_instruction_set_merge_1000.py`
6. Evaluate and merge shard outputs:
   `python scripts/webshop_instruction_set_evaluate_multiturn.py`
   `python scripts/webshop_instruction_set_merge_results.py`
