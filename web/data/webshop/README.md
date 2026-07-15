# WebShop Benchmark Artifacts

## Benchmark files

| File | Rewrite model | Use |
| --- | --- | --- |
| `webshop_instruction_set_1000.jsonl` | GPT-5-mini | Alternate released rewrite set |
| `webshop_instruction_set_1000_gpt54_high.jsonl` | GPT-5.4 (high reasoning) | Default evaluation input |

Both files contain 500 answerable originals, 249 instruction rewrites, and 251
missing-target tasks. The four manifests under `manifests/` partition the
default input into homogeneous index ranges for parallel evaluation.

Component rewrite files are versioned for the GPT-5-mini set; GPT-5.4 rewrites
are embedded in the default combined dataset.

## Lineage

| Stage | Artifacts | Scripts |
| --- | --- | --- |
| Source selection | `webshop_real_goal_manifest.jsonl`, `webshop_source500_manifest.jsonl` | `webshop_abstain_export_real_goal_manifest.py`, `webshop_instruction_set_export_source500.py` |
| Rewrite generation | `webshop_rewrite249_plan.json`, `rewrites_*_gpt5mini_83.jsonl`, `webshop_rewrite249_gpt5mini.jsonl` | `webshop_instruction_set_plan_rewrite249.py`, `webshop_instruction_set_rewrite_*.py`, `webshop_instruction_set_merge_rewrite249.py` |
| Missing-target construction | `webshop_missing_target251_manifest.jsonl`, `pruned_missing_target_251/removed_target_asins_251.json` | `webshop_instruction_set_build_missing_target251.py`, `webshop_instruction_set_build_pruned_env.py` |
| Assembly and sharding | `webshop_instruction_set_1000*.jsonl`, `manifests/` | `webshop_instruction_set_merge_1000.py`, `webshop_instruction_set_build_4way_manifests.py` |

Run scripts from `web/` and use `--help` to set explicit input and output paths.
To regenerate the GPT-5.4 set, pass `--model gpt-5.4 --reasoning-effort high`
and explicit output paths to each category rewrite script. When sharding any
rebuilt dataset, pass it explicitly to `webshop_instruction_set_build_4way_manifests.py`
with `--input`; that script defaults to the released GPT-5.4 file.

Rewrite scripts read API keys from the shell environment. See the [WebShop
guide](../../README.md) for evaluation and the [asset guide](../../download/README.md)
for upstream data.
