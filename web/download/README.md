# Raw asset download guide

This repository does not vendor the raw WebShop data or retrieval resources.

You also need an upstream WebShop checkout at:

- `external/WebShop/`

You need these files:

- `external/WebShop/data/items_shuffle.json`
- `external/WebShop/data/items_ins_v2.json`
- `external/WebShop/data/items_human_ins.json`
- `external/WebShop/search_engine/resources/documents.jsonl`

Recommended source of truth:
- the upstream WebShop repository: https://github.com/princeton-nlp/WebShop
- the corresponding raw data assets or your own mirrored copies of the four required files

After downloading, place the files under the exact paths above. The scripts in
`scripts/` are written to look there first.

If you want to automate the fetch step, use `fetch_webshop_assets.sh` with your
own URLs or local mirrors.
