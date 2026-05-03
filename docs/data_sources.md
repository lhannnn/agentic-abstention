# Data Sources

The repository does not vendor raw datasets or large indexes. Use the links and
instructions below to reconstruct each environment.

## Web

The Web environment depends on the upstream WebShop codebase and raw product
search assets. See `web/download/README.md` for the expected files and
directory layout.

## Q&A

The Q&A environment uses a filtered active roster derived from AbstentionBench
and related public datasets:

- ALCUNA
- BBQ
- BigBench Disambiguate
- BigBench Known Unknowns
- CoCoNot
- FalseQA
- GPQA
- GSM8K
- KUQ
- MediQ
- MMLU Math
- MoralChoice
- QAQA
- SituatedQA
- UMWP
- WorldSense

Dataset loaders download from their upstream HuggingFace or project sources at
runtime. Some sources, such as GPQA, may require accepting upstream terms and
setting `HF_TOKEN`.

The Wikimedia search environment requires:

- an English Wikimedia XML dump, such as `enwiki-20260101-pages-articles.xml.bz2`,
- FlashRAG preprocessing utilities,
- a local dense retrieval index built with `qa/scripts/build_wikimedia_index.py`.

## Terminal

The Terminal environment builds on TerminalBench and Harbor. The release tracks
rewrite metadata, prompt templates, configs, and validation scripts, but not
materialized task directories or job outputs. See `terminal/download/README.md`
and `terminal/README.md`.
