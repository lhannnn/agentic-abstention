# Q&A Data Downloads

This release does not include raw datasets or retrieval indexes.

## Datasets

Dataset loaders in `recipe/abstention_datasets/` download public sources when
they are instantiated. Keep HuggingFace caches outside the repository. For
gated datasets, accept the upstream terms and set:

```bash
export HF_TOKEN=...
```

The active roster is listed in `../README.md`.

## Wikimedia Dump

Download an English Wikimedia dump from Wikimedia Downloads, for example:

```text
enwiki-20260101-pages-articles.xml.bz2
```

The dump is not tracked by git.

## Retrieval Index

Build the retrieval index with:

```bash
PYTHONPATH=. python scripts/build_wikimedia_index.py \
  --dump-path /path/to/enwiki-20260101-pages-articles.xml.bz2 \
  --flashrag-repo-path /path/to/FlashRAG \
  --output-root retrieval_indexes/wikimedia/enwiki-20260101_100w_intfloat_e5-base-v2 \
  --dump-version enwiki-20260101 \
  --chunk-words 100 \
  --encoder-name intfloat/e5-base-v2
```

The generated `retrieval_indexes/` directory can be large and must remain
outside git.
