# Q&A Data Downloads

## Datasets

Dataset loaders in `recipe/abstention_datasets/` fetch upstream sources on first
use. Gated datasets require accepted upstream terms and `HF_TOKEN`:

```bash
export HF_TOKEN=...
```

The active roster is listed in the [Q&A guide](../README.md#active-dataset-roster).

### Local inputs

Place these dataset-specific inputs under `qa/`:

- [`data/kuq/new-category-mapping.csv`](https://github.com/facebookresearch/AbstentionBench/blob/main/data/kuq/new-category-mapping.csv)
- [`data/UMWP_indices_answerable.json`](https://github.com/facebookresearch/AbstentionBench/blob/main/data/UMWP_indices_answerable.json)
- `datasets/situated_qa/geo.jsonl`, exported from the `geo` test split of
  [`siyue/SituatedQA`](https://huggingface.co/datasets/siyue/SituatedQA)

## Wikimedia Dump

Download an English Wikimedia dump from
[Wikimedia Downloads](https://dumps.wikimedia.org/), for example:

```text
enwiki-20260101-pages-articles.xml.bz2
```

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
