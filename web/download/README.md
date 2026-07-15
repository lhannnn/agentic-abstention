# WebShop Assets

Run the upstream WebShop
[full-data setup](https://github.com/princeton-nlp/WebShop#-setup) from
`external/WebShop/` to download the data and create the search resources.

The evaluator expects:

```text
external/WebShop/data/items_shuffle.json
external/WebShop/data/items_ins_v2.json
external/WebShop/data/items_human_ins.json
external/WebShop/search_engine/resources/documents.jsonl
external/WebShop/search_engine/indexes/
```

The upstream setup builds a Lucene 8 index with Pyserini 0.17. After installing
this repository's `pyserini>=0.40` requirement, install JDK 21, point
`JAVA_HOME` to it, and rebuild the full index with the same Pyserini version
used for the pruned index:

```bash
cd external/WebShop/search_engine
mv indexes indexes_pyserini017
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources \
  --index indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
cd ../../..
```

See the [Pyserini installation requirements](https://github.com/castorini/pyserini#-installation)
for JDK compatibility.
