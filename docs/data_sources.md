# Data Sources

External data and indexes are prepared separately for each environment.

## Web

The Web environment uses the upstream
[WebShop](https://github.com/princeton-nlp/WebShop) codebase and product-search
assets. See the [Web asset guide](../web/download/README.md).

## Q&A

The active roster is listed in the
[Q&A guide](../qa/README.md#active-dataset-roster). Most loaders retrieve data
from Hugging Face or project repositories at runtime; gated sources require
upstream access and `HF_TOKEN`. SituatedQA and dataset-specific metadata are
local inputs documented in the [Q&A data guide](../qa/download/README.md).

Wikimedia search requires an English XML dump,
[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), and a local dense retrieval
index.

## Terminal

The Terminal environment builds on
[TerminalBench](https://github.com/harbor-framework/terminal-bench) and
[Harbor](https://github.com/harbor-framework/harbor). Materialize runnable tasks
from an upstream task cache or mirror using the tracked rewrite metadata and
specs. See the [Terminal asset guide](../terminal/download/README.md).
