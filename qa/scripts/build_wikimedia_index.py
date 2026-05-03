#!/usr/bin/env python
"""
Build a Wikimedia dense retrieval index for AbstentionBench.

This script expects either:
1. A FlashRAG-processed JSONL corpus, or
2. A Wikipedia dump path together with a local FlashRAG checkout so the dump can
   be preprocessed first.

The resulting artifacts are stored under an index root directory and include:
- manifest.json
- docstore.jsonl
- embeddings.npy
- dense.index (when faiss is available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recipe.wikimedia_indexing import (
    build_chunked_docstore,
    encode_docstore_to_index,
    run_flashrag_preprocess,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=Path, default=None)
    parser.add_argument("--processed-corpus-path", type=Path, default=None)
    parser.add_argument("--flashrag-repo-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dump-version", type=str, default="enwiki-20260101")
    parser.add_argument("--chunk-words", type=int, default=100)
    parser.add_argument("--encoder-name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--top-k-default", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--disable-faiss",
        action="store_true",
        help="Do not build a faiss index even if faiss is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    processed_corpus_path = args.processed_corpus_path
    if processed_corpus_path is None:
        if args.dump_path is None or args.flashrag_repo_path is None:
            raise ValueError(
                "Provide either --processed-corpus-path or both --dump-path and "
                "--flashrag-repo-path."
            )
        processed_corpus_path = output_root / "flashrag_processed_corpus.jsonl"
        run_flashrag_preprocess(
            dump_path=args.dump_path,
            save_path=processed_corpus_path,
            flashrag_repo_path=args.flashrag_repo_path,
            num_workers=args.num_workers,
        )

    docstore_path = output_root / "docstore.jsonl"
    num_chunks = build_chunked_docstore(
        processed_corpus_path=processed_corpus_path,
        docstore_path=docstore_path,
        chunk_words=args.chunk_words,
    )
    manifest = encode_docstore_to_index(
        docstore_path=docstore_path,
        output_root=output_root,
        dump_version=args.dump_version,
        chunk_words=args.chunk_words,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        build_faiss_index=not args.disable_faiss,
        top_k_default=args.top_k_default,
        device=args.device,
    )
    print(
        "Built Wikimedia index:",
        {
            "output_root": str(output_root),
            "num_chunks": num_chunks,
            "manifest": manifest.model_dump(mode="json"),
        },
    )


if __name__ == "__main__":
    main()
