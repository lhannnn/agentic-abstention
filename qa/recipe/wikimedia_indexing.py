"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from pydantic import BaseModel
from torch import Tensor

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "transformers is required for Wikimedia indexing. "
        "Install it in the abstentionbench environment."
    ) from exc


class WikimediaChunkRecord(BaseModel):
    id: str
    doc_id: str
    chunk_id: str
    title: str
    text: str
    contents: str


class WikimediaIndexManifest(BaseModel):
    dump_version: str
    chunk_words: int
    encoder_name: str
    created_at_utc: str
    num_chunks: int
    embedding_dim: int
    docstore_path: str
    embeddings_path: str
    faiss_index_path: Optional[str] = None
    top_k_default: int = 3


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    text: str


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode_prefixed_texts(["query: " + query for query in queries])

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode_prefixed_texts(["passage: " + passage for passage in passages])

    def _encode_prefixed_texts(self, texts: list[str]) -> np.ndarray:
        batch_dict = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)


def chunk_text_by_words(text: str, chunk_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[start_index : start_index + chunk_words])
        for start_index in range(0, len(words), chunk_words)
    ]


def _extract_source_document(record: dict, fallback_doc_id: int) -> SourceDocument:
    doc_id = str(record.get("doc_id") or record.get("id") or fallback_doc_id)
    title = (
        record.get("title")
        or record.get("doc_title")
        or record.get("name")
        or ""
    )
    text = record.get("text") or record.get("content") or record.get("contents") or ""
    text = str(text).strip()

    if not title and "\n" in text:
        first_line, rest = text.split("\n", 1)
        if 0 < len(first_line.split()) <= 30:
            title = first_line.strip()
            text = rest.strip()

    if not text:
        raise ValueError(f"Processed corpus record {fallback_doc_id} does not contain text.")

    return SourceDocument(
        doc_id=doc_id,
        title=str(title).strip(),
        text=text,
    )


def build_chunked_docstore(
    processed_corpus_path: str | Path,
    docstore_path: str | Path,
    chunk_words: int = 100,
) -> int:
    processed_corpus_path = Path(processed_corpus_path)
    docstore_path = Path(docstore_path)
    docstore_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_counter = 0
    with processed_corpus_path.open("r") as source_file, docstore_path.open("w") as target_file:
        for line_index, line in enumerate(source_file):
            if not line.strip():
                continue
            record = json.loads(line)
            source_document = _extract_source_document(record, fallback_doc_id=line_index)
            chunks = chunk_text_by_words(source_document.text, chunk_words=chunk_words)
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_id = f"{source_document.doc_id}:{chunk_index}"
                contents = (
                    f"{source_document.title}\n{chunk_text}"
                    if source_document.title
                    else chunk_text
                )
                chunk_record = WikimediaChunkRecord(
                    id=str(chunk_counter),
                    doc_id=source_document.doc_id,
                    chunk_id=chunk_id,
                    title=source_document.title,
                    text=chunk_text,
                    contents=contents,
                )
                target_file.write(chunk_record.model_dump_json())
                target_file.write("\n")
                chunk_counter += 1

    return chunk_counter


def count_jsonl_rows(jsonl_path: str | Path) -> int:
    with Path(jsonl_path).open("r") as source_file:
        return sum(1 for line in source_file if line.strip())


def _iter_chunk_records(docstore_path: str | Path) -> Iterable[WikimediaChunkRecord]:
    with Path(docstore_path).open("r") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            yield WikimediaChunkRecord.model_validate_json(line)


def encode_docstore_to_index(
    docstore_path: str | Path,
    output_root: str | Path,
    dump_version: str,
    chunk_words: int,
    encoder_name: str = "intfloat/e5-base-v2",
    batch_size: int = 128,
    build_faiss_index: bool = True,
    top_k_default: int = 3,
    device: Optional[str] = None,
) -> WikimediaIndexManifest:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    docstore_path = Path(docstore_path)
    num_chunks = count_jsonl_rows(docstore_path)
    encoder = E5TextEncoder(model_name=encoder_name, device=device)

    embeddings_path = output_root / "embeddings.npy"
    faiss_index_path = output_root / "dense.index"
    faiss_index = None
    memmap = None
    cursor = 0
    batch_records: list[WikimediaChunkRecord] = []

    for record in _iter_chunk_records(docstore_path):
        batch_records.append(record)
        if len(batch_records) < batch_size:
            continue
        cursor, memmap, faiss_index = _write_embedding_batch(
            batch_records=batch_records,
            encoder=encoder,
            embeddings_path=embeddings_path,
            num_chunks=num_chunks,
            cursor=cursor,
            memmap=memmap,
            faiss_index=faiss_index,
            build_faiss_index=build_faiss_index,
        )
        batch_records = []

    if batch_records:
        cursor, memmap, faiss_index = _write_embedding_batch(
            batch_records=batch_records,
            encoder=encoder,
            embeddings_path=embeddings_path,
            num_chunks=num_chunks,
            cursor=cursor,
            memmap=memmap,
            faiss_index=faiss_index,
            build_faiss_index=build_faiss_index,
        )

    if memmap is None:
        raise ValueError(f"No chunks were found in {docstore_path}.")
    memmap.flush()

    if faiss_index is not None:
        faiss.write_index(faiss_index, str(faiss_index_path))
        faiss_index_file = faiss_index_path.name
    else:
        faiss_index_file = None

    manifest = WikimediaIndexManifest(
        dump_version=dump_version,
        chunk_words=chunk_words,
        encoder_name=encoder_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        num_chunks=num_chunks,
        embedding_dim=int(memmap.shape[1]),
        docstore_path=docstore_path.name,
        embeddings_path=embeddings_path.name,
        faiss_index_path=faiss_index_file,
        top_k_default=top_k_default,
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _write_embedding_batch(
    batch_records: list[WikimediaChunkRecord],
    encoder: E5TextEncoder,
    embeddings_path: Path,
    num_chunks: int,
    cursor: int,
    memmap: Optional[np.memmap],
    faiss_index,
    build_faiss_index: bool,
):
    batch_texts = [record.contents for record in batch_records]
    batch_embeddings = encoder.encode_passages(batch_texts)
    if memmap is None:
        memmap = open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_chunks, batch_embeddings.shape[1]),
        )
    memmap[cursor : cursor + len(batch_records)] = batch_embeddings

    if build_faiss_index and faiss is not None:
        if faiss_index is None:
            faiss_index = faiss.IndexFlatIP(batch_embeddings.shape[1])
        faiss_index.add(batch_embeddings)

    cursor += len(batch_records)
    return cursor, memmap, faiss_index


def run_flashrag_preprocess(
    dump_path: str | Path,
    save_path: str | Path,
    flashrag_repo_path: str | Path,
    num_workers: int = 1,
) -> None:
    flashrag_repo_path = Path(flashrag_repo_path)
    preprocess_script = flashrag_repo_path / "scripts" / "preprocess_wiki.py"
    if not preprocess_script.exists():
        raise FileNotFoundError(
            f"Could not find FlashRAG preprocess script at {preprocess_script}."
        )

    # FlashRAG's documented workflow chunks by sentence. We use a very large sentence
    # window here to preserve article text, then re-chunk to fixed 100-word segments
    # inside AbstentionBench so runtime chunks are deterministic.
    command = [
        sys.executable,
        str(preprocess_script),
        "--dump_path",
        str(dump_path),
        "--save_path",
        str(save_path),
        "--chunk_by",
        "sentence",
        "--seg_size",
        "1000000",
        "--stride",
        "1000000",
        "--num_workers",
        str(num_workers),
    ]
    subprocess.run(command, check=True)
