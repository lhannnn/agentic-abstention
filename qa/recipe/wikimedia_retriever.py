"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from recipe.wikimedia_indexing import E5TextEncoder, WikimediaIndexManifest

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class RetrievedDocument(BaseModel):
    rank: int
    title: str
    doc_id: str
    chunk_id: str
    text: str
    score: float


class WikimediaDenseRetriever:
    def __init__(
        self,
        index_root: str,
        encoder_name: str = "intfloat/e5-base-v2",
        top_k: int = 3,
        device: Optional[str] = None,
        query_encoder: Optional[E5TextEncoder] = None,
    ):
        self.index_root = Path(index_root)
        manifest_path = self.index_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Wikimedia index manifest not found at {manifest_path}. "
                "Build the index before running the Wikimedia multi-turn experiment."
            )

        self.manifest = WikimediaIndexManifest.model_validate_json(
            manifest_path.read_text()
        )
        if self.manifest.encoder_name != encoder_name:
            raise ValueError(
                "Retriever encoder_name does not match index manifest: "
                f"{encoder_name} != {self.manifest.encoder_name}"
            )

        self.top_k = top_k
        self.device = device
        self.query_encoder = query_encoder or E5TextEncoder(
            model_name=encoder_name,
            device=device,
        )
        self.docstore = self._load_docstore(self.index_root / self.manifest.docstore_path)
        self.embeddings = np.load(
            self.index_root / self.manifest.embeddings_path,
            mmap_mode="r",
        )
        if self.embeddings.shape[0] != len(self.docstore):
            raise ValueError(
                "Wikimedia docstore row count does not match embedding matrix rows."
            )
        self.faiss_index = self._load_faiss_index()

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDocument]:
        top_k = top_k or self.top_k
        query_embedding = self.query_encoder.encode_queries([query])[0]
        query_embedding = query_embedding.astype(np.float32, copy=False)

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_embedding[None, :], top_k)
            flat_scores = scores[0].tolist()
            flat_indices = indices[0].tolist()
        else:
            scores = np.asarray(self.embeddings @ query_embedding)
            if top_k >= len(scores):
                flat_indices = np.argsort(-scores).tolist()
            else:
                candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
                flat_indices = candidate_indices[np.argsort(-scores[candidate_indices])].tolist()
            flat_scores = [float(scores[index]) for index in flat_indices]

        retrieved_docs = []
        for rank, (score, index) in enumerate(zip(flat_scores, flat_indices), start=1):
            if index < 0:
                continue
            doc = self.docstore[index]
            retrieved_docs.append(
                RetrievedDocument(
                    rank=rank,
                    title=str(doc.get("title", "")),
                    doc_id=str(doc["doc_id"]),
                    chunk_id=str(doc["chunk_id"]),
                    text=str(doc["text"]),
                    score=float(score),
                )
            )
        return retrieved_docs

    def _load_docstore(self, docstore_path: Path) -> list[dict]:
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"Wikimedia docstore not found at {docstore_path}."
            )
        docstore = []
        with docstore_path.open("r") as source_file:
            for line in source_file:
                if not line.strip():
                    continue
                docstore.append(json.loads(line))
        return docstore

    def _load_faiss_index(self):
        if faiss is None:
            return None
        if not self.manifest.faiss_index_path:
            return None
        faiss_index_path = self.index_root / self.manifest.faiss_index_path
        if not faiss_index_path.exists():
            return None
        return faiss.read_index(str(faiss_index_path))
