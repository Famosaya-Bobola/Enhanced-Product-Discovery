"""
Load tuned BM25 + dense (SBERT or E5) + hybrid RRF for online search.

Expects `benchmark/retrieval_config.json` from the tuning notebook and
`data/products.csv` (indices are rebuilt from this file).
Optional: `artifacts/sbert_embeddings.npy` or `e5_embeddings.npy` when row count
matches the preprocessed dataframe — skips a long corpus encode at startup.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from flask_product_search.text_utils import clean_lexical_text, preprocess_products_df, to_text


class ProductRetrievalEngine:
    def __init__(
        self,
        products_csv: str | Path,
        retrieval_config_path: str | Path,
        artifacts_dir: str | Path | None = None,
        encode_batch_size: int = 64,
    ) -> None:
        self.products_csv = Path(products_csv)
        self.retrieval_config_path = Path(retrieval_config_path)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.encode_batch_size = encode_batch_size

        self.df: pd.DataFrame | None = None
        self.idx_to_doc_id: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.dense_model: SentenceTransformer | None = None
        self.dense_family: str | None = None
        self.faiss_index: faiss.Index | None = None
        self.config: dict[str, Any] = {}
        self.rrf_k0: int = 60
        self.rrf_pool_k: int = 100

    def load(self) -> None:
        with open(self.retrieval_config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        tuning = self.config["tuning"]
        self.rrf_k0 = int(tuning["hybrid_rrf"]["k0"])
        self.rrf_pool_k = int(tuning["hybrid_rrf"]["pool_k"])
        bm25_params = tuning["bm25"]
        hybrid_dense = tuning["hybrid_dense_leg"]
        self.dense_family = hybrid_dense["family"].lower()
        dense_name = hybrid_dense["model_name"]

        raw = pd.read_csv(self.products_csv)
        self.df = preprocess_products_df(raw)
        self.idx_to_doc_id = self.df["doc_id"].to_numpy()

        corpus_tokens = [doc.split() for doc in self.df["lexical_text"].tolist()]
        self.bm25 = BM25Okapi(
            corpus_tokens,
            k1=bm25_params["k1"],
            b=bm25_params["b"],
            epsilon=bm25_params["epsilon"],
        )

        self.dense_model = SentenceTransformer(dense_name)
        texts = self.df["semantic_text"].tolist()
        emb = self._load_or_encode_embeddings(texts)
        emb = np.asarray(emb, dtype=np.float32)
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        self.faiss_index = index

    def _load_or_encode_embeddings(self, texts: list[str]) -> np.ndarray:
        if self.artifacts_dir and self.artifacts_dir.is_dir():
            stem = (
                "sbert_embeddings.npy"
                if self.dense_family == "sbert"
                else "e5_embeddings.npy"
            )
            path = self.artifacts_dir / stem
            if path.is_file():
                loaded = np.load(path)
                if loaded.shape[0] == len(texts):
                    return loaded

        assert self.dense_model is not None
        if self.dense_family == "e5":
            passages = ["passage: " + t for t in texts]
            emb = self.dense_model.encode(
                passages,
                show_progress_bar=False,
                batch_size=self.encode_batch_size,
            )
        else:
            emb = self.dense_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=self.encode_batch_size,
            )
        return np.asarray(emb, dtype=np.float32)

    def _bm25_ranked(self, query: str, top_k: int) -> list[tuple[str, float]]:
        assert self.bm25 is not None and self.idx_to_doc_id is not None
        tokens = clean_lexical_text(query).split()
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        k = min(top_k, len(scores))
        if k == 0:
            return []
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(str(self.idx_to_doc_id[i]), float(scores[i])) for i in idx]

    def _dense_ranked(self, query: str, top_k: int) -> list[tuple[str, float]]:
        assert self.dense_model is not None and self.faiss_index is not None
        assert self.idx_to_doc_id is not None
        if self.dense_family == "e5":
            q = np.asarray(
                self.dense_model.encode(
                    ["query: " + to_text(query)],
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        else:
            q = np.asarray(
                self.dense_model.encode(
                    [to_text(query)],
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        distances, indices = self.faiss_index.search(q, top_k)
        out: list[tuple[str, float]] = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1:
                out.append((str(self.idx_to_doc_id[i]), float(-d)))
        return out

    def search_hybrid(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        pool = self.rrf_pool_k
        k0 = self.rrf_k0
        bm = self._bm25_ranked(query, top_k=pool)
        dn = self._dense_ranked(query, top_k=pool)
        scores: dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm, start=1):
            scores[doc_id] += 1.0 / (k0 + rank)
        for rank, (doc_id, _) in enumerate(dn, start=1):
            scores[doc_id] += 1.0 / (k0 + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return self._bm25_ranked(query, top_k)

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return self._dense_ranked(query, top_k)

    def results_payload(
        self, ranked: list[tuple[str, float]], fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        assert self.df is not None
        if fields is None:
            fields = [
                "doc_id",
                "title",
                "main_category",
                "price",
                "average_rating",
                "description",
            ]
        order = {d: i for i, (d, _) in enumerate(ranked)}
        out: list[dict[str, Any]] = []
        for doc_id, score in sorted(ranked, key=lambda x: order[x[0]]):
            rows = self.df[self.df["doc_id"].astype(str) == str(doc_id)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            item: dict[str, Any] = {"score": float(score)}
            for f in fields:
                if f not in row.index:
                    continue
                v = row[f]
                if pd.isna(v):
                    item[f] = None
                elif isinstance(v, (np.integer, np.floating)):
                    item[f] = v.item()
                else:
                    item[f] = v
            out.append(item)
        return out


_engine: ProductRetrievalEngine | None = None


def get_engine() -> ProductRetrievalEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized; call init_engine() at app startup.")
    return _engine


def init_engine(
    products_csv: str | Path | None = None,
    retrieval_config_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> ProductRetrievalEngine:
    global _engine
    root = Path(__file__).resolve().parents[1]
    products_csv = products_csv or os.environ.get("PRODUCTS_CSV", str(root / "data" / "products.csv"))
    retrieval_config_path = retrieval_config_path or os.environ.get(
        "RETRIEVAL_CONFIG", str(root / "benchmark" / "retrieval_config.json")
    )
    artifacts_dir = artifacts_dir or os.environ.get("ARTIFACTS_DIR", str(root / "artifacts"))

    _engine = ProductRetrievalEngine(
        products_csv=products_csv,
        retrieval_config_path=retrieval_config_path,
        artifacts_dir=artifacts_dir if Path(artifacts_dir).is_dir() else None,
    )
    _engine.load()
    return _engine
