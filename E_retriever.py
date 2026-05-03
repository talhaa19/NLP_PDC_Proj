"""
TF-IDF, BM25-lite, and GPU/CPU-safe Dense retrievers used by serial and parallel RAG.

Retriever indices are optionally persisted to disk with joblib so repeated
benchmark runs skip the fitting step.

DenseRetriever requires:
    pip install sentence-transformers torch faiss-cpu

Optional GPU FAISS:
    faiss-gpu is not easy on Windows, so this code safely falls back to CPU FAISS.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from A_config import get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corpus_fingerprint(corpus: list[dict]) -> str:
    """Compute a short hash of corpus doc IDs and text lengths."""
    digest = hashlib.md5(
        "".join(f"{d['doc_id']}:{len(d['text'])}" for d in corpus).encode()
    ).hexdigest()
    return digest[:8]


def _cache_path(cache_dir: Path, name: str, fingerprint: str) -> Path:
    """Return cache path for retriever artifacts."""
    return cache_dir / f"{name}_{fingerprint}.joblib"


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TfidfRetriever:
    """Scikit-learn TF-IDF retriever with optional disk caching."""

    def __init__(
        self,
        corpus: list[dict],
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        shard_delay_sec: float | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        cfg = get_config()["retrieval"]

        self.corpus = corpus
        self.shard_delay_sec = (
            shard_delay_sec
            if shard_delay_sec is not None
            else cfg["shard_delay_sec"]
        )

        stop_words = "english" if analyzer == "word" else None

        if cache_dir is None:
            cache_dir = Path(__file__).parent / cfg["cache_dir"]

        cache_dir = Path(cache_dir)
        name = f"tfidf_{analyzer}_{ngram_range[0]}_{ngram_range[1]}"
        fp = _corpus_fingerprint(corpus)
        cpath = _cache_path(cache_dir, name, fp)

        if cpath.exists():
            cached = joblib.load(cpath)
            self.vectorizer = cached["vectorizer"]
            self.matrix = cached["matrix"]
            print(f"[TfidfRetriever] Loaded cached TF-IDF index from {cpath}")
        else:
            self.vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                analyzer=analyzer,
                ngram_range=ngram_range,
            )
            self.matrix = self.vectorizer.fit_transform(
                [doc["text"] for doc in corpus]
            )

            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                },
                cpath,
            )
            print(f"[TfidfRetriever] Cached TF-IDF index to {cpath}")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search corpus using TF-IDF cosine-style scoring."""
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]

        if self.shard_delay_sec:
            time.sleep(self.shard_delay_sec)

        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()

        if len(scores) == 0:
            return []

        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.corpus[i],
                "score": float(scores[i]),
                "retriever": "tfidf",
            }
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# BM25-lite retriever
# ---------------------------------------------------------------------------

class KeywordBM25LiteRetriever:
    """Simplified BM25 retriever backed by an in-memory inverted index."""

    def __init__(
        self,
        corpus: list[dict],
        shard_delay_sec: float | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        cfg = get_config()["retrieval"]

        self.corpus = corpus
        self.shard_delay_sec = (
            shard_delay_sec
            if shard_delay_sec is not None
            else cfg["shard_delay_sec"]
        )

        self._k1: float = cfg["bm25"]["k1"]
        self._b: float = cfg["bm25"]["b"]

        if cache_dir is None:
            cache_dir = Path(__file__).parent / cfg["cache_dir"]

        cache_dir = Path(cache_dir)
        fp = _corpus_fingerprint(corpus)
        cpath = _cache_path(cache_dir, "bm25_lite", fp)

        if cpath.exists():
            cached = joblib.load(cpath)

            self.docs = cached["docs"]
            self.doc_lens = cached["doc_lens"]
            self.avgdl = cached["avgdl"]
            self.df = cached["df"]
            self.inverted_index = cached["inverted_index"]

            print(f"[BM25Lite] Loaded cached BM25 index from {cpath}")
        else:
            self.docs = [self._tokens(doc["text"]) for doc in corpus]
            self.doc_lens = [len(doc) for doc in self.docs]
            self.avgdl = sum(self.doc_lens) / max(1, len(self.docs))

            self.df: Counter[str] = Counter()
            self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

            for idx, doc_tokens in enumerate(self.docs):
                tf = Counter(doc_tokens)
                self.df.update(tf.keys())

                for term, freq in tf.items():
                    self.inverted_index[term].append((idx, freq))

            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "docs": self.docs,
                    "doc_lens": self.doc_lens,
                    "avgdl": self.avgdl,
                    "df": self.df,
                    "inverted_index": self.inverted_index,
                },
                cpath,
            )

            print(f"[BM25Lite] Cached BM25 index to {cpath}")

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric terms."""
        return re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search corpus using BM25 scoring."""
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]

        if self.shard_delay_sec:
            time.sleep(self.shard_delay_sec)

        q_terms = self._tokens(query)
        scores: dict[int, float] = defaultdict(float)

        total_docs = len(self.docs)
        k1 = self._k1
        b = self._b

        for term in q_terms:
            postings = self.inverted_index.get(term, [])

            if not postings:
                continue

            idf = log(
                1
                + (total_docs - self.df[term] + 0.5)
                / (self.df[term] + 0.5)
            )

            for idx, freq in postings:
                denom = freq + k1 * (
                    1 - b + b * self.doc_lens[idx] / max(1, self.avgdl)
                )
                scores[idx] += idf * freq * (k1 + 1) / denom

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        if len(ranked) < top_k:
            seen = {idx for idx, _ in ranked}

            for idx in range(total_docs):
                if idx not in seen:
                    ranked.append((idx, 0.0))

                if len(ranked) >= top_k:
                    break

        return [
            {
                **self.corpus[i],
                "score": float(score),
                "retriever": "bm25_lite",
            }
            for i, score in ranked
        ]


# ---------------------------------------------------------------------------
# Dense retriever with safe FAISS GPU fallback
# ---------------------------------------------------------------------------

def _build_faiss_index(dim: int) -> "faiss.Index":
    """
    Build FAISS inner-product index.

    Uses GPU only when:
    1. CUDA is available, and
    2. installed FAISS package supports GPU APIs.

    Otherwise falls back to CPU FAISS.
    """
    import faiss

    cpu_index = faiss.IndexFlatIP(dim)

    has_faiss_gpu = (
        hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "index_cpu_to_gpu")
    )

    if torch.cuda.is_available() and has_faiss_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

            print(
                f"[DenseRetriever] FAISS index moved to GPU 0 "
                f"({torch.cuda.get_device_name(0)})"
            )

            return gpu_index

        except Exception as e:
            print(
                "[DenseRetriever] Tried GPU FAISS but failed. "
                f"Falling back to CPU FAISS. Reason: {e}"
            )

    if torch.cuda.is_available() and not has_faiss_gpu:
        print(
            "[DenseRetriever] CUDA is available, but your installed FAISS "
            "does not support GPU. Using CPU FAISS index."
        )
    else:
        print("[DenseRetriever] CUDA not available — using CPU FAISS index.")

    return cpu_index


class DenseRetriever:
    """Dense retriever using sentence-transformers + FAISS."""

    def __init__(
        self,
        corpus: list[dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        shard_delay_sec: float | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        cfg = get_config()["retrieval"]

        self.corpus = corpus
        self.shard_delay_sec = (
            shard_delay_sec
            if shard_delay_sec is not None
            else cfg["shard_delay_sec"]
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.name = f"dense:{model_name.split('/')[-1]}"

        if cache_dir is None:
            cache_dir = Path(__file__).parent / cfg["cache_dir"]

        cache_dir = Path(cache_dir)

        safe_model = model_name.replace("/", "_").replace("-", "_")
        fp = _corpus_fingerprint(corpus)
        cpath = _cache_path(cache_dir, f"dense_{safe_model}", fp)

        print(f"[DenseRetriever] Loading model {model_name} on {device.upper()}...")
        self.model = SentenceTransformer(model_name, device=device)

        if cpath.exists():
            embeddings: np.ndarray = joblib.load(cpath)
            embeddings = embeddings.astype("float32")

            print(f"[DenseRetriever] Loaded cached embeddings from {cpath}")
        else:
            print(
                f"[DenseRetriever] Encoding {len(corpus)} docs with "
                f"{model_name} on {device.upper()}..."
            )

            embeddings = self.model.encode(
                [doc["text"] for doc in corpus],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype("float32")

            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(embeddings, cpath)

            print(f"[DenseRetriever] Embeddings cached to {cpath}")

        self._index = _build_faiss_index(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search corpus using dense semantic similarity."""
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]

        if self.shard_delay_sec:
            time.sleep(self.shard_delay_sec)

        q_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores_arr, indices = self._index.search(q_vec, top_k)

        results = []

        for score, idx in zip(scores_arr[0], indices[0]):
            if idx < 0 or idx >= len(self.corpus):
                continue

            results.append(
                {
                    **self.corpus[idx],
                    "score": float(score),
                    "retriever": self.name,
                }
            )

        return results


# ---------------------------------------------------------------------------
# Result fusion
# ---------------------------------------------------------------------------

def merge_results(
    result_sets: list[list[dict]],
    top_k: int | None = None,
) -> list[dict]:
    """Fuse results from multiple retrievers using reciprocal rank fusion."""
    if top_k is None:
        top_k = get_config()["retrieval"]["top_k"]

    merged: dict[int, dict] = {}

    for results in result_sets:
        for rank, doc in enumerate(results):
            doc_id = doc["doc_id"]
            score = doc["score"] + 1 / (rank + 1)

            existing = merged.get(doc_id)

            if existing is None:
                merged[doc_id] = {
                    **doc,
                    "merged_score": score,
                }
            else:
                existing["merged_score"] += score
                existing["retriever"] = (
                    f"{existing['retriever']}+{doc['retriever']}"
                )

    return sorted(
        merged.values(),
        key=lambda item: item["merged_score"],
        reverse=True,
    )[:top_k]


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_processed_corpus(limit: int | None = None) -> list[dict]:
    """Load the preprocessed corpus CSV from data/processed_corpus.csv."""
    if limit is None:
        limit = get_config()["data"]["corpus_limit"]

    path = Path(__file__).parent / "data" / "processed_corpus.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Run python D_preprocess.py first to create data/processed_corpus.csv"
        )

    df = pd.read_csv(path)

    if limit is not None:
        df = df.head(limit)

    corpus = []

    for _, row in df.iterrows():
        corpus.append(
            {
                "doc_id": int(row["doc_id"]),
                "source": row["source"],
                "question": row["question"],
                "gold_answer": row["answer"],
                "text": row["text"],
                "label": row["label"],
                "dataset": row["dataset"],
            }
        )

    return corpus


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke-test: load corpus and run a sample query with all three retrievers."""
    corpus = load_processed_corpus()
    query = "What are the symptoms of diabetes?"

    print("── TF-IDF ──────────────────────────────────────────────────")
    tfidf = TfidfRetriever(corpus)

    for i, doc in enumerate(tfidf.search(query), 1):
        preview = doc["text"][:120].replace(chr(10), " ")
        print(
            f"{i}. [{doc['source']}] "
            f"score={doc['score']:.4f}  {preview}"
        )

    print("\n── BM25-Lite ───────────────────────────────────────────────")
    bm25 = KeywordBM25LiteRetriever(corpus)

    for i, doc in enumerate(bm25.search(query), 1):
        preview = doc["text"][:120].replace(chr(10), " ")
        print(
            f"{i}. [{doc['source']}] "
            f"score={doc['score']:.4f}  {preview}"
        )

    print("\n── Dense ───────────────────────────────────────────────────")
    dense = DenseRetriever(corpus)

    for i, doc in enumerate(dense.search(query), 1):
        preview = doc["text"][:120].replace(chr(10), " ")
        print(
            f"{i}. [{doc['source']}] "
            f"score={doc['score']:.4f}  {preview}"
        )


if __name__ == "__main__":
    main()