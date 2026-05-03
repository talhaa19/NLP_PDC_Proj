"""Parallel RAG pipeline with two levels of concurrency.

**Retriever-level parallelism** (existing): for a single query, all
retrievers run concurrently inside a ``ThreadPoolExecutor``.

**Query-level parallelism** (new -- :meth:`ParallelRAG.batch_answer`):
multiple queries are dispatched to the thread pool simultaneously so that
both retrieval and generation overlap across queries.  ``evaluate.py`` uses
``batch_answer`` for the parallel condition, which yields deeper speedup than
per-query retriever parallelism alone.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import torch

from A_config import get_config
from D_preprocess import build_queries
from E_retriever import (
    DenseRetriever,
    KeywordBM25LiteRetriever,
    TfidfRetriever,
    load_processed_corpus,
    merge_results,
)
from F_llm_generator import generate_answer


def critic_check(evidence_docs: list[dict]) -> bool:
    """Validate that the merged evidence contains useful biomedical content.

    Returns ``False`` when the evidence looks uninformative so the generator
    can fall back to a minimal citation rather than fabricating an answer.

    Args:
        evidence_docs: Top merged documents from :func:`retriever.merge_results`.

    Returns:
        ``True`` if at least one biomedical signal term is found in the top
        three documents; ``False`` otherwise.
    """
    useful_terms = ("study", "clinical", "treatment", "symptoms", "evidence", "reduced", "significant")
    text = " ".join(doc["text"].lower() for doc in evidence_docs[:3])
    return any(term in text for term in useful_terms)


class ParallelRAG:
    """Two-level parallel retrieve-then-generate pipeline.

    **Retriever level**: all retrievers for one query run concurrently.
    **Query level**: :meth:`batch_answer` dispatches multiple queries at
    once so retrieval and generation overlap across the batch.

    Args:
        retrievers: List of retriever objects.
        max_workers: Thread pool size.  Defaults to ``2``.
    """

    def __init__(self, retrievers: list, max_workers: int = 2) -> None:
        self.retrievers = retrievers
        self.max_workers = max_workers

    def answer(self, query: dict, top_k: int | None = None) -> dict:
        """Answer a single query using retriever-level parallelism.

        All configured retrievers search in parallel; their results are fused,
        critic-checked, and then passed to the answer generator.

        Args:
            query: Query dict with at least a ``"question"`` field.
            top_k: Evidence documents per retriever.  Defaults to
                ``retrieval.top_k`` from config.

        Returns:
            Dict with keys ``prediction``, ``answer``, ``citations``,
            ``latency``, ``evidence``, and ``critic_passed``.
        """
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(ret.search, query["question"], top_k) for ret in self.retrievers]
            result_sets = [future.result() for future in futures]
        evidence = merge_results(result_sets, top_k=top_k)
        passed_critic = critic_check(evidence)
        generated = generate_answer(query["question"], evidence if passed_critic else evidence[:1])
        latency = time.perf_counter() - start
        return {**generated, "latency": latency, "evidence": evidence, "critic_passed": passed_critic}

    def batch_answer(self, queries: list[dict], top_k: int | None = None) -> list[dict]:
        """Answer a batch of queries with full query-level parallelism.

        Each query is submitted as a separate task to the thread pool, so
        retrieval and generation overlap across queries in addition to the
        per-query retriever-level parallelism.

        Args:
            queries: List of query dicts, each with at least a ``"question"``
                field.
            top_k: Evidence documents per retriever per query.  Defaults to
                ``retrieval.top_k`` from config.

        Returns:
            List of result dicts in the same order as *queries*.
        """
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]

        results: list[dict | None] = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.answer, query, top_k): idx
                for idx, query in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]


def main() -> None:
    """Run a parallel benchmark across worker counts and save detailed results."""
    cfg = get_config()
    corpus = load_processed_corpus(limit=cfg["data"]["corpus_limit"])
    n_queries = cfg["benchmark"]["query_set_sizes"][1]   # medium query set (250)
    queries = build_queries(corpus, n_queries)
    shard_delay_sec = cfg["retrieval"]["shard_delay_sec"]
    tfidf_cfg = cfg["retrieval"]["tfidf_word"]
    char_cfg = cfg["retrieval"]["tfidf_char"]
    worker_counts = cfg["benchmark"]["worker_counts"]

    # Use CUDA for dense retrieval if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrievers = [
        TfidfRetriever(corpus, analyzer=tfidf_cfg["analyzer"], ngram_range=tuple(tfidf_cfg["ngram_range"]), shard_delay_sec=shard_delay_sec),
        TfidfRetriever(corpus, analyzer=char_cfg["analyzer"], ngram_range=tuple(char_cfg["ngram_range"]), shard_delay_sec=shard_delay_sec),
        KeywordBM25LiteRetriever(corpus, shard_delay_sec=shard_delay_sec),
        # Dense retrieval: finds semantically relevant docs even when keywords
        # don't match exactly -- significantly improves evidence quality and
        # downstream answer accuracy.  Embeddings are cached after first run.
        DenseRetriever(
            corpus,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=device,
            shard_delay_sec=shard_delay_sec,
        ),
    ]

    print("=" * 65)
    print("  H_parallel_rag -- Parallel RAG Pipeline")
    print(f"  Device       : {device.upper()}")
    print(f"  Corpus size  : {len(corpus):,} documents")
    print(f"  Queries      : {len(queries)}")
    print(f"  Worker counts: {worker_counts}")
    print(f"  Retrievers   : TF-IDF (word), TF-IDF (char_wb), BM25-lite, Dense (MiniLM)")
    print("=" * 65)

    rows = []
    worker_summaries = []

    for workers in worker_counts:
        print(f"\n  Running with {workers} worker(s) ...")
        rag = ParallelRAG(retrievers, max_workers=workers)
        outputs = rag.batch_answer(queries)
        critic_pass = 0
        for output, query in zip(outputs, queries):
            if output.get("critic_passed", False):
                critic_pass += 1
            rows.append(
                {
                    "query_id": query["query_id"],
                    "dataset": query.get("dataset", ""),
                    "question": query["question"],
                    "predicted_answer": output["prediction"],
                    "ground_truth": query["gold_label"],
                    "correct": output["prediction"] == query["gold_label"],
                    "latency_sec": round(output["latency"], 4),
                    "workers": workers,
                    "critic_passed": output.get("critic_passed", False),
                    "n_citations": len(output["citations"]),
                    "citation": "; ".join(output["citations"]),
                    "answer_text": output["answer"],
                    "top_evidence": output["evidence"][0]["text"][:200] if output["evidence"] else "",
                }
            )
        grp_df = pd.DataFrame([r for r in rows if r["workers"] == workers])
        avg_lat = grp_df["latency_sec"].mean()
        answerable = grp_df[grp_df["ground_truth"].isin(["yes", "no", "maybe"])]
        acc = answerable["correct"].mean() if len(answerable) else float("nan")
        worker_summaries.append(
            {
                "workers": workers,
                "avg_latency_sec": avg_lat,
                "accuracy": acc,
                "critic_pass_rate": critic_pass / len(queries),
            }
        )
        print(f"    ✓ Done -- avg latency: {avg_lat:.4f} sec | accuracy: {acc:.1%} | critic pass: {critic_pass/len(queries):.1%}")

    out_dir = Path(__file__).parent / "report_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "parallel_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # -- Comparative summary table --------------------------------------------
    # Use workers=1 as the baseline for speedup
    base_lat = next(s["avg_latency_sec"] for s in worker_summaries if s["workers"] == 1) if any(s["workers"] == 1 for s in worker_summaries) else worker_summaries[0]["avg_latency_sec"]

    print("\n" + "-" * 65)
    print("  PARALLEL RAG -- WORKER COMPARISON")
    print("-" * 65)
    print(f"  {'Workers':<10} {'Avg Latency':>14} {'Speedup':>10} {'Efficiency':>12} {'Accuracy':>10} {'Critic OK':>10}")
    print(f"  {'-'*8:<10} {'-'*12:>14} {'-'*8:>10} {'-'*10:>12} {'-'*8:>10} {'-'*9:>10}")
    for s in worker_summaries:
        speedup = base_lat / s["avg_latency_sec"]
        efficiency = speedup / s["workers"]
        print(
            f"  {s['workers']:<10} {s['avg_latency_sec']:>13.4f}s "
            f"{speedup:>10.3f} {efficiency:>12.3f} "
            f"{s['accuracy']:>10.1%} {s['critic_pass_rate']:>10.1%}"
        )

    print("-" * 65)
    print(f"\n  Total rows saved : {len(df)}")
    print(f"  Saved -> {out_path}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()
