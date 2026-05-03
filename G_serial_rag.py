"""Serial RAG baseline.

Queries are processed one at a time: retrieve -> merge -> generate -> cite.
This establishes the latency baseline against which the parallel pipeline
is benchmarked.
"""

from __future__ import annotations

import time
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


class SerialRAG:
    """Sequential retrieve-then-generate pipeline.

    All retrievers are called one after another; their results are fused
    before the answer is generated.

    Args:
        retrievers: A single retriever or a list of retrievers.
    """

    def __init__(self, retrievers) -> None:
        self.retrievers = retrievers if isinstance(retrievers, list) else [retrievers]

    def answer(self, query: dict, top_k: int | None = None) -> dict:
        """Answer a single query using serial retrieval.

        Args:
            query: Query dict with at least a ``"question"`` field.
            top_k: Number of evidence documents to retrieve per retriever.
                Defaults to ``retrieval.top_k`` from config.

        Returns:
            Dict with keys ``prediction``, ``answer``, ``citations``,
            ``latency`` (seconds), and ``evidence`` (list of doc dicts).
        """
        if top_k is None:
            top_k = get_config()["retrieval"]["top_k"]
        start = time.perf_counter()
        result_sets = [retriever.search(query["question"], top_k=top_k) for retriever in self.retrievers]
        evidence = merge_results(result_sets, top_k=top_k)
        generated = generate_answer(query["question"], evidence)
        latency = time.perf_counter() - start
        return {**generated, "latency": latency, "evidence": evidence}


def main() -> None:
    """Run a serial benchmark and save detailed results to ``report_results/``."""
    cfg = get_config()
    corpus = load_processed_corpus(limit=cfg["data"]["corpus_limit"])
    n_queries = cfg["benchmark"]["query_set_sizes"][1]   # medium query set (250)
    queries = build_queries(corpus, n_queries)
    shard_delay_sec = cfg["retrieval"]["shard_delay_sec"]
    tfidf_cfg = cfg["retrieval"]["tfidf_word"]
    char_cfg = cfg["retrieval"]["tfidf_char"]

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
    rag = SerialRAG(retrievers)

    print("=" * 65)
    print("  G_serial_rag -- Serial RAG Baseline")
    print(f"  Device      : {device.upper()}")
    print(f"  Corpus size : {len(corpus):,} documents")
    print(f"  Queries     : {len(queries)}")
    print(f"  Retrievers  : TF-IDF (word), TF-IDF (char_wb), BM25-lite, Dense (MiniLM)")
    print("=" * 65)

    rows = []
    for i, query in enumerate(queries, 1):
        output = rag.answer(query)
        rows.append(
            {
                "query_id": query["query_id"],
                "dataset": query.get("dataset", ""),
                "question": query["question"],
                "predicted_answer": output["prediction"],
                "ground_truth": query["gold_label"],
                "correct": output["prediction"] == query["gold_label"],
                "latency_sec": round(output["latency"], 4),
                "n_citations": len(output["citations"]),
                "citation": "; ".join(output["citations"]),
                "answer_text": output["answer"],
                "top_evidence": output["evidence"][0]["text"][:200] if output["evidence"] else "",
            }
        )
        if i % 10 == 0 or i == len(queries):
            print(f"  Processed {i:>4}/{len(queries)} queries ...", end="\r", flush=True)

    print()   # newline after progress
    out_dir = Path(__file__).parent / "report_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "serial_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # -- Summary statistics ---------------------------------------------------
    answerable = df[df["ground_truth"].isin(["yes", "no", "maybe"])]
    accuracy = answerable["correct"].mean() if len(answerable) else float("nan")
    pred_dist = df["predicted_answer"].value_counts().to_dict()

    print("\n" + "-" * 65)
    print("  RESULTS SUMMARY")
    print("-" * 65)
    print(f"  Total queries processed : {len(df)}")
    print(f"  Average latency         : {df['latency_sec'].mean():.4f} sec/query")
    print(f"  Min / Max latency       : {df['latency_sec'].min():.4f} / {df['latency_sec'].max():.4f} sec")
    print(f"  Accuracy (PubMedQA)     : {accuracy:.1%}  ({len(answerable)} answerable queries)")
    print(f"  Citation coverage       : {(df['n_citations'] > 0).mean():.1%}")
    print(f"  Prediction distribution : ", end="")
    print("  |  ".join(f"{k}: {v}" for k, v in sorted(pred_dist.items())))

    print("\n  Sample Predictions (first 5):")
    print(f"  {'#':<4} {'Dataset':<10} {'Prediction':<12} {'Truth':<8} {'Question'}")
    print(f"  {'-'*4} {'-'*10} {'-'*12} {'-'*8} {'-'*40}")
    for _, row in df.head(5).iterrows():
        q_short = row["question"][:45] + "..." if len(row["question"]) > 45 else row["question"]
        match = "[OK]" if row["correct"] else "[X]"
        print(f"  {match:<4} {row['dataset']:<10} {row['predicted_answer']:<12} {row['ground_truth']:<8} {q_short}")

    print("-" * 65)
    print(f"\n  Saved -> {out_path}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()
