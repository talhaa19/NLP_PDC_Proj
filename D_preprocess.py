"""Preprocessing utilities for corpus and query construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from A_config import get_config
from C_data_loader import RawSample
from C_data_loader import load_all_datasets


def build_corpus(samples: list[RawSample]) -> list[dict]:
    """Convert a list of raw samples into the corpus format expected by retrievers.

    Each document dict contains ``doc_id``, ``dataset``, ``text``,
    ``question``, ``gold_answer``, ``label``, and ``source``.

    Args:
        samples: Raw samples from :func:`data_loader.load_all_datasets`.

    Returns:
        List of document dicts suitable for :class:`retriever.TfidfRetriever`.
    """
    return [
        {
            "doc_id": idx,
            "dataset": sample.dataset,
            "text": sample.evidence,
            "question": sample.question,
            "gold_answer": sample.answer,
            "label": sample.label,
            "source": f"{sample.dataset}-doc-{idx}",
        }
        for idx, sample in enumerate(samples)
    ]


def build_queries(corpus: list[dict], n_queries: int) -> list[dict]:
    """Sample the first *n_queries* corpus entries as evaluation queries.

    Each query has a known ``gold_doc_id`` equal to its corpus position,
    which is used to compute retrieval hit rate in evaluation.

    Args:
        corpus: Output of :func:`build_corpus`.
        n_queries: Number of queries to build; capped at ``len(corpus)``.

    Returns:
        List of query dicts with keys ``query_id``, ``question``,
        ``gold_doc_id``, ``gold_label``, and ``dataset``.
    """
    size = min(n_queries, len(corpus))
    return [
        {
            "query_id": item["doc_id"],
            "question": item["question"],
            "gold_doc_id": item["doc_id"],
            "gold_label": item["label"],
            "dataset": item["dataset"],
        }
        for item in corpus[:size]
    ]


def main() -> None:
    """Build and save the processed corpus CSV to ``data/processed_corpus.csv``."""
    cfg = get_config()
    max_samples = cfg["data"]["max_samples_per_dataset"]
    samples, mode = load_all_datasets(max_samples_per_dataset=max_samples)
    corpus = build_corpus(samples)
    rows = [
        {
            "doc_id": item["doc_id"],
            "source": item["source"],
            "question": item["question"],
            "answer": item["gold_answer"],
            "text": item["text"],
            "label": item["label"],
            "dataset": item["dataset"],
        }
        for item in corpus
    ]
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "processed_corpus.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # -- Summary --------------------------------------------------------------
    print("=" * 60)
    print("  D_preprocess -- Corpus & Query Builder")
    print("=" * 60)
    print(f"  Data source     : {mode}")
    print(f"  Corpus rows     : {len(rows):,}")
    print(f"  Saved to        : {out_path}")
    print(f"  Columns         : {', '.join(df.columns.tolist())}")

    print("\n  Dataset breakdown:")
    for dataset, grp in df.groupby("dataset"):
        print(f"    {dataset:<12} {len(grp):>5} docs")

    print("\n  Label distribution (across all datasets):")
    for label, cnt in df["label"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {label:<10} {cnt:>5}  ({pct:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
