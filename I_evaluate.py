"""Experiment runner and chart/report artifact generation.

Runs the full benchmark matrix: every combination of query-set size and
worker count is evaluated under both the serial and parallel pipelines.
The parallel condition uses :meth:`ParallelRAG.batch_answer` for query-level
concurrency in addition to per-query retriever-level parallelism.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd

from A_config import get_config
from G_serial_rag import SerialRAG
from H_parallel_rag import ParallelRAG


def _score_outputs(outputs: list[dict], queries: list[dict]) -> dict:
    """Compute retrieval and generation quality metrics for a result set.

    Args:
        outputs: List of result dicts from :meth:`SerialRAG.answer` or
            :meth:`ParallelRAG.answer`.
        queries: Corresponding query dicts with ``gold_doc_id`` and
            ``gold_label`` fields.

    Returns:
        Dict with keys ``hit_rate``, ``citation_coverage``, and ``accuracy``.
        ``accuracy`` is only computed for queries whose gold label is in
        ``{"yes", "no", "maybe"}``; it is ``0.0`` when none qualify.
    """
    # Hit = at least one retrieved doc is from the same dataset as the query
    # but is NOT the exact source document (prevents trivial 100% from exact-match recall
    # when queries are derived from corpus docs).  This measures genuine cross-document
    # retrieval quality: can the system surface *other* relevant docs beyond the source?
    hit_rate = mean(
        1.0 if any(
            doc.get("dataset") == query.get("dataset")
            and doc["doc_id"] != query["gold_doc_id"]
            for doc in out["evidence"]
        ) else 0.0
        for out, query in zip(outputs, queries)
    )
    citation_coverage = mean(1.0 if out["citations"] else 0.0 for out in outputs)
    answerable = [
        (out, query)
        for out, query in zip(outputs, queries)
        if query["gold_label"] in {"yes", "no", "maybe"}
    ]
    accuracy = (
        mean(1.0 if out["prediction"] == query["gold_label"] else 0.0 for out, query in answerable)
        if answerable
        else 0.0
    )
    return {"hit_rate": hit_rate, "citation_coverage": citation_coverage, "accuracy": accuracy}


def run_experiments(
    corpus: list[dict],
    query_sets: dict[int, list[dict]],
    serial_retriever,
    parallel_retrievers: list,
    output_dir: Path,
    worker_counts: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Execute the full benchmark matrix and write result CSVs and charts.

    For each query-set size the serial pipeline runs once; the parallel
    pipeline runs once per worker count using :meth:`ParallelRAG.batch_answer`
    so queries execute concurrently.

    Args:
        corpus: Full document corpus (unused directly; passed for size info).
        query_sets: Mapping of ``n_queries -> list[query_dict]``.
        serial_retriever: Retriever or list of retrievers for the serial
            baseline.
        parallel_retrievers: List of retrievers for the parallel pipeline.
        output_dir: Directory where CSVs and charts are written.
        worker_counts: Thread pool sizes to test.  Defaults to
            ``benchmark.worker_counts`` from config.

    Returns:
        DataFrame with one row per (query_set_size, worker_count) combination
        containing latency, speedup, efficiency, accuracy, and citation metrics.
    """
    if worker_counts is None:
        worker_counts = tuple(get_config()["benchmark"]["worker_counts"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_rows: list[dict] = []
    serial = SerialRAG(serial_retriever)

    for n_queries, queries in query_sets.items():
        serial_outputs = [serial.answer(query) for query in queries]
        serial_latency = sum(out["latency"] for out in serial_outputs)
        serial_scores = _score_outputs(serial_outputs, queries)

        for workers in worker_counts:
            parallel = ParallelRAG(parallel_retrievers, max_workers=workers)
            if get_config()["benchmark"].get("use_batch_parallel", False):
                parallel_outputs = parallel.batch_answer(queries)
            else:
                parallel_outputs = [parallel.answer(query) for query in queries]
            parallel_latency = sum(out["latency"] for out in parallel_outputs)
            parallel_scores = _score_outputs(parallel_outputs, queries)
            speedup = serial_latency / parallel_latency if parallel_latency else 0.0
            rows.append(
                {
                    "queries": n_queries,
                    "workers": workers,
                    "serial_latency_sec": serial_latency,
                    "parallel_latency_sec": parallel_latency,
                    "speedup": speedup,
                    "efficiency": speedup / workers,
                    "accuracy": parallel_scores["accuracy"],
                    "citation_coverage": parallel_scores["citation_coverage"],
                    "retrieval_hit_rate": parallel_scores["hit_rate"],
                    "serial_accuracy": serial_scores["accuracy"],
                    "serial_citation_coverage": serial_scores["citation_coverage"],
                }
            )

            if n_queries == min(query_sets) and workers == max(worker_counts):
                for query, output in list(zip(queries, parallel_outputs))[:5]:
                    sample_rows.append(
                        {
                            "question": query["question"],
                            "gold_label": query["gold_label"],
                            "prediction": output["prediction"],
                            "citations": "; ".join(output["citations"]),
                            "answer": output["answer"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "latency_results.csv", index=False)
    df[["queries", "workers", "accuracy", "citation_coverage", "retrieval_hit_rate"]].to_csv(
        output_dir / "citation_results.csv", index=False
    )
    pd.DataFrame(sample_rows).to_csv(output_dir / "sample_outputs.csv", index=False)
    _make_charts(df, output_dir)
    return df


def _make_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate speedup and efficiency line charts and save as PNGs.

    Args:
        df: Results DataFrame from :func:`run_experiments`.
        output_dir: Directory where chart PNGs are written.
    """
    for metric, filename, ylabel in [
        ("speedup", "speedup_chart.png", "Speedup over serial"),
        ("efficiency", "efficiency_chart.png", "Parallel efficiency"),
    ]:
        plt.figure(figsize=(8, 5))
        for workers, group in df.groupby("workers"):
            group = group.sort_values("queries")
            plt.plot(group["queries"], group[metric], marker="o", label=f"{workers} worker(s)")
        plt.xlabel("Number of queries")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by workload")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    """Recompute metrics from previously saved serial/parallel CSV files."""
    output_dir = Path(__file__).parent / "report_results"
    serial_path = output_dir / "serial_results.csv"
    parallel_path = output_dir / "parallel_results.csv"
    if not serial_path.exists() or not parallel_path.exists():
        raise SystemExit("Run python G_serial_rag.py and python H_parallel_rag.py before evaluation")

    serial = pd.read_csv(serial_path)
    parallel = pd.read_csv(parallel_path)
    # Support both old column name ("latency") and new ("latency_sec")
    serial_lat_col = "latency_sec" if "latency_sec" in serial.columns else "latency"
    par_lat_col = "latency_sec" if "latency_sec" in parallel.columns else "latency"
    serial_avg = serial[serial_lat_col].mean()

    def _cross_doc_hit_rate(df: pd.DataFrame) -> float:
        """Hit rate: fraction of queries where at least one returned citation
        is from the *same dataset* as the query but is NOT the exact source doc.

        This avoids trivial 100% hit rates that arise when queries are derived
        directly from corpus documents (the source doc is always retrievable).
        It instead measures genuine cross-document retrieval quality.
        """
        hits = 0
        for _, row in df.iterrows():
            dataset = str(row.get("dataset", ""))
            qid = row.get("query_id", -1)
            source_doc = f"{dataset}-doc-{qid}"
            citations = [c.strip() for c in str(row.get("citation", "")).split(";") if c.strip()]
            # A hit = retrieved a different doc from the same dataset (not the trivial exact match)
            if any(c.startswith(f"{dataset}-doc-") and c != source_doc for c in citations):
                hits += 1
        return hits / len(df) if len(df) else 0.0

    rows = []
    for workers, group in parallel.groupby("workers"):
        parallel_avg = group[par_lat_col].mean()
        speedup = serial_avg / parallel_avg if parallel_avg else 0.0
        answerable = group[group["ground_truth"].isin(["yes", "no", "maybe"])]
        accuracy = (answerable["predicted_answer"] == answerable["ground_truth"]).mean() if len(answerable) else float("nan")
        citation_coverage = group["citation"].fillna("").astype(str).str.len().gt(0).mean()
        hit_rate = _cross_doc_hit_rate(group)
        rows.append(
            {
                "queries": len(serial),
                "workers": int(workers),
                "serial_latency_sec": serial_avg,
                "parallel_latency_sec": parallel_avg,
                "speedup": speedup,
                "efficiency": speedup / int(workers),
                "accuracy": accuracy,
                "citation_coverage": citation_coverage,
                "retrieval_hit_rate": hit_rate,
            }
        )

    results = pd.DataFrame(rows).sort_values("workers")
    results.to_csv(output_dir / "latency_results.csv", index=False)
    results[["queries", "workers", "accuracy", "citation_coverage", "retrieval_hit_rate"]].to_csv(
        output_dir / "citation_results.csv", index=False
    )
    _make_charts(results, output_dir)

    # -- Comprehensive printed summary ----------------------------------------
    print("=" * 72)
    print("  I_evaluate -- Benchmark Results Summary")
    print("=" * 72)
    print(f"  Serial baseline  : {serial_avg:.4f} sec/query  ({len(serial)} queries)")
    print(f"  Parallel datasets: {len(results)} worker configurations evaluated")

    print("\n" + "-" * 72)
    print(f"  {'Workers':>7} {'Avg Par.(s)':>12} {'Speedup':>9} {'Efficiency':>11} "
          f"{'Accuracy':>10} {'Citation%':>10} {'HitRate':>9}")
    print(f"  {'-'*7:>7} {'-'*10:>12} {'-'*7:>9} {'-'*9:>11} "
          f"{'-'*8:>10} {'-'*8:>10} {'-'*7:>9}")
    for _, row in results.iterrows():
        print(
            f"  {int(row['workers']):>7} {row['parallel_latency_sec']:>12.4f} "
            f"{row['speedup']:>9.3f} {row['efficiency']:>11.3f} "
            f"{row['accuracy']:>9.1%}  {row['citation_coverage']:>9.1%}  {row['retrieval_hit_rate']:>8.1%}"
        )
    print("-" * 72)

    # -- Interpretation notes -------------------------------------------------
    best_row = results.loc[results["speedup"].idxmax()]
    print(f"\n  Best speedup    : {best_row['speedup']:.3f}x with {int(best_row['workers'])} worker(s)")
    print(f"  Accuracy note   : Only PubMedQA queries (yes/no/maybe) count toward accuracy.")
    print(f"  MedQuAD / MedQA queries are labelled 'faq'/'exam' and excluded from accuracy.")

    # -- Sample predictions ---------------------------------------------------
    sample_path = output_dir / "sample_outputs.csv"
    if sample_path.exists():
        samples = pd.read_csv(sample_path)
        print(f"\n  Sample Predictions ({len(samples)} shown):")
        print(f"  {'Gold':>6}  {'Pred':>6}  {'Question'}")
        print(f"  {'-'*6:>6}  {'-'*6:>6}  {'-'*50}")
        for _, srow in samples.iterrows():
            match = "[OK]" if srow["gold_label"] == srow["prediction"] else "[X]"
            q_short = str(srow["question"])[:55] + "..." if len(str(srow["question"])) > 55 else str(srow["question"])
            print(f"  {srow['gold_label']:>6}  {srow['prediction']:>5}{match}  {q_short}")

    print("\n" + "-" * 72)
    print(f"  Artifacts saved to: {output_dir}")
    for fname in ["latency_results.csv", "citation_results.csv", "speedup_chart.png", "efficiency_chart.png"]:
        print(f"    ✓ {fname}")
    print("=" * 72)


if __name__ == "__main__":
    main()
