"""Main entry point — full serial-vs-parallel benchmark using dense retrievers.

Runs the complete pipeline end-to-end:
  1. Load datasets from local CSVs (no internet required after first run)
  2. Build corpus and query sets from config.yaml
  3. Initialise three dense retrievers (all-mpnet-base-v2, all-MiniLM-L6-v2,
     intfloat/e5-small-v2) -- GPU-accelerated when CUDA is available
  4. Run full benchmark matrix: all query-set sizes x all worker counts
     Writes CSVs and charts to report_results/

Usage:
    python L_main.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from A_config import get_config
from C_data_loader import load_all_datasets
from D_preprocess import build_corpus, build_queries
from E_retriever import DenseRetriever
from I_evaluate import run_experiments


def main() -> None:
    """Run the full serial-vs-parallel benchmark using config.yaml values."""
    t_start = time.perf_counter()
    cfg = get_config()
    output_dir = Path(__file__).parent / "report_results"
    output_dir.mkdir(exist_ok=True)

    # ── Hardware detection ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A (CPU only)"
    print("=" * 68)
    print("  L_main -- Full Distributed RAG Benchmark")
    print("=" * 68)
    print(f"  Device       : {device.upper()}  {gpu_name}")
    print(f"  Query sets   : {cfg['benchmark']['query_set_sizes']}")
    print(f"  Worker counts: {cfg['benchmark']['worker_counts']}")

    # ── Data loading ──────────────────────────────────────────────────────────
    print("\n  [1/3] Loading datasets ...")
    samples, mode = load_all_datasets(
        max_samples_per_dataset=cfg["benchmark"]["main_max_samples"]
    )
    corpus = build_corpus(samples)
    query_sets = {
        n: build_queries(corpus, n)
        for n in cfg["benchmark"]["query_set_sizes"]
    }
    print(f"       Mode: {mode} | Corpus: {len(corpus):,} docs | "
          f"Query sets: {list(query_sets.keys())}")

    # ── Retrievers ────────────────────────────────────────────────────────────
    retrieval_cfg = cfg["retrieval"]
    shard_delay_sec = retrieval_cfg["shard_delay_sec"]

    print("\n  [2/3] Initialising dense retrievers ...")
    dense_models = [
        ("sentence-transformers/all-mpnet-base-v2",  "MPNet-base  (768-dim)"),
        ("sentence-transformers/all-MiniLM-L6-v2",   "MiniLM-L6  (384-dim)"),
        ("intfloat/e5-small-v2",                     "E5-small    (384-dim)"),
    ]
    retrievers = []
    for model_name, label in dense_models:
        print(f"       Loading {label} ...")
        r = DenseRetriever(
            corpus,
            model_name=model_name,
            device=device,
            shard_delay_sec=shard_delay_sec,
        )
        retrievers.append(r)
        print(f"       OK -- {label}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\n  [3/3] Running benchmark matrix ...")
    results = run_experiments(
        corpus=corpus,
        query_sets=query_sets,
        serial_retriever=retrievers,
        parallel_retrievers=retrievers,
        output_dir=output_dir,
        worker_counts=tuple(cfg["benchmark"]["worker_counts"]),
    )

    elapsed = time.perf_counter() - t_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BENCHMARK COMPLETE")
    print("=" * 68)
    print(f"  Data mode    : {mode}")
    print(f"  Corpus size  : {len(corpus):,} documents")
    print(f"  Device used  : {device.upper()}")
    print(f"  Total time   : {elapsed/60:.1f} min")
    print(f"\n  Results:")
    print(results.to_string(index=False))
    print(f"\n  Artifacts written to: {output_dir}")
    for fname in ["latency_results.csv", "citation_results.csv",
                  "speedup_chart.png", "efficiency_chart.png"]:
        p = output_dir / fname
        status = "OK" if p.exists() else "MISSING"
        print(f"    [{status}] {fname}")
    print("=" * 68)


if __name__ == "__main__":
    main()
