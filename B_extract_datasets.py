"""Extract local/remote datasets into clean CSV files for the benchmark.

Usage
-----
Run once before the main benchmark to populate ``data/``:

    python B_extract_datasets.py

PubMedQA
    Downloaded from the official GitHub repository via HTTP if the local
    ``pubmedqa_labeled.csv`` is missing.  Pass ``--local-pubmedqa`` together
    with the path to the ``ori_pqal.json`` file if you have a manual download::

        python B_extract_datasets.py --local-pubmedqa /path/to/ori_pqal.json

MedQuAD / MedQA
    Pulled from Hugging Face with ``datasets.load_dataset``.  Requires an
    internet connection on first run; results are cached by HF locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "data"

_PUBMEDQA_GITHUB_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)


def extract_pubmedqa(local_json: Path | None = None) -> int:
    """Extract PubMedQA into ``data/pubmedqa_labeled.csv``.

    Resolution order:
    1. ``local_json`` argument (explicit file path supplied by caller).
    2. ``data/pubmedqa_labeled.csv`` already exists — skip extraction.
    3. Download ``ori_pqal.json`` from the official GitHub repository.

    Args:
        local_json: Optional path to a locally downloaded ``ori_pqal.json``.
            When ``None``, the function tries to download from GitHub.

    Returns:
        Row count written (or already present) in the output CSV.

    Raises:
        requests.HTTPError: If the GitHub download fails.
        FileNotFoundError: If ``local_json`` is given but does not exist.
    """
    existing = OUT / "pubmedqa_labeled.csv"

    if local_json is not None:
        if not local_json.exists():
            raise FileNotFoundError(
                f"Provided --local-pubmedqa path does not exist: {local_json}"
            )
        source_path = local_json
    else:
        if existing.exists():
            return len(pd.read_csv(existing))
        # Download from official GitHub mirror
        print("Downloading PubMedQA from GitHub …")
        response = requests.get(_PUBMEDQA_GITHUB_URL, timeout=120)
        response.raise_for_status()
        data = response.json()
        rows = [
            {
                "dataset": "PubMedQA",
                "doc_id": pmid,
                "question": row.get("QUESTION", ""),
                "evidence": " ".join(row.get("CONTEXTS", [])),
                "long_answer": row.get("LONG_ANSWER", ""),
                "label": row.get("final_decision", ""),
                "source": "official_pubmedqa_github",
            }
            for pmid, row in data.items()
        ]
        OUT.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(existing, index=False)
        print(f"  Saved {len(rows)} rows → {existing}")
        return len(rows)

    data = json.loads(source_path.read_text(encoding="utf-8"))
    rows = [
        {
            "dataset": "PubMedQA",
            "doc_id": pmid,
            "question": row.get("QUESTION", ""),
            "evidence": " ".join(row.get("CONTEXTS", [])),
            "long_answer": row.get("LONG_ANSWER", ""),
            "label": row.get("final_decision", ""),
            "source": "local_pubmedqa_json",
        }
        for pmid, row in data.items()
    ]
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(existing, index=False)
    print(f"  Saved {len(rows)} rows → {existing}")
    return len(rows)


def extract_medquad() -> int:
    """Extract MedQuAD from HuggingFace into ``data/medquad.csv``.

    If ``data/medquad.csv`` already exists the download is skipped and the
    existing row count is returned.

    Returns:
        Row count written (or already present) in the output CSV.
    """
    existing = OUT / "medquad.csv"
    if existing.exists():
        count = len(pd.read_csv(existing))
        print(f"  medquad.csv already present — {count} rows, skipping download.")
        return count
    print("Loading MedQuAD from HuggingFace …")
    ds = load_dataset("lavita/MedQuAD")
    medquad = ds["train"].to_pandas()
    cols = [
        "document_id",
        "document_source",
        "document_url",
        "question_id",
        "question",
        "answer",
        "question_type",
        "umls_semantic_group",
    ]
    medquad = medquad[cols]
    medquad.insert(0, "dataset", "MedQuAD")
    OUT.mkdir(exist_ok=True)
    medquad.to_csv(existing, index=False)
    print(f"  Saved {len(medquad)} rows → {existing}")
    return len(medquad)


def extract_medqa() -> int:
    """Extract MedQA USMLE from HuggingFace into ``data/medqa_usmle.csv``.

    If ``data/medqa_usmle.csv`` already exists the download is skipped and
    the existing row count is returned.

    Returns:
        Row count written (or already present) in the output CSV.
    """
    existing = OUT / "medqa_usmle.csv"
    if existing.exists():
        count = len(pd.read_csv(existing))
        print(f"  medqa_usmle.csv already present — {count} rows, skipping download.")
        return count
    print("Loading MedQA from HuggingFace …")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    medqa = ds["train"].to_pandas()
    cols = ["question", "answer", "options", "meta_info", "answer_idx"]
    medqa = medqa[cols]
    medqa.insert(0, "dataset", "MedQA")
    OUT.mkdir(exist_ok=True)
    medqa.to_csv(existing, index=False)
    print(f"  Saved {len(medqa)} rows → {existing}")
    return len(medqa)


def _write_summary(pubmed_count: int, medquad_count: int, medqa_count: int) -> None:
    """Write a human-readable dataset summary to ``data/dataset_summary.txt``.

    Args:
        pubmed_count: Rows in PubMedQA CSV.
        medquad_count: Rows in MedQuAD CSV.
        medqa_count: Rows in MedQA CSV.
    """
    pubmed = pd.read_csv(OUT / "pubmedqa_labeled.csv")
    medquad = pd.read_csv(OUT / "medquad.csv")
    medqa = pd.read_csv(OUT / "medqa_usmle.csv")
    summary = "\n".join(
        [
            f"PubMedQA rows: {pubmed_count}",
            f"MedQuAD rows: {medquad_count}",
            f"MedQA rows: {medqa_count}",
            f"PubMedQA labels: {pubmed['label'].value_counts().to_dict()}",
            f"MedQuAD top sources: {medquad['document_source'].value_counts().head(10).to_dict()}",
            f"MedQA answer option labels: {medqa['answer_idx'].value_counts().sort_index().to_dict()}",
            "",
        ]
    )
    (OUT / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    """CLI entry point.  Run ``python B_extract_datasets.py --help`` for options."""
    parser = argparse.ArgumentParser(description="Extract medical QA datasets to CSV")
    parser.add_argument(
        "--local-pubmedqa",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to a locally downloaded ori_pqal.json. "
            "When omitted the file is fetched from GitHub automatically."
        ),
    )
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    pubmed_count = extract_pubmedqa(local_json=args.local_pubmedqa)
    medquad_count = extract_medquad()
    medqa_count = extract_medqa()
    _write_summary(pubmed_count, medquad_count, medqa_count)


if __name__ == "__main__":
    main()
