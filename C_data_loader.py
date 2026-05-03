"""Dataset loading for the medical FAQ RAG benchmark.

The loader uses the requested Hugging Face datasets when available.  A small
deterministic fallback corpus is included so the project remains runnable in
offline classroom environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from A_config import get_config


@dataclass
class RawSample:
    """A single question/evidence/answer triple from any of the three datasets."""

    dataset: str
    question: str
    evidence: str
    answer: str
    label: str


def _safe_text(value: Any) -> str:
    """Coerce any value to a clean string, flattening lists and dicts.

    Args:
        value: Any Python value returned by a dataset row field.

    Returns:
        Stripped string representation, or empty string for ``None``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(_safe_text(v) for v in value if _safe_text(v)).strip()
    if isinstance(value, dict):
        return " ".join(_safe_text(v) for v in value.values() if _safe_text(v)).strip()
    return str(value).strip()


def _load_pubmedqa_github(max_samples: int) -> list[RawSample]:
    """Fetch PubMedQA JSON from the official GitHub repository.

    Args:
        max_samples: Maximum number of samples to return.

    Returns:
        List of :class:`RawSample` objects.

    Raises:
        requests.RequestException: On network or HTTP errors.
    """
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    data = requests.get(url, timeout=60).json()
    samples: list[RawSample] = []
    for pmid, row in list(data.items())[:max_samples]:
        question = _safe_text(row.get("QUESTION"))
        evidence = _safe_text(row.get("CONTEXTS"))
        answer = _safe_text(row.get("LONG_ANSWER"))
        label = _safe_text(row.get("final_decision")).lower()
        if question and evidence:
            samples.append(RawSample("PubMedQA", question, evidence, answer, label or "maybe"))
    return samples


def _load_local_csvs(max_samples: int) -> list[RawSample]:
    """Load datasets from pre-extracted CSV files in ``data/``.

    Args:
        max_samples: Maximum rows to read from each CSV file.

    Returns:
        Combined list of :class:`RawSample` objects, or empty list if no CSVs
        are found.
    """
    data_dir = Path(__file__).parent / "data"
    samples: list[RawSample] = []

    pubmed_path = data_dir / "pubmedqa_labeled.csv"
    if pubmed_path.exists():
        pubmed = pd.read_csv(pubmed_path).head(max_samples)
        for _, row in pubmed.iterrows():
            samples.append(
                RawSample(
                    "PubMedQA",
                    _safe_text(row.get("question")),
                    _safe_text(row.get("evidence")),
                    _safe_text(row.get("long_answer")),
                    _safe_text(row.get("label")).lower(),
                )
            )

    medquad_path = data_dir / "medquad.csv"
    if medquad_path.exists():
        medquad = pd.read_csv(medquad_path).head(max_samples)
        for _, row in medquad.iterrows():
            samples.append(
                RawSample(
                    "MedQuAD",
                    _safe_text(row.get("question")),
                    _safe_text(row.get("answer")),
                    _safe_text(row.get("answer")),
                    "faq",
                )
            )

    medqa_path = data_dir / "medqa_usmle.csv"
    if medqa_path.exists():
        medqa = pd.read_csv(medqa_path).head(max_samples)
        for _, row in medqa.iterrows():
            question = _safe_text(row.get("question"))
            answer = _safe_text(row.get("answer"))
            options = _safe_text(row.get("options"))
            samples.append(
                RawSample(
                    "MedQA",
                    question,
                    f"{question} Options: {options} Correct answer: {answer}",
                    answer,
                    "exam",
                )
            )

    return samples


def _load_hf_datasets(max_samples: int) -> list[RawSample]:
    """Download datasets from HuggingFace Hub.

    Tries PubMedQA GitHub, then MedQuAD, then MedQA.  Each source failure is
    logged but does not abort the others.

    Args:
        max_samples: Maximum samples to pull from each dataset.

    Returns:
        Combined list of :class:`RawSample` objects from all successful sources.
    """
    from datasets import load_dataset

    samples: list[RawSample] = []

    try:
        samples.extend(_load_pubmedqa_github(max_samples))
    except Exception as exc:
        print(f"PubMedQA GitHub load failed: {exc}")

    try:
        medquad = load_dataset("lavita/MedQuAD")
        split = medquad["train"] if "train" in medquad else next(iter(medquad.values()))
        for row in split.select(range(min(max_samples, len(split)))):
            question = _safe_text(row.get("Question") or row.get("question"))
            answer = _safe_text(row.get("Answer") or row.get("answer"))
            if question and answer:
                samples.append(RawSample("MedQuAD", question, answer, answer, "faq"))
    except Exception as exc:
        print(f"MedQuAD load failed: {exc}")

    try:
        medqa = load_dataset("GBaker/MedQA-USMLE-4-options")
        split = medqa["train"] if "train" in medqa else next(iter(medqa.values()))
        for row in split.select(range(min(max_samples, len(split)))):
            question = _safe_text(row.get("question"))
            answer = _safe_text(row.get("answer") or row.get("answer_idx"))
            options = _safe_text(row.get("options"))
            evidence = f"{question} Options: {options} Correct answer: {answer}"
            if question:
                samples.append(RawSample("MedQA", question, evidence, answer, "exam"))
    except Exception as exc:
        print(f"MedQA mirror load failed: {exc}")

    return samples


def _fallback_samples(target_size: int) -> list[RawSample]:
    """Generate a deterministic synthetic corpus for offline use.

    Cycles through six template entries to reach *target_size* samples.

    Args:
        target_size: Number of synthetic samples to generate.

    Returns:
        List of :class:`RawSample` objects.
    """
    templates = [
        (
            "PubMedQA",
            "Does aspirin reduce recurrent cardiovascular events?",
            "A randomized biomedical study reported that aspirin significantly reduced recurrent cardiovascular events in high risk patients, although bleeding risk increased.",
            "Aspirin reduced recurrent cardiovascular events in the cited trial.",
            "yes",
        ),
        (
            "PubMedQA",
            "Is vitamin D supplementation associated with fewer asthma attacks?",
            "The clinical abstract found no significant reduction in asthma attacks after vitamin D supplementation compared with placebo.",
            "The evidence did not show fewer asthma attacks.",
            "no",
        ),
        (
            "PubMedQA",
            "Can probiotics improve symptoms of irritable bowel syndrome?",
            "A systematic review found mixed results: some probiotic strains improved symptoms, while other trials showed uncertain benefit.",
            "The evidence is mixed and depends on probiotic strain.",
            "maybe",
        ),
        (
            "MedQuAD",
            "What are the symptoms of diabetes?",
            "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, blurred vision, and slow wound healing.",
            "Diabetes symptoms include thirst, urination, fatigue, weight loss, and blurred vision.",
            "faq",
        ),
        (
            "MedQuAD",
            "How is hypertension treated?",
            "Hypertension treatment may include diet changes, lower salt intake, exercise, weight management, limiting alcohol, and medicines such as diuretics, ACE inhibitors, or calcium channel blockers.",
            "Hypertension is treated with lifestyle changes and antihypertensive medicines.",
            "faq",
        ),
        (
            "MedQA",
            "Which medication is first line therapy for anaphylaxis?",
            "Medical board explanation: intramuscular epinephrine is the first line treatment for anaphylaxis because it reverses airway edema and hypotension.",
            "Epinephrine.",
            "exam",
        ),
    ]
    samples: list[RawSample] = []
    topics = ["cardiology", "endocrinology", "pulmonology", "neurology", "infectious disease"]
    for i in range(target_size):
        dataset, question, evidence, answer, label = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        samples.append(
            RawSample(
                dataset=dataset,
                question=f"{question} ({topic} case {i + 1})",
                evidence=f"{evidence} This document is indexed as {topic} evidence item {i + 1}.",
                answer=answer,
                label=label,
            )
        )
    return samples


def load_all_datasets(max_samples_per_dataset: int | None = None) -> tuple[list[RawSample], str]:
    """Load the full dataset collection, trying sources in priority order.

    Priority: local CSVs → HuggingFace download → offline fallback.

    Args:
        max_samples_per_dataset: Maximum samples to take from each individual
            dataset source.  Defaults to ``data.max_samples_per_dataset`` from
            ``config.yaml``.

    Returns:
        Tuple of ``(samples, mode)`` where *mode* is one of
        ``"local_csv"``, ``"huggingface"``, or ``"offline_fallback"``.
    """
    if max_samples_per_dataset is None:
        max_samples_per_dataset = get_config()["data"]["max_samples_per_dataset"]

    local_samples = _load_local_csvs(max_samples_per_dataset)
    if local_samples:
        return local_samples, "local_csv"

    try:
        samples = _load_hf_datasets(max_samples_per_dataset)
        if samples:
            return samples, "huggingface"
    except Exception as exc:
        print(f"Dataset download unavailable, using fallback demo corpus: {exc}")
    return _fallback_samples(max_samples_per_dataset), "offline_fallback"


def main() -> None:
    """Smoke-test: verify local CSVs are present and report row counts."""
    data_dir = Path(__file__).parent / "data"
    medquad_rows = len(pd.read_csv(data_dir / "medquad.csv")) if (data_dir / "medquad.csv").exists() else 0
    pubmedqa_rows = len(pd.read_csv(data_dir / "pubmedqa_labeled.csv")) if (data_dir / "pubmedqa_labeled.csv").exists() else 0
    medqa_rows = len(pd.read_csv(data_dir / "medqa_usmle.csv")) if (data_dir / "medqa_usmle.csv").exists() else 0
    if medquad_rows == 0 or pubmedqa_rows == 0:
        raise SystemExit("Data loading failed: required MedQuAD/PubMedQA CSV files are missing or empty")
    print(f"MedQuAD rows: {medquad_rows}")
    print(f"PubMedQA rows: {pubmedqa_rows}")
    print(f"MedQA rows: {medqa_rows}")
    print("Data loading successful")


if __name__ == "__main__":
    main()
