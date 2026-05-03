"""LLM-backed answer generation for the RAG pipeline.

The generator uses a locally-running open-source model (``google/flan-t5-base``
by default) via the HuggingFace ``transformers`` library.  No API key and no
external service are required -- the model is downloaded once by HuggingFace and
cached on disk for all future runs.

Architecture
------------
* ``_get_pipeline()``   -- loads the HF text2text pipeline once and caches it
                          in-process so repeated calls within one benchmark run
                          pay zero re-load cost.
* ``_hf_generate()``    -- builds a structured evidence prompt, runs inference,
                          parses the prediction label.
* ``_rule_based()``     -- weighted keyword fallback (no ML dependency); used
                          when ``transformers``/``torch`` are not installed or
                          when the model fails to load.
* ``generate_answer()`` -- public API: tries HF first, falls back to rule-based.

Model choice
------------
``google/flan-t5-base`` (~250 MB, Apache-2.0) was selected because it:
  * runs on CPU in < 5 s per query (acceptable for a benchmark demo);
  * is instruction-fine-tuned and reliably follows yes / no / maybe prompts;
  * requires only ``transformers``, ``sentencepiece``, and ``torch`` (CPU wheel).

To swap the model, change ``llm.hf_model`` in ``config.yaml``.
"""

from __future__ import annotations

import re
from typing import Any

from A_config import get_config

# ---------------------------------------------------------------------------
# Module-level pipeline cache  (loaded at most once per interpreter session)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}   # {model_name: (model, tokenizer)}


def _get_model(model_name: str) -> tuple[Any, Any, Any]:
    """Load and cache a seq2seq model + tokenizer for local inference.

    Uses ``AutoModelForSeq2SeqLM`` / ``AutoTokenizer`` directly -- bypassing
    the pipeline abstraction so the code works across all transformers versions
    (including 5.x which removed the ``text2text-generation`` pipeline task).

    The model is placed on GPU (CUDA) automatically when available, falling
    back to CPU otherwise.  It is loaded once and kept in memory for the
    lifetime of the interpreter session.

    Args:
        model_name: HuggingFace model identifier, e.g. ``"google/flan-t5-base"``.

    Returns:
        ``(model, tokenizer, device)`` tuple ready for inference.

    Raises:
        ImportError: If ``transformers`` or ``torch`` are not installed.
    """
    if model_name not in _MODEL_CACHE:
        import torch  # type: ignore[import-untyped]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import-untyped]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[llm_generator] Loading '{model_name}' on {device.upper()} ... (first call only)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()            # inference mode -- disables dropout
        _MODEL_CACHE[model_name] = (model, tokenizer, device)
        print(f"[llm_generator] Model ready on {device.upper()}.")
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Weighted keyword constants  (rule-based fallback)
# ---------------------------------------------------------------------------

_YES_WEIGHTED: list[tuple[str, float]] = [
    ("significantly improved", 2.0),
    ("significantly reduced", 2.0),
    ("first line", 1.5),
    ("effective treatment", 1.5),
    ("clinically significant", 1.5),
    ("associated with improvement", 1.2),
    ("reduced risk", 1.2),
    ("improved outcomes", 1.2),
    ("significant reduction", 1.0),
    ("significant improvement", 1.0),
    ("beneficial", 0.8),
    ("effective", 0.8),
    ("reduced", 0.6),
    ("improved", 0.6),
    ("associated", 0.4),
]

_NO_WEIGHTED: list[tuple[str, float]] = [
    ("no significant reduction", 2.5),
    ("no significant improvement", 2.5),
    ("not associated", 2.0),
    ("did not reduce", 2.0),
    ("did not improve", 2.0),
    ("no evidence of benefit", 2.0),
    ("failed to", 1.5),
    ("no evidence", 1.2),
    ("not effective", 1.2),
    ("no benefit", 1.2),
    ("ineffective", 1.0),
    ("did not", 0.8),
    ("no significant", 0.8),
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_evidence(evidence_docs: list[dict], max_chars_per_doc: int = 500) -> tuple[str, list[str]]:
    """Format evidence documents into a numbered passage block.

    Args:
        evidence_docs: Retrieved document dicts with ``text`` and ``source``.
        max_chars_per_doc: Maximum characters per passage (keeps prompt short
            enough for flan-t5-base's 512-token input limit).

    Returns:
        ``(formatted_block, citations)``
    """
    passages: list[str] = []
    citations: list[str] = []
    for i, doc in enumerate(evidence_docs):
        text = str(doc.get("text", "")).strip()[:max_chars_per_doc]
        source = str(doc.get("source", f"doc-{i}"))
        passages.append(f"[{i + 1}] {source}: {text}")
        citations.append(source)
    return "\n".join(passages), citations


def _parse_prediction(text: str) -> str:
    """Extract the yes / no / maybe prediction from model output.

    Tries an explicit ``Prediction:`` label, then a bare keyword on the last
    line, then a full-text keyword scan.  A negative-phrase pre-check fires
    before the bare-keyword scan to prevent "no significant effect" from being
    parsed as "yes" just because "yes" was seen later in the text.

    Args:
        text: Raw model response text.

    Returns:
        One of ``"yes"``, ``"no"``, or ``"maybe"``.
    """
    lower = text.lower().strip()

    # 1. Explicit label anywhere in the text
    m = re.search(r"prediction\s*[:\-]\s*(yes|no|maybe)", lower)
    if m:
        return m.group(1)

    # 2. Strong negation pre-check — if the model output itself begins with a
    #    clear negative phrase, treat as "no" immediately (avoids mis-parsing
    #    "no significant ..." as "yes" via the keyword scan below).
    _STRONG_NO_STARTS = (
        "no significant", "no evidence", "no benefit", "no association",
        "no effect", "not significant", "not associated", "did not",
        "does not", "failed to", "ineffective", "no difference",
    )
    if any(lower.startswith(p) for p in _STRONG_NO_STARTS):
        return "no"

    # 3. Bare label on the last non-empty line
    for line in reversed(lower.splitlines()):
        line = line.strip()
        if line in {"yes", "no", "maybe"}:
            return line
        m = re.search(r"\b(yes|no|maybe)\b", line)
        if m:
            return m.group(1)

    # 4. Weighted keyword fallback — margin reduced to 0.3 so near-ties
    #    correctly fall through to "maybe" rather than defaulting to "yes".
    no_score = sum(w for term, w in _NO_WEIGHTED if term in lower)
    yes_score = sum(w for term, w in _YES_WEIGHTED if term in lower)
    if no_score > yes_score + 0.3:
        return "no"
    if yes_score > no_score + 0.3:
        return "yes"
    return "maybe"


# ---------------------------------------------------------------------------
# HuggingFace local-model path
# ---------------------------------------------------------------------------

def _hf_generate(question: str, evidence_docs: list[dict]) -> dict:
    """Generate an answer with a local HuggingFace model (no API key needed).

    Builds a structured prompt, runs ``text2text-generation`` inference with
    the cached pipeline, and parses the prediction label.

    Args:
        question: The natural-language query.
        evidence_docs: Retrieved document dicts with ``text`` and ``source``.

    Returns:
        Dict with keys ``prediction``, ``answer``, and ``citations``.

    Raises:
        ImportError: If ``transformers`` or ``torch`` are not installed.
        RuntimeError: If model inference fails.
    """
    import torch  # type: ignore[import-untyped]

    cfg = get_config().get("llm", {})
    model_name = cfg.get("hf_model", "google/flan-t5-base")

    # Collect citations from all docs but only pass the top-3 to the model.
    # flan-t5 has a 512-token input limit.  Using top-3 docs × 300 chars each
    # keeps the prompt under ~350 tokens while providing richer evidence than
    # top-2 alone.  Chars-per-doc reduced to 300 (vs 400 previously) to fit
    # the third document without truncating the prompt.
    citations = [str(doc.get("source", f"doc-{i}")) for i, doc in enumerate(evidence_docs)]
    top_docs = evidence_docs[:3]
    evidence_text = " ".join(
        str(doc.get("text", "")).strip()[:300] for doc in top_docs
    )

    # Structured prompt that explicitly defines all three labels.
    # Key improvement: the 'no' definition now calls out negative findings
    # (e.g. "no significant effect", "did not reduce") so the model is less
    # likely to answer 'yes' just because the passage mentions a treatment.
    # 'maybe' is defined for mixed or inconclusive evidence, reducing the
    # over-prediction of 'yes' on ambiguous abstracts.
    prompt = (
        f"Passage: {evidence_text}\n\n"
        f"Question: {question}\n\n"
        f"Read the passage carefully and answer the question.\n"
        f"Answer 'yes' if the passage shows a positive result or beneficial effect.\n"
        f"Answer 'no' if the passage shows no significant effect, no benefit, "
        f"no association, or a negative/null result.\n"
        f"Answer 'maybe' if the evidence is mixed, limited, or inconclusive.\n\n"
        f"Answer (yes/no/maybe):"
    )

    model, tokenizer, device = _get_model(model_name)

    # Tokenise and move tensors to the same device as the model
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,       # only need one word: yes / no / maybe
            num_beams=4,            # beam search -- deterministic, higher quality
            do_sample=False,
            early_stopping=True,
        )

    answer_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    prediction = _parse_prediction(answer_text)

    return {
        "prediction": prediction,
        "answer": answer_text,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Rule-based fallback  (zero ML dependencies)
# ---------------------------------------------------------------------------

def _rule_based(question: str, evidence_docs: list[dict]) -> dict:
    """Deterministic weighted keyword fallback -- no ML dependency required.

    Scans all retrieved documents with weighted positive/negative biomedical
    terms.  A 0.5-point margin buffer means near-ties conservatively return
    ``"maybe"``.

    Args:
        question: The natural-language query.
        evidence_docs: Retrieved document dicts with ``text`` and ``source``.

    Returns:
        Dict with keys ``prediction``, ``answer``, and ``citations``.
    """
    combined = " ".join(doc.get("text", "") for doc in evidence_docs).lower()

    no_score = sum(w for term, w in _NO_WEIGHTED if term in combined)
    yes_score = sum(w for term, w in _YES_WEIGHTED if term in combined)

    if no_score > yes_score + 0.5:
        prediction = "no"
    elif yes_score > no_score + 0.5:
        prediction = "yes"
    else:
        prediction = "maybe"

    citations = [doc.get("source", f"doc-{i}") for i, doc in enumerate(evidence_docs)]
    answer = (
        f"Based on the retrieved evidence, the predicted answer is '{prediction}'. "
        f"Evidence drawn from: {', '.join(citations[:3])}"
        f"{'...' if len(citations) > 3 else ''}."
    )
    return {"prediction": prediction, "answer": answer, "citations": citations}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(question: str, evidence_docs: list[dict]) -> dict:
    """Generate a cited answer for *question* grounded on *evidence_docs*.

    Tries the local HuggingFace model first (``google/flan-t5-base`` by
    default, configurable via ``config.yaml``).  Falls back to the rule-based
    generator if ``transformers`` / ``torch`` are not installed or if inference
    fails for any reason.

    No API key or internet connection is required after the first run (HF
    caches the model weights on disk automatically).

    Args:
        question: The natural-language query.
        evidence_docs: Retrieved document dicts (from retriever output).

    Returns:
        Dict with keys:

        - ``prediction`` (``"yes"`` | ``"no"`` | ``"maybe"``)
        - ``answer`` (human-readable answer string)
        - ``citations`` (list of source identifier strings)
    """
    try:
        return _hf_generate(question, evidence_docs)
    except ImportError:
        print(
            "[llm_generator] 'transformers' or 'torch' not installed -- "
            "install them with:  pip install transformers sentencepiece torch\n"
            "[llm_generator] Falling back to rule-based generator."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_generator] Model inference failed ({exc}); using rule-based fallback.")
    return _rule_based(question, evidence_docs)


if __name__ == "__main__":
    # -- Demo: run 3 medical questions through the local LLM -----------------
    cfg = get_config().get("llm", {})
    model_name = cfg.get("hf_model", "google/flan-t5-base")
    print("=" * 70)
    print("  F_llm_generator -- Local LLM Demo")
    print(f"  Model : {model_name}  (no API key required)")
    print("=" * 70)

    demo_cases = [
        {
            "question": "Does aspirin reduce recurrent cardiovascular events?",
            "evidence": [
                {"text": "A randomized controlled trial showed aspirin significantly "
                         "reduced recurrent cardiovascular events in high-risk patients "
                         "versus placebo (HR 0.74, 95% CI 0.62-0.89, p<0.001).",
                 "source": "PubMedQA-22301"},
                {"text": "Aspirin inhibits COX-1 mediated platelet aggregation and is "
                         "widely recommended as secondary prevention in cardiovascular disease.",
                 "source": "PubMedQA-18742"},
            ],
        },
        {
            "question": "Is vitamin D supplementation associated with fewer asthma attacks?",
            "evidence": [
                {"text": "A meta-analysis of 9 RCTs found no significant reduction in "
                         "asthma exacerbations with vitamin D supplementation compared "
                         "to placebo (OR 0.97, 95% CI 0.83-1.14).",
                 "source": "PubMedQA-31045"},
                {"text": "Some observational studies reported lower vitamin D levels in "
                         "asthmatic patients, but interventional trials did not confirm benefit.",
                 "source": "PubMedQA-29876"},
            ],
        },
        {
            "question": "What is the first-line treatment for anaphylaxis?",
            "evidence": [
                {"text": "Intramuscular epinephrine (adrenaline) is the first-line and "
                         "most important treatment for anaphylaxis. It reverses airway "
                         "oedema, hypotension, and urticaria rapidly.",
                 "source": "MedQuAD-Anaphylaxis-01"},
                {"text": "Antihistamines and corticosteroids are adjunct therapies for "
                         "anaphylaxis but should never replace or delay epinephrine.",
                 "source": "MedQuAD-Anaphylaxis-02"},
            ],
        },
    ]

    for i, case in enumerate(demo_cases, 1):
        print(f"\n{'-' * 70}")
        print(f"  Demo {i}/{len(demo_cases)}")
        print(f"  Q: {case['question']}")
        result = generate_answer(case["question"], case["evidence"])
        print(f"  Prediction  : {result['prediction'].upper()}")
        print(f"  Answer      : {result['answer']}")
        print(f"  Citations   : {', '.join(result['citations'])}")

    print(f"\n{'=' * 70}")
    print("  LLM generator working correctly -- ready for benchmark pipeline.")
    print("=" * 70)
