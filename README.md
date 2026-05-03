# Distributed Agentic GraphRAG for Evidence-Grounded FAQ Assistants

Course-scale medical RAG benchmark comparing **serial retrieval** with **two-level parallel retrieval**. The system uses MedQuAD, PubMedQA, and MedQA, builds a searchable evidence corpus, runs TF-IDF and BM25-lite retrievers in parallel, generates cited answers (rule-based or via Claude API), and reports latency, speedup, efficiency, accuracy, citation coverage, and retrieval hit rate.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Run Order](#run-order)
6. [Enabling LLM Generation](#enabling-llm-generation)
7. [Outputs](#outputs)
8. [Architecture](#architecture)
9. [Module Reference](#module-reference)
10. [Tests](#tests)
11. [Main Results](#main-results)
12. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
project/
├── A_config.py               # Central YAML config loader
├── B_extract_datasets.py     # One-time dataset extraction to CSV
├── C_data_loader.py          # Multi-source dataset loader with fallback
├── D_preprocess.py           # Corpus and query builder
├── E_retriever.py            # TF-IDF + BM25-lite retrievers (with joblib cache)
├── F_llm_generator.py        # Answer generator: Claude API or rule-based fallback
├── G_serial_rag.py           # Serial baseline RAG pipeline
├── H_parallel_rag.py         # Parallel RAG (retriever-level + query-level)
├── I_evaluate.py             # Benchmark runner + metrics + charts
├── J_write_materials.py      # Report and slide Markdown writer
├── K_convert_deliverables.py # Markdown → DOCX / PDF / PPTX
├── L_main.py                 # Full pipeline entry point
├── M_create_colab_notebook.py# Generates a runnable Colab notebook
├── config.yaml               # All tunable parameters (single source of truth)
├── pyproject.toml            # Package definition + entry points
├── requirements.txt          # All runtime + dev dependencies
├── .gitignore
├── data/                     # Raw + processed CSVs
│   ├── pubmedqa_labeled.csv
│   ├── medquad.csv
│   ├── medqa_usmle.csv
│   └── processed_corpus.csv
├── report_results/           # Generated artifacts (charts, CSVs, report, slides)
└── tests/                    # pytest test suite
```

---

## Datasets

| Dataset | File | Rows | Purpose |
|---|---|---:|---|
| MedQuAD | `data/medquad.csv` | 47,441 | Medical FAQ evidence corpus |
| PubMedQA | `data/pubmedqa_labeled.csv` | 1,000 | Biomedical yes/no/maybe evaluation |
| MedQA USMLE | `data/medqa_usmle.csv` | 10,178 | Exam-style QA corpus/evaluation |

If the CSV files are missing, regenerate them:

```bash
# Downloads PubMedQA from GitHub + MedQuAD/MedQA from HuggingFace
python B_extract_datasets.py

# If you have a local copy of ori_pqal.json (avoids the GitHub download):
python B_extract_datasets.py --local-pubmedqa /path/to/ori_pqal.json
```

---

## Installation

```bash
# Install runtime + test dependencies
python -m pip install -r requirements.txt

# Or install as an editable package (adds the `rag-benchmark` command)
python -m pip install -e .
```

Requires **Python ≥ 3.10**.

---

## Configuration

All tunable parameters live in `config.yaml` — **never edit source files to change a parameter**.

```yaml
data:
  max_samples_per_dataset: 700   # Rows loaded from each dataset source
  corpus_limit: 1500             # Documents indexed by retrievers

retrieval:
  shard_delay_sec: 0.004         # Simulated distributed-index latency per search
  top_k: 5                       # Retrieved documents per retriever
  cache_dir: ".retriever_cache"  # joblib cache directory (auto-created)
  tfidf_word:
    analyzer: "word"
    ngram_range: [1, 2]
  tfidf_char:
    analyzer: "char_wb"
    ngram_range: [3, 5]
  bm25:
    k1: 1.5
    b: 0.75

benchmark:
  query_set_sizes: [100, 250, 500]
  worker_counts: [1, 2, 4]
  main_max_samples: 500
  use_batch_parallel: false      # true → query-level parallelism in I_evaluate.py

llm:
  enabled: false                 # Set true + export ANTHROPIC_API_KEY to use Claude
  model: "claude-haiku-4-5-20251001"
  max_tokens: 256
  temperature: 0.0
```

---

## Run Order

### Full pipeline (recommended)

```bash
python L_main.py
```

### Step-by-step

```bash
python C_data_loader.py          # Verify CSVs exist and report row counts
python D_preprocess.py           # Build data/processed_corpus.csv
python E_retriever.py            # Smoke-test: run one sample TF-IDF query
python G_serial_rag.py           # Run 100-query serial baseline
python H_parallel_rag.py         # Run parallel benchmark across worker counts
python I_evaluate.py             # Compute speedup, efficiency, metrics + charts
python J_write_materials.py      # Write final_report.md + presentation_slides.md
python K_convert_deliverables.py # Convert Markdown → DOCX, PDF, PPTX
```

---

## Enabling LLM Generation

By default the pipeline uses a fast rule-based keyword classifier so the benchmark runs without any API key. To switch to real Claude-backed generation:

1. Set `llm.enabled: true` in `config.yaml`.
2. Export your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Run normally. `F_llm_generator.py` will call the Claude API; if the call fails for any reason it automatically falls back to the rule-based generator and logs a warning.

> **Note:** LLM generation adds per-query API latency, which is intentionally excluded from the PDC speedup measurement when `llm.enabled` is `false`.

---

## Outputs

| File | Description |
|---|---|
| `data/processed_corpus.csv` | Cleaned RAG corpus (doc_id, text, source, label) |
| `report_results/serial_results.csv` | Per-query serial latency + predictions |
| `report_results/parallel_results.csv` | Per-query parallel latency + predictions |
| `report_results/latency_results.csv` | Speedup / efficiency summary table |
| `report_results/citation_results.csv` | Accuracy / citation coverage / hit rate |
| `report_results/sample_outputs.csv` | 5 sample Q&A pairs with citations |
| `report_results/speedup_chart.png` | Speedup vs queries line chart |
| `report_results/efficiency_chart.png` | Efficiency vs queries line chart |
| `report_results/final_report.md` | Full academic report (Markdown) |
| `report_results/final_report.docx` | Word version of the report |
| `report_results/final_report.pdf` | PDF version of the report |
| `report_results/presentation_slides.pptx` | PowerPoint presentation |

---

## Architecture

```
                  ┌─────────────────────────────────────────┐
                  │  PubMedQA + MedQuAD + MedQA CSV files   │
                  └──────────────────┬──────────────────────┘
                                     │ C_data_loader.py
                                     ▼
                          D_preprocess.py
                    (build_corpus, build_queries)
                                     │
                   ┌─────────────────┼──────────────────────┐
                   │                 │                       │
             TF-IDF(word)      TF-IDF(char_wb)          BM25-lite
             E_retriever        E_retriever             E_retriever
             (joblib cache)    (joblib cache)           (joblib cache)
                   │                 │                       │
                   └─────────────────┼───────────────────────┘
                                     │ merge_results (RRF)
                                     ▼
                              critic_check()
                       (H_parallel_rag.py)
                                     │
                              F_llm_generator.py
                    (Claude API  or  rule-based fallback)
                                     │
                          ┌──────────┴──────────┐
                          │   citations + answer │
                          └──────────┬──────────┘
                                     │
                    I_evaluate.py  ──┘
          (latency, speedup, efficiency, hit rate, accuracy)
                                     │
               J_write_materials.py + K_convert_deliverables.py
                    (Markdown → DOCX / PDF / PPTX)

Serial path  : G_serial_rag.py  → retrievers called sequentially
Parallel path: H_parallel_rag.py→ retrievers called via ThreadPoolExecutor
               (+ batch_answer for query-level concurrency when use_batch_parallel=true)
```

---

## Module Reference

| Module | Class / Key Function | Role |
|---|---|---|
| `A_config.py` | `get_config()` | Load + cache `config.yaml` |
| `B_extract_datasets.py` | `extract_pubmedqa/medquad/medqa()` | One-time CSV extraction |
| `C_data_loader.py` | `load_all_datasets()` | Local CSV → HF → fallback |
| `D_preprocess.py` | `build_corpus()`, `build_queries()` | Corpus + query construction |
| `E_retriever.py` | `TfidfRetriever`, `KeywordBM25LiteRetriever`, `merge_results()` | Retrieval + RRF fusion |
| `F_llm_generator.py` | `generate_answer()` | LLM or rule-based generation |
| `G_serial_rag.py` | `SerialRAG` | Sequential baseline |
| `H_parallel_rag.py` | `ParallelRAG`, `critic_check()` | Concurrent retrieval pipeline |
| `I_evaluate.py` | `run_experiments()` | Benchmark matrix + metrics |
| `J_write_materials.py` | `write_report_and_presentation()` | Markdown report + slides |
| `K_convert_deliverables.py` | `markdown_to_docx/pdf/pptx()` | Document format conversion |
| `L_main.py` | `main()` | Full pipeline orchestration |

---

## Tests

```bash
# Run all tests
python -m pytest

# With coverage report
python -m pytest --cov=. --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_retriever.py -v
```

Tests live in `tests/` and cover: corpus building, query construction, TF-IDF retrieval, BM25 retrieval, RRF merge, joblib cache hits, rule-based generation, critic check, score metrics, config loading, and dataset extraction.

---

## Main Results

Best observed result from the 250-query benchmark with 2 workers:

```
Serial latency  : 8.41 sec
Parallel latency: 4.66 sec
Speedup         : 1.80×
Efficiency      : 0.90
Citation coverage: 100 %
Retrieval hit rate: 98.4 %
```

Four workers did not outperform two because the workload is lightweight and thread-management overhead becomes dominant — consistent with Amdahl's Law.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/processed_corpus.csv` | Run `python D_preprocess.py` first |
| `ModuleNotFoundError: yaml` | Run `pip install pyyaml` |
| `ModuleNotFoundError: joblib` | Run `pip install joblib` |
| `ModuleNotFoundError: anthropic` | Run `pip install anthropic` |
| HuggingFace download fails | CSVs in `data/` are already provided; re-run `python B_extract_datasets.py` when online |
| LLM generation not activating | Ensure `llm.enabled: true` in `config.yaml` **and** `ANTHROPIC_API_KEY` is exported |
| Slow benchmark on first run | Retriever indices are being built and cached to `.retriever_cache/`; subsequent runs are faster |
| Tests fail with `ImportError` | Run `pip install -e .` to make sure the project root is on `sys.path` |
