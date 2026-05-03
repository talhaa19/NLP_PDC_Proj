"""Create a Colab notebook that mirrors the ordered Python files.

The generated notebook is intentionally sequential:

1. install dependencies
2. write ``config.yaml``
3. write ``A_config.py`` through ``L_main.py`` in execution order
4. run the project commands in the same order

This lets the notebook run in Colab even when the original folder structure is
not uploaded.  The source code in the notebook is copied from the current local
``.py`` files, so rerunning this script refreshes the notebook after codebase
changes.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "graphrag_colab_pipeline.ipynb"

ORDERED_FILES = [
    "A_config.py",
    "B_extract_datasets.py",
    "C_data_loader.py",
    "D_preprocess.py",
    "E_retriever.py",
    "F_llm_generator.py",
    "G_serial_rag.py",
    "H_parallel_rag.py",
    "I_evaluate.py",
    "J_write_materials.py",
    "K_convert_deliverables.py",
    "L_main.py",
]


def code_cell(source: str) -> dict:
    """Return a Jupyter code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    """Return a Jupyter Markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(path: str, content: str) -> dict:
    """Create a Colab cell that writes *content* to *path*."""
    return code_cell(f"%%writefile {path}\n{content.rstrip()}\n")


def main() -> None:
    """Generate ``graphrag_colab_pipeline.ipynb`` from current source files."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    config = (ROOT / "config.yaml").read_text(encoding="utf-8")

    cells: list[dict] = [
        markdown_cell(
            """# Distributed Agentic GraphRAG for Evidence-Grounded FAQ Assistants

This notebook is generated from the current project codebase. Run it top-to-bottom in Google Colab.

Execution order follows the renamed Python files exactly:

`A_config.py -> B_extract_datasets.py -> C_data_loader.py -> D_preprocess.py -> E_retriever.py -> F_llm_generator.py -> G_serial_rag.py -> H_parallel_rag.py -> I_evaluate.py -> J_write_materials.py -> K_convert_deliverables.py -> L_main.py`

Recommended runtime: Colab T4 GPU. The local Hugging Face generator can use the available runtime; if model loading fails, the code falls back to the rule-based generator.
"""
        ),
        markdown_cell("## 0. Install Dependencies"),
        writefile_cell("requirements.txt", requirements),
        code_cell("!pip install -q -r requirements.txt\n"),
        markdown_cell("## 1. Project Configuration"),
        writefile_cell("config.yaml", config),
    ]

    for index, filename in enumerate(ORDERED_FILES, start=2):
        content = (ROOT / filename).read_text(encoding="utf-8")
        cells.append(markdown_cell(f"## {index}. `{filename}`"))
        cells.append(writefile_cell(filename, content))

    cells.extend(
        [
            markdown_cell("## Run Step-by-Step Pipeline"),
            code_cell("!python B_extract_datasets.py\n"),
            code_cell("!python C_data_loader.py\n"),
            code_cell("!python D_preprocess.py\n"),
            code_cell("!python E_retriever.py\n"),
            code_cell("!python G_serial_rag.py\n"),
            code_cell("!python H_parallel_rag.py\n"),
            code_cell("!python I_evaluate.py\n"),
            code_cell("!python J_write_materials.py\n"),
            code_cell("!python K_convert_deliverables.py\n"),
            markdown_cell("## Optional: Run Full Pipeline Entry Point"),
            code_cell("!python L_main.py\n"),
            markdown_cell("## Inspect Results"),
            code_cell(
                """import pandas as pd
from pathlib import Path

results_dir = Path("report_results")
display(pd.read_csv(results_dir / "latency_results.csv"))
display(pd.read_csv(results_dir / "citation_results.csv"))
"""
            ),
            markdown_cell("## Download Results ZIP"),
            code_cell(
                """import zipfile
from pathlib import Path

zip_path = Path("graphrag_colab_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in [Path("data"), Path("report_results")]:
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path)
    for path in Path(".").glob("*.py"):
        zf.write(path, path)
    zf.write("config.yaml", "config.yaml")
    zf.write("requirements.txt", "requirements.txt")

print("Created:", zip_path)

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    print("Download manually:", zip_path)
"""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Created {NOTEBOOK}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
