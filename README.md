# Agentic Task: Research Paper Entity Extraction & Citation Analysis

A benchmark for evaluating LLM agentic reasoning on data-driven tasks. Designed to test **Gemini 3 Pro Preview** headroom — the task requires multi-step reasoning, entity resolution under ambiguity, and graph-based anomaly detection that weaker models struggle with.

## Task Overview

Given a synthetic corpus of ~100 research papers with metadata, citations, and author affiliations, the agent must:

1. **Extract and resolve entities** — authors, institutions, topics — handling name variations, typos, and ambiguous identifiers (e.g., "J. Smith" at MIT vs. "J. Smith" at Oxford are different people)
2. **Analyze the citation network** — build a directed graph, compute PageRank, detect orphan/self-citations
3. **Detect anomalies** — citation rings (circular citation patterns), temporal anomalies (paper from 2021 citing a 2023 paper), conflicting affiliations
4. **Produce a structured report** — with strict schema, validation checks, and summary statistics

## Project Structure

```
├── pyproject.toml                  # pip-installable package config
├── requirements.txt
├── TASK.md                         # original task specification
├── AGENTIC_PROMPT_GUIDELINES.md    # prompt design guidelines
└── benchmark/
    ├── config.py                   # constants, thresholds, model list
    ├── agent_colab.ipynb           # agent evaluation notebook (Colab Pro)
    ├── golden_solution.ipynb       # reference solution notebook
    ├── benchmark_prompt.md         # full benchmark prompt (human-readable)
    ├── data/
    │   ├── generate.py             # synthetic dataset generation
    │   └── loader.py               # data loading utilities
    ├── pipeline/
    │   ├── resolution.py           # fuzzy matching, entity disambiguation
    │   ├── extraction.py           # author/institution/topic extraction
    │   ├── citations.py            # graph construction, PageRank, anomaly detection
    │   ├── validation.py           # validation checks, summary statistics
    │   ├── report.py               # final report builder
    │   └── runner.py               # end-to-end pipeline orchestrator
    ├── evaluation/
    │   ├── prompt.py               # benchmark prompt (importable string)
    │   ├── agent.py                # model selection, code extraction, execution
    │   └── tests.py                # 23 unit tests across 8 test classes
    └── dataset/
        └── generate_dataset.py     # CLI entry point for data generation
```

## Installation

```bash
pip install git+https://github.com/EhsanKA/agentic_task.git
```

## Quick Start

### Run the golden solution locally

```python
from benchmark.data.loader import setup_data
from benchmark.pipeline.runner import run_pipeline
from benchmark.evaluation.tests import set_context, run_all_tests

papers, citations, affiliations, _ = setup_data()
results = run_pipeline(papers, citations, affiliations)

set_context(results)
run_all_tests()  # 23 tests
```

### Run on Google Colab

Upload and run either notebook:

- **`golden_solution.ipynb`** — reference implementation, proves task solvability
- **`agent_colab.ipynb`** — sends the task to Gemini 3 Pro Preview via `google.colab.ai`

Both notebooks install the package, generate data, and run the shared test suite automatically.

## Headroom Design

The benchmark is designed so that **Gemini 3 Pro Preview passes all tests** while weaker models fail on the advanced challenges:

| Challenge | What it tests |
|-----------|---------------|
| Ambiguous authors | "J. Smith" at MIT ≠ "J. Smith" at Oxford — requires institution-aware disambiguation |
| Typo correction | "Jonh Smith" → "John Smith" — requires fuzzy matching (Levenshtein distance) |
| Citation rings | Circular citation patterns (A→B→C→D→E→A) — requires cycle detection |
| Temporal anomalies | Paper from 2021 citing a 2023 paper — requires year comparison logic |
| Venue normalization | "NIPS" → "NeurIPS" (renamed in 2018) |
| Affiliation conflicts | Author listed at wrong institution — requires cross-referencing |

## Test Suite

23 deterministic unit tests organized into 8 classes:

- `TestDataLoading` — schema and shape validation
- `TestEntityExtraction` — authors and institutions extracted
- `TestEntityResolution` — resolution maps populated correctly
- `TestCitationNetwork` — graph, PageRank, orphans, self-citations
- `TestHeadroomChallenges` — rings, temporal anomalies, typos, disambiguation
- `TestValidationResults` — all validation keys present and passing
- `TestSummaryStats` — required statistics computed
- `TestFinalReport` — report schema, anomaly section, all checks passed

## Dependencies

- `pandas` — data manipulation
- `networkx` — citation graph analysis, PageRank, cycle detection
- `python-Levenshtein` — fuzzy string matching for typo correction
- `numpy` — numeric operations
