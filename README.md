# LLM Benchmark Unified Framework

A provider-neutral benchmarking pipeline for evaluating large language models across multiple datasets and evidence strategies. Built as a team project at the University of Mannheim (FSS 2026).

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Datasets](#datasets)
- [Evidence Strategies](#evidence-strategies)
- [Evaluation Strategies](#evaluation-strategies)
- [Supported Models](#supported-models)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Overview

The pipeline lets you benchmark any OpenRouter-compatible LLM against a dataset with a single command. It handles evidence retrieval, prompt construction, model calls, evaluation, and result logging automatically.

Two execution modes are supported per dataset:

- **`llm_only`** — evidence is retrieved and passed directly to the model as text context
- **`sql_detour`** — model generates SQL, which is executed against a SQLite database; the result is then scored against ground truth

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nishitsuthar/LLM_Benchmark_Unified_Framework.git
cd LLM_Benchmark_Unified_Framework
```

### 2. Create and activate a virtual environment

Using a virtual environment prevents conflicts with packages already installed on your system.

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Copy the example env file and fill in your OpenRouter API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
MODEL_PROVIDER=openrouter
MODEL_API_KEY=sk-or-v1-...
MODEL_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=meta-llama/llama-3.3-70b-instruct
```

> **Note:** `.env` is gitignored and must never be committed.

### 5. Large evidence files (download separately)

The following files are too large for git and must be placed locally before running their respective datasets:

| File | Dataset | Size |
|------|---------|------|
| `evidence/hotpot/hotpot_validation_distractor.json` | `hotpot` | ~50 MB |
| `evidence/imdb_20/imdb_20_movies_screenshot_evidence.pdf` | `imdb_20` | ~250 MB |

---

## Configuration

All configuration is in `.env`. The key fields are:

| Variable | Description |
|----------|-------------|
| `MODEL_PROVIDER` | Always `openrouter` |
| `MODEL_API_KEY` | Your OpenRouter API key |
| `MODEL_BASE_URL` | `https://openrouter.ai/api/v1` |
| `MODEL_NAME` | Default model (overridden by `--model` flag) |

---

## Running the Pipeline

```bash
python3 run.py --dataset <dataset> --model <model>
```

An interactive menu appears to select a prompt style. Press `1` for the default (dataset-specific zero-shot).

### Options

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset name matching a file in `datasets/` |
| `--model` | Model alias (see table below) or raw OpenRouter model ID |
| `--task_ids` | Run only specific questions, e.g. `--task_ids T01 T05` |

### Examples

```bash
# Run imdb_controlled with gemini_flash (llm_only, interactive prompt selection)
python3 run.py --dataset imdb_controlled --model gemini_flash

# Run f1_racing SQL detour with mistral_small, single question
python3 run.py --dataset f1_racing --model mistral_small --task_ids T01

# Run hotpot multi-hop QA with nemotron
python3 run.py --dataset hotpot --model nemotron
```

Results are written to `results/<model>/<dataset>_<prompt>_metrics.csv` after each question (crash-safe).

---

## Datasets

| Dataset | Domain | Questions | Evidence | Mode |
|---------|--------|-----------|----------|------|
| `imdb_controlled` | Movies (structured) | 30 | CSV full-table | llm_only + sql_detour |
| `imdb_20` | Movies (visual PDF) | 10 | Visual PDF screenshots | llm_only |
| `f1_racing` | Formula 1 racing | 30 | Prose documents + SQLite DB | llm_only + sql_detour |
| `music` | Music artists | 30 | Prose documents + SQLite DB | llm_only + sql_detour |
| `hotpot` | Multi-hop QA | 124 | Wikipedia paragraphs | llm_only |
| `hotpot_sql` | Multi-hop QA (structured) | 124 | SQLite fact database | llm_only |
| `finhybrid` | Finance (PDF) | 31 | RAG over PDF | llm_only |
| `tathybrid` | Tables (PDF) | 29 | RAG over PDF | llm_only |
| `music_structured` | Music (PDF) | 16 | RAG over PDF | llm_only |

---

## Evidence Strategies

| Key | Class | Description |
|-----|-------|-------------|
| `csv_full` | `CsvFullEvidenceBuilder` | Passes the entire CSV table as text context |
| `prose_docs` | `ProseEvidenceBuilder` | Retrieves relevant prose document chunks |
| `pdf_rag` | `RagEvidenceBuilder` | RAG retrieval over PDF documents using ChromaDB |
| `pdf_visual` | `VisualPdfEvidenceBuilder` | Passes PDF pages as base64 images to vision models |
| `hotpot_context` | `HotpotEvidenceBuilder` | Looks up 10 Wikipedia paragraphs from HotpotQA distractor JSON |
| `hotpot_sql` | `HotpotSqlEvidenceBuilder` | Queries a pre-built SQLite fact database of HotpotQA triples |

---

## Evaluation Strategies

| Key | Class | Description |
|-----|-------|-------------|
| `table_f1` | `TableF1Evaluator` | Token-level F1 between predicted and ground-truth tables |
| `exact_match` | `ExactMatchEvaluator` | Exact string match after normalization |
| `span_f1` | `SpanF1Evaluator` | Token-overlap F1 for extractive spans |
| `partial_credit` | `PartialCreditEvaluator` | Per-question partial credit scoring |
| `hotpot_f1` | `HotpotEvaluator` | Standard HotpotQA token-overlap F1 (strips articles, punctuation) |
| `sql_scalar_match` | `SqlScalarEvaluator` | Numeric/string match for single-value SQL results |

---

## Supported Models

| Alias | OpenRouter Model ID |
|-------|-------------------|
| `gemini_flash` | `google/gemini-3.1-flash-lite` |
| `mistral_small` | `mistralai/mistral-small-2603` |
| `nemotron` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `llama_70b` | `meta-llama/llama-3.3-70b-instruct` |
| `deepseek_r1` | `deepseek/deepseek-r1` |
| `qwen_72b` | `qwen/qwen-2.5-72b-instruct` |

Any raw OpenRouter model ID can also be passed directly via `--model`.

---

## Project Structure

```
.
├── run.py                          # Main pipeline entry point
├── requirements.txt
├── .env.example                    # Copy to .env and fill in your API key
├── datasets/                       # Dataset YAML configs
│   ├── imdb_controlled.yaml
│   ├── f1_racing.yaml
│   ├── hotpot.yaml
│   └── ...
├── prompts/                        # Prompt templates (.txt)
│   ├── imdb_controlled_llm_only_zero_shot.txt
│   ├── f1_racing_sql_detour_zero_shot.txt
│   └── ...
├── evidence/                       # Evidence files per dataset
│   ├── imdb_controlled/
│   ├── f1_racing/
│   ├── hotpot/
│   └── ...
├── ground_truth/                   # Ground truth answers per dataset
│   ├── imdb_controlled/
│   ├── hotpot/
│   └── ...
├── results/                        # Output CSVs (per model, per dataset)
│   ├── gemini_flash/
│   ├── mistral_small/
│   └── nemotron/
├── unified_pipeline/
│   ├── config.py                   # Registry: evidence builders + evaluators + model aliases
│   ├── model_router.py             # OpenRouter API client + SQL utilities
│   ├── reporter.py                 # CSV result writer
│   ├── base.py                     # Base classes
│   ├── evidence/                   # Evidence builder implementations
│   └── evaluators/                 # Evaluator implementations
├── data/
│   └── hotpot_facts.db             # Pre-built SQLite fact DB for hotpot_sql
└── scripts/
    └── rescore_imdb20.py           # Offline rescoring utility for imdb_20
```

---

## Team

| Member | Dataset | Evidence Strategy |
|--------|---------|------------------|
| Nishit | `finhybrid`, `tathybrid`, `music_structured` | RAG (ChromaDB + sentence-transformers) |
| Krittika | `imdb_controlled` | CSV full-table + SQL detour |
| Arpitha | `imdb_20` | Visual PDF (base64 image) |
| Satwik | `f1_racing`, `music` | Prose documents + SQL detour |
| Sushma | `hotpot`, `hotpot_sql` | HotpotQA context + SQL fact DB |
