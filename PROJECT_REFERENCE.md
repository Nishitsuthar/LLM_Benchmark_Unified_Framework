# LLM Benchmark Unified Framework — Complete Project Reference

**Version:** Sprint 5  
**University:** University of Mannheim, FSS 2026  
**Repository:** https://github.com/Nishitsuthar/LLM_Benchmark_Unified_Framework  
**Project Lead:** Nishit Suthar (I772947)

---

## 1. Project Background

This is a university team project benchmarking how well large language models (LLMs) answer structured and unstructured questions across multiple datasets. The project ran across 5 sprints:

- **Sprints 1–2:** Problem definition, dataset selection, tooling setup
- **Sprint 3:** Each team member independently built their own pipeline for their own dataset
- **Sprint 4:** Krittika built a robust, production-quality pipeline for the IMDb 878-movie dataset. Others refined their Sprint 3 pipelines.
- **Sprint 5 (current):** Merge all 4 independent pipelines into one unified, modular framework that can run any dataset through any model under any prompting strategy and produce results in a consistent, comparable format.

The Sprint 4 archive (Krittika's work) lives at:
```
/Users/I772947/personal work/LLM Benchmark Team Project/imdb-controlled-experiments/
```
This folder is **read-only reference** — it is never modified. The unified framework is built separately.

---

## 2. The Problem Sprint 5 Solves

Before Sprint 5, the team had 4 completely separate codebases:

| Person | Dataset | Evidence Format | Evaluation |
|---|---|---|---|
| Krittika | IMDb 878 movies (30 tasks) | Full CSV in-context | Row-level precision/recall/F1 |
| Satwik | Music (30Q), F1 Racing (55Q) | Prose text documents | Normalised exact match |
| Nishit | UDA-Benchmark (NQ-text sub-dataset) | PDF → RAG (ChromaDB + MiniLM) | Token-level span F1 |
| Arpitha | IMDb 20 movies (10 questions) | PDF pages as images | Manual scoring: 1.0/0.5/0.0 |

Problems with 4 separate codebases:
- Results are not comparable — different output formats, different metrics
- Running a new model requires editing 4 separate files
- No shared reporter — each person's crash behaviour differs
- No single command to reproduce all experiments

Sprint 5 solution: one framework, one command, one output format, zero code changes to add a new dataset or model.

---

## 3. Repository Location and Structure

The unified framework lives at:
```
/Users/I772947/personal work/LLM Benchmark Team Project/LLM_Benchmark_Unified_Framework/
```

This folder IS the git repository connected to:
```
https://github.com/Nishitsuthar/LLM_Benchmark_Unified_Framework.git
```

### Complete File Tree

```
LLM_Benchmark_Unified_Framework/
│
├── .env.example                            ← API config template
├── .gitignore                              ← Ignores .env, results/, evidence/, chroma_db/
├── LICENSE                                 ← MIT License
├── run.py                                  ← Single CLI entry point
│
├── unified_pipeline/                       ← Core Python package
│   ├── __init__.py
│   ├── base.py                             ← Abstract base classes (shared contracts)
│   ├── config.py                           ← Registry dictionaries + model aliases
│   ├── model_router.py                     ← API client, retry, SQL-detour helpers
│   ├── reporter.py                         ← Crash-safe CSV writer
│   │
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── csv_evidence.py                 ← Krittika's module
│   │   ├── prose_evidence.py               ← Satwik's module
│   │   ├── rag_evidence.py                 ← Nishit's module (stub, needs porting)
│   │   └── visual_pdf_evidence.py          ← Arpitha's module
│   │
│   └── evaluators/
│       ├── __init__.py
│       ├── row_f1_evaluator.py             ← Krittika's evaluator
│       ├── exact_match_evaluator.py        ← Satwik's evaluator
│       ├── span_f1_evaluator.py            ← Nishit's evaluator
│       └── partial_credit_evaluator.py     ← Arpitha's evaluator
│
├── datasets/                               ← One YAML config per dataset
│   ├── imdb_878.yaml
│   ├── music.yaml
│   ├── f1_racing.yaml
│   ├── uda_nqtext.yaml
│   └── imdb_20.yaml
│
├── prompts/                                ← Prompt templates (copy from Sprint 4 archive)
│   ├── llm_only_zero_shot.txt
│   ├── llm_only_one_shot.txt
│   ├── llm_only_few_shot.txt
│   └── sql_detour_zero_shot.txt
│
├── evidence/                               ← Raw data files (gitignored, added locally)
│   ├── imdb_878/
│   │   └── imdb_structured_subset.csv
│   ├── music/
│   │   └── *.txt
│   ├── f1_racing/
│   │   └── *.txt
│   ├── uda/
│   │   └── *.pdf
│   └── imdb_20/
│       └── report.pdf
│
├── ground_truth/                           ← Expected outputs (gitignored, added locally)
│   ├── imdb_878/
│   │   └── T01_sql_result.csv ... T30_sql_result.csv
│   ├── music/
│   └── uda/
│
└── results/                                ← All run outputs (gitignored, generated at runtime)
    └── <model_name>/
        └── <dataset>_<mode>_<prompt>_metrics.csv
```

---

## 4. Architecture — How It All Fits Together

### The Key Design Principle

A new dataset = one new YAML file, zero code changes.  
A new model = one new entry in `MODEL_ALIASES` in `config.py`, zero code changes elsewhere.  
A new team member = implement 2 methods (`build()` and `evaluate()`), nothing else.

### The 5-Stage Pipeline

Every run goes through exactly these 5 stages in order:

```
Stage 1 — Evidence Builder
    Reads raw files (CSV / .txt docs / PDFs)
    Returns a prompt-ready string

Stage 2 — Prompt Engine (inside run.py)
    Fills a template with {evidence} + {question} + {answer_format}
    Loads the right .txt template from prompts/

Stage 3 — Model Router
    Sends the filled prompt to any OpenAI-compatible API
    Handles retry, timeout, degeneration detection
    Separates <think> reasoning from final answer

Stage 4 — Execution Fork
    llm_only:    use model's text response directly
    sql_detour:  extract SQL → validate → execute on SQLite → use result as answer

Stage 5 — Evaluator
    Compares model answer against ground truth CSV
    Returns exact_match, f1, precision, recall
```

### How the YAML Config Drives Everything

`run.py` reads the dataset YAML and uses two keys to look up which classes to instantiate:

```
datasets/imdb_878.yaml
    evidence_type: csv      →  config.EVIDENCE_BUILDERS["csv"]  →  CsvEvidenceBuilder()
    eval_strategy: row_f1   →  config.EVALUATORS["row_f1"]      →  RowF1Evaluator()
```

`run.py` never imports any concrete class directly. It only reads `config.py` dictionaries. This means adding a new dataset never requires touching `run.py`.

### Data Flow Diagram

```
python run.py --dataset imdb_878 --model llama_70b --prompt zero_shot --mode llm_only
                        │
                        ▼
            Load datasets/imdb_878.yaml
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    CsvEvidenceBuilder()      RowF1Evaluator()
    (from EVIDENCE_BUILDERS)  (from EVALUATORS)
            │
            ▼
    evidence.text = full CSV as plain text string
            │
            ▼
    Fill prompts/llm_only_zero_shot.txt
    with {evidence}, {question}, {answer_format}
            │
            ▼
    ModelRouter("llama_70b")
    → expands alias → "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    → reads MODEL_API_KEY, MODEL_BASE_URL from .env
    → calls OpenAI-compatible chat completions API
    → strips <think> tags, detects degeneration
            │
            ▼
    ModelResponse
    .final_text        ← the answer
    .reasoning_text    ← <think> block if any
    .latency_seconds
    .input_tokens / .output_tokens
            │
       ┌────┴────┐
  llm_only    sql_detour
       │            │
       │       extract_sql()
       │       validate_sql()
       │       execute_sql() on in-memory SQLite
       │            │
       └────┬───────┘
            ▼
    RowF1Evaluator.evaluate(response_text, ground_truth_path, expected_columns)
            │
            ▼
    EvalResult
    .content_exact_match   ← 1 or 0
    .row_f1                ← float
    .precision / .recall
    .evaluation_status
            │
            ▼
    Reporter.write_row()   ← writes immediately, crash-safe
    results/llama_70b/imdb_878_llm_only_zero_shot_metrics.csv
```

---

## 5. File-by-File Reference

### `unified_pipeline/base.py`
The shared contract. Written once, never modified by individual team members.

Contains:
- `EvidenceResult` dataclass — what every `build()` must return
- `ModelResponse` dataclass — what `ModelRouter.call()` returns
- `EvalResult` dataclass — what every `evaluate()` must return
- `BaseEvidenceBuilder` abstract class — defines `build()` signature
- `BaseEvaluator` abstract class — defines `evaluate()` signature

### `unified_pipeline/config.py`
The registry. Nishit updates this as each person finishes their module.

Contains:
- `EVIDENCE_BUILDERS` dict — maps YAML `evidence_type` string to a class
- `EVALUATORS` dict — maps YAML `eval_strategy` string to a class
- `MODEL_ALIASES` dict — maps short names to full provider model IDs

When all 4 team members are done, Nishit uncomments the 8 import lines and adds entries to the two dicts. `run.py` never changes.

### `unified_pipeline/model_router.py`
Ported from `run_experiment.py` (Sprint 4 archive). Provider-neutral API client.

Key features:
- Reads `MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL_NAME` from `.env`
- Model alias passthrough: if `--model` is not in `MODEL_ALIASES`, uses it as-is
- Retry loop with exponential back-off (non-retryable errors like 401/403 skip retry)
- `<think>` tag separation for reasoning models (DeepSeek-R1, QwQ, etc.)
- `is_degenerate()` static method — detects repetitive output before evaluation
- `extract_sql()`, `validate_sql()`, `execute_sql()` — full SQL-detour support
- SQLite math functions registered: SQRT, CEIL, FLOOR, POWER

### `unified_pipeline/reporter.py`
Crash-safe writer. Appends one CSV row immediately after every question completes. If the run crashes at question 17, questions 1–16 are already on disk.

### `unified_pipeline/evidence/csv_evidence.py` — Krittika
Serializes the full shared CSV as plain text for the prompt. Supports `max_rows` config key to truncate if needed. Ported from `dataframe_to_prompt_csv()` in Sprint 4's `run_experiment.py`.

### `unified_pipeline/evidence/prose_evidence.py` — Satwik
Loads all `.txt` files from the evidence folder, sorts them, concatenates with `---` separator. Ported from Satwik's Sprint 3–4 pipeline.

### `unified_pipeline/evidence/rag_evidence.py` — Nishit
ChromaDB + MiniLM retrieval. Indexes PDF chunks on first call, caches the collection, retrieves top-k chunks for the question. Nishit needs to port the PDF chunking logic from his Sprint 3 code into `_get_or_index()`. Currently raises `NotImplementedError` with a clear TODO comment.

### `unified_pipeline/evidence/visual_pdf_evidence.py` — Arpitha
Renders each PDF page to a 150 DPI JPEG, base64-encodes it, and returns the image list in `EvidenceResult.metadata["images"]`. The `text` field is intentionally empty — images carry the evidence for multimodal models. Ported from Arpitha's Sprint 4 pipeline.

### `unified_pipeline/evaluators/row_f1_evaluator.py` — Krittika
Treats each CSV row as a unit. Uses multiset intersection (`Counter & Counter`) so duplicate rows are counted correctly. Returns precision, recall, F1, and exact match. Ported from `compute_counter_metrics()` in Sprint 4's `evaluate_results.py`.

### `unified_pipeline/evaluators/exact_match_evaluator.py` — Satwik
Normalises both predicted and expected text (lowercase, strip currency/percent, remove thousands separators, collapse whitespace) then compares. Returns 1 if equal, 0 otherwise.

### `unified_pipeline/evaluators/span_f1_evaluator.py` — Nishit
Tokenises both strings by whitespace, computes set intersection. Returns precision, recall, F1, and exact match. Ported from Nishit's `my_eval.py` in Sprint 3.

### `unified_pipeline/evaluators/partial_credit_evaluator.py` — Arpitha
Reads a `score_tier` column from the ground truth CSV (`correct` / `partial` / `incorrect`) and maps it to 1.0 / 0.5 / 0.0. Ported from Arpitha's Sprint 4 manual scoring approach.

### `run.py`
The only file a user ever runs. Parses CLI args, loads YAML, instantiates builder + evaluator + router + reporter, loops over questions, and calls the 4 stages in order. Handles both `llm_only` and `sql_detour` modes.

### `datasets/*.yaml`
One file per dataset. The entire configuration for a dataset lives here. Keys:

| Key | Type | Purpose |
|---|---|---|
| `name` | string | Human-readable name |
| `evidence_type` | string | Key into `EVIDENCE_BUILDERS` |
| `eval_strategy` | string | Key into `EVALUATORS` |
| `evidence_path` | path | Where to find the raw data |
| `questions_file` | path | JSON file with question list |
| `ground_truth_dir` | path | Folder containing expected output CSVs |
| `max_rows` | int | (csv only) Truncate CSV rows in prompt |
| `rag_top_k` | int | (pdf_rag only) Chunks retrieved per question |

### `.env.example`
Template showing all supported environment variables. Users copy this to `.env` and fill in their API key. `.env` is gitignored and never committed.

---

## 6. Dataset Configs at a Glance

| Dataset | YAML file | evidence_type | eval_strategy | Owner |
|---|---|---|---|---|
| IMDb 878 movies, 30 tasks | `imdb_878.yaml` | `csv` | `row_f1` | Krittika |
| Music, 30 questions | `music.yaml` | `prose_docs` | `exact_match` | Satwik |
| F1 Racing, 55 questions | `f1_racing.yaml` | `prose_docs` | `exact_match` | Satwik |
| UDA NQ-text | `uda_nqtext.yaml` | `pdf_rag` | `span_f1` | Nishit |
| IMDb 20 movies, 10 questions | `imdb_20.yaml` | `pdf_visual` | `partial_credit` | Arpitha |

---

## 7. Environment Setup

### Prerequisites
```bash
pip install openai pandas pyyaml python-dotenv sentence-transformers chromadb PyMuPDF openpyxl
```

### API Configuration
```bash
cp .env.example .env
# Edit .env and fill in your values
```

`.env` file:
```
MODEL_PROVIDER=together
MODEL_API_KEY=your_api_key_here
MODEL_BASE_URL=https://api.together.ai/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct-Turbo
MODEL_TEMPERATURE=0.0
MODEL_MAX_OUTPUT_TOKENS=16384
MODEL_TOKEN_PARAMETER=max_tokens
MODEL_REQUEST_TIMEOUT=600
MODEL_MAX_RETRIES=3
MODEL_RETRY_SLEEP_SECONDS=10
```

Supported providers (any OpenAI-compatible API works):

| Provider | MODEL_BASE_URL |
|---|---|
| Together.ai | `https://api.together.ai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` |
| OpenAI | (leave MODEL_BASE_URL empty) |

---

## 8. Run Commands

```bash
# Basic run
python run.py --dataset imdb_878 --model llama_70b --prompt zero_shot

# SQL detour mode
python run.py --dataset imdb_878 --model llama_70b --prompt zero_shot --mode sql_detour

# Different prompt styles
python run.py --dataset imdb_878 --model llama_70b --prompt one_shot
python run.py --dataset imdb_878 --model llama_70b --prompt few_shot

# Different datasets
python run.py --dataset music       --model gemini_flash --prompt zero_shot
python run.py --dataset f1_racing   --model gemini_flash --prompt one_shot
python run.py --dataset uda_nqtext  --model deepseek_r1  --prompt zero_shot
python run.py --dataset imdb_20     --model llama_70b    --prompt zero_shot

# Raw provider model ID instead of alias (passthrough)
python run.py --dataset imdb_878 --model "meta-llama/Llama-3.3-70B-Instruct-Turbo" --prompt zero_shot

# Run only specific task IDs
python run.py --dataset imdb_878 --model llama_70b --prompt zero_shot --task_ids T01 T05 T12
```

### Model Aliases
| Alias | Resolves to |
|---|---|
| `llama_70b` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `gemini_flash` | `google/gemini-flash-1.5` |
| `deepseek_r1` | `deepseek-ai/DeepSeek-R1` |
| `qwen_72b` | `Qwen/Qwen2.5-72B-Instruct-Turbo` |

---

## 9. Git Workflow

### Branch Structure
```
main          ← protected, always working, merge via PR only
└── dev       ← integration branch, all feature PRs merge here
    ├── feature/krittika-csv-evidence
    ├── feature/satwik-prose-evidence
    ├── feature/nishit-rag-evidence
    ├── feature/arpitha-visual-pdf-evidence
    └── feature/sushma
```

### Rules on `main`
- No direct pushes allowed
- Pull request required with at least 1 approval
- No force pushes
- No deletions

### How Each Teammate Works

```bash
# 1. Clone the repo
git clone https://github.com/Nishitsuthar/LLM_Benchmark_Unified_Framework.git
cd LLM_Benchmark_Unified_Framework

# 2. Switch to your branch (already exists, no need to create)
git checkout feature/krittika-csv-evidence   # replace with your branch name

# 3. Work only in your 2 files
#    evidence/csv_evidence.py
#    evaluators/row_f1_evaluator.py

# 4. Commit and push
git add unified_pipeline/evidence/csv_evidence.py
git add unified_pipeline/evaluators/row_f1_evaluator.py
git commit -m "Implement CsvEvidenceBuilder and RowF1Evaluator"
git push

# 5. Open a Pull Request on GitHub
#    From: feature/krittika-csv-evidence
#    Into: dev   ← NOT main
```

### What Nishit Does After All PRs Are Merged to dev
```bash
# 1. Uncomment all imports in config.py
# 2. Add all 4 classes to EVIDENCE_BUILDERS and EVALUATORS dicts
# 3. Integration test all 5 datasets
# 4. Merge dev into main via PR
```

---

## 10. What Each Person Needs to Implement

### Krittika — `feature/krittika-csv-evidence`
Files to implement:
- `unified_pipeline/evidence/csv_evidence.py` — `CsvEvidenceBuilder.build()`
- `unified_pipeline/evaluators/row_f1_evaluator.py` — `RowF1Evaluator.evaluate()`

Source to port from:
- `imdb-controlled-experiments/src/run_experiment.py` → `dataframe_to_prompt_csv()`
- `imdb-controlled-experiments/src/evaluate_results.py` → `compute_counter_metrics()`

Test with: `python run.py --dataset imdb_878 --model <your_model> --prompt zero_shot`

---

### Satwik — `feature/satwik-prose-evidence`
Files to implement:
- `unified_pipeline/evidence/prose_evidence.py` — `ProseEvidenceBuilder.build()`
- `unified_pipeline/evaluators/exact_match_evaluator.py` — `ExactMatchEvaluator.evaluate()`

Source to port from: Satwik's Sprint 3–4 pipeline (his own repo)

Test with: `python run.py --dataset music --model <your_model> --prompt zero_shot`

---

### Nishit — `feature/nishit-rag-evidence`
Files to implement:
- `unified_pipeline/evidence/rag_evidence.py` — `RagEvidenceBuilder._get_or_index()` (port PDF chunking from Sprint 3)
- `unified_pipeline/evaluators/span_f1_evaluator.py` — `SpanF1Evaluator.evaluate()`
- `unified_pipeline/config.py` — wire all 4 modules once everyone is done

Source to port from: Nishit's Sprint 3 RAG pipeline (his own repo)

Test with: `python run.py --dataset uda_nqtext --model <your_model> --prompt zero_shot`

---

### Arpitha — `feature/arpitha-visual-pdf-evidence`
Files to implement:
- `unified_pipeline/evidence/visual_pdf_evidence.py` — `VisualPdfEvidenceBuilder.build()`
- `unified_pipeline/evaluators/partial_credit_evaluator.py` — `PartialCreditEvaluator.evaluate()`

Source to port from: Arpitha's Sprint 4 pipeline (her own repo)

Test with: `python run.py --dataset imdb_20 --model <your_model> --prompt zero_shot`

---

### Sushma — `feature/sushma`
Role TBD. Branch is ready — coordinate with Nishit for task assignment.

---

## 11. The Abstract Interface (What Every Person Must Follow)

Every evidence builder must implement exactly this:

```python
from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult

class YourEvidenceBuilder(BaseEvidenceBuilder):
    def build(self, question: str, config: dict) -> EvidenceResult:
        # config contains all keys from the dataset YAML
        # config["evidence_path"] is always the path to your data
        # return EvidenceResult(text=..., metadata={...})
        ...
```

Every evaluator must implement exactly this:

```python
from unified_pipeline.base import BaseEvaluator, EvalResult

class YourEvaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,       # the model's final answer text
        ground_truth_path: str,   # full path to the expected output file
        expected_columns: list[str],  # column names from the task definition
    ) -> EvalResult:
        # return EvalResult(content_exact_match, row_f1, precision, recall, evaluation_status)
        ...
```

These two methods are the only integration points. No other file needs to be touched.

---

## 12. Important Design Decisions and Constraints

| Decision | Reason |
|---|---|
| Temperature fixed at 0.0 | Reproducibility — all models must produce deterministic output |
| `.env` never committed | API keys must never appear in git history |
| `evidence/`, `results/`, `ground_truth/` gitignored | Data files are too large and are user-specific |
| `run.py` never imports concrete classes | Decoupling — adding a dataset/module never requires editing the runner |
| Crash-safe reporter | Partial results are preserved if a run is interrupted |
| SQL-detour restricted to zero_shot only | Controlled experiment design from Sprint 4 |
| Provider-neutral via OpenAI-compatible API | Any provider can be used by changing only `.env` |
| Feature branches → dev → main | Prevents broken code from landing on main |
| Local git config (name/email) set per-repo | Commits appear under correct identity on GitHub |

---

## 13. File Status Summary

| File | Status | Owner |
|---|---|---|
| `base.py` | Done | Nishit |
| `config.py` | Done (stubs commented out, to be wired at end) | Nishit |
| `model_router.py` | Done | Nishit |
| `reporter.py` | Done | Nishit |
| `run.py` | Done | Nishit |
| `csv_evidence.py` | Done (example provided) | Krittika to verify/port |
| `row_f1_evaluator.py` | Done (example provided) | Krittika to verify/port |
| `prose_evidence.py` | Done (example provided) | Satwik to verify/port |
| `exact_match_evaluator.py` | Done (example provided) | Satwik to verify/port |
| `rag_evidence.py` | Stub — `_get_or_index()` needs Sprint 3 port | Nishit |
| `span_f1_evaluator.py` | Done (example provided) | Nishit to verify |
| `visual_pdf_evidence.py` | Done (example provided) | Arpitha to verify/port |
| `partial_credit_evaluator.py` | Done (example provided) | Arpitha to verify/port |
| `datasets/*.yaml` | All 5 done | Nishit |
| `prompts/*.txt` | Not yet copied | Nishit (copy from Sprint 4 archive) |
| `evidence/` | Not committed — each person adds locally | Each person |
| `ground_truth/` | Not committed — each person adds locally | Each person |
