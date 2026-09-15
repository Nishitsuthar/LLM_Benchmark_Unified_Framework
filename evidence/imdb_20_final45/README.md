# IMDb-20 final45, evidence v2

Condition: **image-only normalized metadata**.

This is researcher-generated visual evidence, not unaltered IMDb screenshots or purely unstructured webpage evidence. It represents the same frozen 20-movie dataset with all six benchmark fields visible on every card. All 45 questions use all 20 cards, grouped into five composite images identically for Gemini and Mistral. The CSV, manifest, questions CSV, ground truths and validation reports are offline preparation/evaluation artifacts; they are never supplied as evidence to a candidate model.

The historical `imdb_20` dataset, screenshot PDF, questions, ground truths, prompts and results are preserved. Final45 is a separate experiment. F1-F10 JSON objects are preserved and their ground-truth files are byte-for-byte copies in the new ground-truth directory. F11-F45 retain the originally verified computations and expected answers, including the original genre-based outlier questions. Only the approved wording corrections were applied. Prior workaround replacement questions are not used.

## Evidence

- PDF: `evidence/imdb_20/imdb_20_final_v2_metadata_cards.pdf`
- Twenty 1200 x 900 image-only PDF pages, one independent movie per page.
- Card title: 48 px; values: 40 px; field labels: 27 px; note: 25 px.
- Final model rendering: scale 1.0, JPEG quality 75, four pages per image in a 2x2 grid (2420 x 1820 pixels, 20-pixel gutters, no card resizing).
- `page_layout: metadata_only` supplies pages 0 through 19, with `pages_per_image: 4`.
- The original builder defaults remain alternating metadata/cast selection and two pages per image.
- Each card shows only title, year, runtime in minutes, IMDb rating, Benchmark Director, Normalized Genres, and the consistent normalization note.
- There are no expected answers, question IDs, aggregates, thresholds, derived statistics, or outlier annotations in the card images.

## Frozen-source provenance

`frozen_movies.csv` is a byte-preserved copy of the existing frozen source. All 20 movie records and complete genre memberships were checked against the existing read-only SQLite database. Source, font, PDF and card hashes are in `manifest.json`.

The source's director is the **benchmark director**, not necessarily the complete set of IMDb directing credits. In archived IMDb HTML, genres differ from this frozen source for Memento, The Dark Knight and Inglourious Basterds. The cards visibly disclose the benchmark-normalized genres instead of silently filling hidden screenshot fields. No frozen movie values were changed.

## Dataset and evaluation

- Config: `datasets/imdb_20_final45.yaml`
- CSV: `evidence/imdb_20_final45/focused_45_question_pdf_questions.csv`
- Questions: `evidence/imdb_20_final45/questions.json`
- Ground truths: `ground_truth/imdb_20_final45/F1.csv` through `F45.csv`
- Prompt: `prompts/imdb_20_final45_llm_only_zero_shot.txt`
- Counts: 15 arithmetic_aggregation, 15 genre_level_aggregation, 15 statistical_outlier_detection.
- The inherited evaluator registry key remains `partial_credit`; the IMDb runner records binary exact-match accuracy, preserving the historical automatic-scoring convention. This preparation does not add semantic judging or claim that exact match measures semantic correctness.
- F1-F10 must be rerun along with F11-F45 for a final-condition comparison; old screenshot-condition responses must not be combined with new-condition responses as one uniform run.

## Offline validation

Run from the repository root:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_imdb_final45.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_imdb*.py' -v
```

The validator independently recomputes every F1-F45 answer, checks question/ground-truth consistency, source/card/PDF identity, image count, absence of a PDF text layer, and historical rendering compatibility. `offline_validation.json` contains the numeric checks. `visual_validation.json` records inspection of every actual model-input JPEG. Readability is an inspection finding for the actual five JPEG payloads; downstream provider resizing is outside this offline check. The final candidate runs are preserved separately and are not rerun by validation.

The integration tests use a fake router and temporary result directories. Final45 tests explicitly prohibit network sockets and dotenv reads. No candidate model or OpenRouter API was invoked.

## Reproduction

`scripts/prepare_imdb_final45.py` takes `--source-csv`, `--source-db`, `--font`, and `--bold-font`. It refuses to overwrite the target files/directories. Use a separate clean checkout when reproducing, with the frozen source/database and the fonts whose hashes are recorded in the manifest. The current artwork used macOS Arial and Arial Bold. Generation uses Pillow and PyMuPDF.

Status: final comparable runs used the five-image condition. This support-file validation makes no API calls and does not alter any existing run or judge output.
