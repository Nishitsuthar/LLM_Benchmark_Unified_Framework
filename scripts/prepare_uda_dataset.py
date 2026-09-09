"""One-time script to prepare the UDA NQ-text dataset for the unified pipeline.

Uses the exact 57 hard-case questions from Sprint 4 that failed all models.
Reads from:
  - Sprint 4 all_questions_combined.csv  (question text + ground truth)
  - Sprint 3 wiki_nq_docs/pdfs/          (4 PDF evidence files)

Writes:
  - evidence/uda/questions.json           (pipeline questions file)
  - evidence/uda/*.pdf                    (copied from Sprint 3)
  - ground_truth/uda/<question_id>.txt    (short answer per question)

Ground truth in the CSV is "short_answer | long_answer" — we store
only the short answer (the part before the first pipe) as the expected
answer, matching what SpanF1Evaluator scores against.

Usage:
    python3 scripts/prepare_uda_dataset.py
"""

import csv
import json
import shutil
from pathlib import Path

SPRINT3_BASE = Path(__file__).parent.parent.parent / "LLM_Benchmark_Team_Project_2026" / "Sprint 3" / "UDA-Benchmark"
SPRINT4_BASE = Path(__file__).parent.parent.parent / "LLM_Benchmark_Team_Project_2026" / "Sprint 4"

PDF_SRC          = SPRINT3_BASE / "dataset" / "src_doc_files_example" / "wiki_nq_docs" / "pdfs"
ALL_QUESTIONS_CSV = SPRINT4_BASE / "benchmark" / "questions" / "all_questions_combined.csv"

EVIDENCE_DIR     = Path(__file__).parent.parent / "evidence" / "uda"
GROUND_TRUTH_DIR = Path(__file__).parent.parent / "ground_truth" / "uda"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

# Copy PDFs into evidence/uda/
available_pdfs = {p.stem: p for p in PDF_SRC.glob("*.pdf")}
print(f"Found {len(available_pdfs)} PDFs: {list(available_pdfs.keys())}")
for stem, src_path in available_pdfs.items():
    dst = EVIDENCE_DIR / src_path.name
    shutil.copy2(src_path, dst)
    print(f"  Copied: {src_path.name}")

# Read Sprint 4 hard-case nqtext questions
questions = []
skipped = 0

with open(ALL_QUESTIONS_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["dataset"].strip() != "nqtext":
            continue
        if row["is_sprint4_hard_case"].strip().upper() != "YES":
            continue

        raw_gt = row["ground_truth"].strip()
        # ground truth format: "short_answer | long_answer passage"
        short_answer = raw_gt.split("|")[0].strip()
        if not short_answer:
            skipped += 1
            continue

        original_id = row["question_id"].strip()
        q_num       = len(questions) + 1
        q_id        = f"Q{q_num:04d}"
        gt_filename = f"{q_id}.txt"

        questions.append({
            "id":                q_id,
            "text":              row["question"].strip(),
            "doc_name":          row["doc_name"].strip(),
            "original_id":       original_id,
            "ground_truth_file": gt_filename,
            "answer_format":     "short text answer",
        })

        (GROUND_TRUTH_DIR / gt_filename).write_text(short_answer, encoding="utf-8")

print(f"\nQuestions included : {len(questions)}")
print(f"Skipped (no answer): {skipped}")

questions_path = EVIDENCE_DIR / "questions.json"
with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"\nWrote: {questions_path}")
print(f"Wrote: {len(questions)} ground truth files to {GROUND_TRUTH_DIR}/")
print("\nDone. Run the pipeline with:")
print("  python3 run.py --dataset uda_nqtext --model llama_70b --prompt zero_shot")
