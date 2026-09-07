"""One-time script to prepare the UDA NQ-text dataset for the unified pipeline.

Reads:
  - Sprint 3 nq_qa.csv  (pipe-delimited)
  - Sprint 3 wiki_nq_docs/pdfs/  (4 PDF files)

Writes:
  - evidence/uda/questions.json          (pipeline questions file)
  - evidence/uda/*.pdf                   (copied from Sprint 3)
  - ground_truth/uda/<question_id>.txt   (one short-answer file per question)

Only questions whose doc_name matches one of the available PDFs are included.
Questions with an empty short_answer are skipped (unanswerable).

Usage:
    python scripts/prepare_uda_dataset.py
"""

import csv
import json
import shutil
from pathlib import Path

SPRINT3_BASE = Path(__file__).parent.parent.parent / "LLM_Benchmark_Team_Project_2026" / "Sprint 3" / "UDA-Benchmark"
PDF_SRC      = SPRINT3_BASE / "dataset" / "src_doc_files_example" / "wiki_nq_docs" / "pdfs"
NQ_QA_CSV    = SPRINT3_BASE / "dataset" / "qa" / "nq_qa.csv"

EVIDENCE_DIR    = Path(__file__).parent.parent / "evidence" / "uda"
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

# Read nq_qa.csv and filter to questions with matching PDFs and non-empty answers
questions = []
skipped_no_pdf   = 0
skipped_no_answer = 0

with open(NQ_QA_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="|")
    for row in reader:
        doc_name = row["doc_name"].strip()
        if doc_name not in available_pdfs:
            skipped_no_pdf += 1
            continue

        short_answer = row["short_answer"].strip()
        if not short_answer:
            skipped_no_answer += 1
            continue

        q_uid = row["q_uid"].strip()
        question_text = row["question"].strip()

        # Use a clean sequential ID: Q0001, Q0002, ...
        q_id = f"Q{len(questions) + 1:04d}"
        gt_filename = f"{q_id}.txt"

        questions.append({
            "id":               q_id,
            "text":             question_text,
            "doc_name":         doc_name,
            "q_uid":            q_uid,
            "ground_truth_file": gt_filename,
            "answer_format":    "short text answer",
        })

        # Write ground truth file
        gt_path = GROUND_TRUTH_DIR / gt_filename
        gt_path.write_text(short_answer, encoding="utf-8")

print(f"\nQuestions included : {len(questions)}")
print(f"Skipped (no PDF)   : {skipped_no_pdf}")
print(f"Skipped (no answer): {skipped_no_answer}")

# Write questions.json
questions_path = EVIDENCE_DIR / "questions.json"
with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"\nWrote: {questions_path}")
print(f"Wrote: {len(questions)} ground truth files to {GROUND_TRUTH_DIR}/")
print("\nDone. Run the pipeline with:")
print("  python run.py --dataset uda_nqtext --model llama_70b --prompt zero_shot")
