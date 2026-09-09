"""One-time script to prepare the 3 Sprint 4 datasets for the unified pipeline.

Reads question data from the Sprint 4 dashboard HTML (the single source of truth)
and copies evidence PDFs from Sprint 3.

Datasets:
  - tathybrid       (29 questions, 4 annual report PDFs)
  - finhybrid       (31 questions, 4 financial report PDFs)
  - music_structured (16 questions, 1 music dataset PDF)

Writes per dataset:
  - evidence/<dataset>/<pdf files>
  - evidence/<dataset>/questions.json
  - ground_truth/<dataset>/<question_id>.txt

Usage:
    python3 scripts/prepare_datasets.py
"""

import json
import re
import shutil
from pathlib import Path

SPRINT3 = Path(__file__).parent.parent.parent / "LLM_Benchmark_Team_Project_2026" / "Sprint 3" / "UDA-Benchmark" / "dataset" / "src_doc_files_example"
SPRINT4 = Path(__file__).parent.parent.parent / "LLM_Benchmark_Team_Project_2026" / "Sprint 4"
DASHBOARD = SPRINT4 / "benchmark" / "comparison" / "benchmark_dashboard.html"
BASE = Path(__file__).parent.parent

# PDF source locations per dataset
PDF_SOURCES = {
    "tathybrid":        SPRINT3 / "tat_docs",
    "finhybrid":        SPRINT3 / "fin_docs",
    "music_structured": SPRINT4,
}

# Music PDF filename
MUSIC_PDF = "music_dataset.pdf"

def load_dashboard_data():
    content = DASHBOARD.read_text(encoding="utf-8")
    match = re.search(r'var DATA = (\[.*?\]);', content, re.DOTALL)
    return json.loads(match.group(1))

def prepare_dataset(ds_name, records):
    ev_dir = BASE / "evidence" / ds_name
    gt_dir = BASE / "ground_truth" / ds_name
    ev_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDFs
    src = PDF_SOURCES[ds_name]
    if ds_name == "music_structured":
        pdf_files = [src / MUSIC_PDF]
    else:
        pdf_files = list(src.glob("*.pdf"))

    for pdf in pdf_files:
        dst = ev_dir / pdf.name
        shutil.copy2(pdf, dst)
        print(f"  Copied: {pdf.name}")

    # Write questions.json and ground truth files
    questions = []
    for i, r in enumerate(records, 1):
        q_id = f"Q{i:04d}"
        gt_filename = f"{q_id}.txt"
        # ground truth: take first value before pipe if present
        gt = r["gt"].split("|")[0].strip()

        questions.append({
            "id":                q_id,
            "text":              r["q"],
            "original_id":       r["qid"],
            "category":          r["cat"],
            "difficulty":        r["diff"],
            "ground_truth_file": gt_filename,
            "answer_format":     "short answer (number, name, or short phrase)",
        })
        (gt_dir / gt_filename).write_text(gt, encoding="utf-8")

    (ev_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Wrote {len(questions)} questions + ground truth files")

def main():
    data = load_dashboard_data()
    for ds in ["tathybrid", "finhybrid", "music_structured"]:
        records = [d for d in data if d["ds"] == ds]
        print(f"\n=== {ds} ({len(records)} questions) ===")
        prepare_dataset(ds, records)
    print("\nDone.")

if __name__ == "__main__":
    main()
