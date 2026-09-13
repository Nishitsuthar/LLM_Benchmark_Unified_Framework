"""Rescore imdb_20 results using partial credit logic.

Reads existing results CSVs (model_response + ground_truth already present),
applies partial credit scoring, and overwrites the files with corrected
exact_match, row_f1, eval_status columns.

No API calls needed — works entirely from the saved responses.

Usage:
    python scripts/rescore_imdb20.py
"""

import csv
import io
import re
from pathlib import Path


RESULTS_FILES = [
    "results/gemini_flash/imdb_20_llm_only_zero_shot_metrics.csv",
    "results/mistral_small/imdb_20_llm_only_zero_shot_metrics.csv",
]

NUMERIC_TOL = 0.05   # tolerance for numeric comparisons


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def extract_gt(gt_raw: str) -> str:
    """Strip the 'answer_text\\n' header from ground truth."""
    lines = gt_raw.strip().splitlines()
    if lines and lines[0].strip().lower() == "answer_text":
        return "\n".join(lines[1:]).strip()
    return gt_raw.strip()


def normalize(text: str) -> str:
    return text.strip().lower()


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def numbers_close(a: float, b: float, tol: float = NUMERIC_TOL) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------------
# Per-question scoring logic
# ------------------------------------------------------------------

def score(question_id: str, gt_raw: str, response: str) -> tuple[int, float, str]:
    """Return (exact_match, row_f1, eval_status)."""
    gt = extract_gt(gt_raw)
    resp = response.strip()

    if not resp:
        return 0, 0.0, "empty_response"

    gt_norm = normalize(gt)
    resp_norm = normalize(resp)

    # F1, F10 — single numeric answer
    if question_id in ("F1", "F10"):
        gt_nums = extract_numbers(gt)
        resp_nums = extract_numbers(resp)
        if not gt_nums or not resp_nums:
            return 0, 0.0, "evaluated"
        match = numbers_close(resp_nums[0], gt_nums[0])
        score_val = 1.0 if match else 0.0
        return int(match), score_val, "evaluated"

    # F2 — genre name + number (e.g. "Sci-Fi - 8.08")
    # Score: 0.5 for correct genre name, 0.5 for correct number
    if question_id == "F2":
        gt_parts = re.split(r"\s*[-–]\s*", gt, maxsplit=1)
        resp_parts = re.split(r"\s*[-–]\s*", resp, maxsplit=1)
        gt_genre = gt_parts[0].strip().lower() if gt_parts else ""
        gt_num = extract_numbers(gt_parts[1]) if len(gt_parts) > 1 else []
        resp_genre = resp_parts[0].strip().lower() if resp_parts else resp_norm
        resp_num = extract_numbers(resp_parts[1]) if len(resp_parts) > 1 else extract_numbers(resp)

        name_correct = gt_genre in resp_genre or resp_genre in gt_genre
        num_correct = bool(gt_num and resp_num and numbers_close(resp_num[0], gt_num[0]))

        f1 = round((0.5 * name_correct) + (0.5 * num_correct), 4)
        exact = int(name_correct and num_correct)
        return exact, f1, "evaluated"

    # F3 — semicolon-separated list of movie titles
    if question_id == "F3":
        gt_items = {normalize(x) for x in re.split(r"[;,]", gt) if x.strip()}
        # model may use commas or semicolons
        resp_items = {normalize(x) for x in re.split(r"[;,]", resp) if x.strip()}
        if not gt_items:
            return 0, 0.0, "evaluated"
        # partial: for each gt item check if it appears anywhere in resp
        matched = sum(1 for item in gt_items if any(item in r or r in item for r in resp_items))
        f1 = round(matched / len(gt_items), 4)
        exact = int(matched == len(gt_items))
        return exact, f1, "evaluated"

    # F4 — total runtime in minutes (model may say "X minutes" or "X hours Y minutes")
    if question_id == "F4":
        gt_nums = extract_numbers(gt)
        # convert "X hours Y minutes" → total minutes
        hours_match = re.search(r"(\d+)\s*hour", resp, re.IGNORECASE)
        mins_match = re.search(r"(\d+)\s*min", resp, re.IGNORECASE)
        if hours_match:
            total = int(hours_match.group(1)) * 60 + (int(mins_match.group(1)) if mins_match else 0)
            resp_nums = [float(total)]
        else:
            resp_nums = extract_numbers(resp)
        if not gt_nums or not resp_nums:
            return 0, 0.0, "evaluated"
        match = numbers_close(resp_nums[0], gt_nums[0], tol=1.0)
        return int(match), float(match), "evaluated"

    # F5 — genre name + rating range (e.g. "Action - 2.2 rating points")
    if question_id == "F5":
        gt_parts = re.split(r"\s*[-–]\s*", gt, maxsplit=1)
        gt_genre = gt_parts[0].strip().lower() if gt_parts else ""
        gt_num = extract_numbers(gt_parts[1]) if len(gt_parts) > 1 else []
        resp_genre_match = re.match(r"([a-zA-Z\-/]+)", resp.strip())
        resp_genre = resp_genre_match.group(1).strip().lower() if resp_genre_match else resp_norm
        resp_num = extract_numbers(resp)

        name_correct = gt_genre in resp_genre or resp_genre in gt_genre
        num_correct = bool(gt_num and resp_num and numbers_close(resp_num[0], gt_num[0], tol=0.1))

        f1 = round((0.5 * name_correct) + (0.5 * num_correct), 4)
        exact = int(name_correct and num_correct)
        return exact, f1, "evaluated"

    # F6 — single numeric (average runtime in minutes)
    if question_id == "F6":
        gt_nums = extract_numbers(gt)
        resp_nums = extract_numbers(resp)
        if not gt_nums or not resp_nums:
            return 0, 0.0, "evaluated"
        match = numbers_close(resp_nums[0], gt_nums[0], tol=1.0)
        return int(match), float(match), "evaluated"

    # F7 — movie name + optional rating (e.g. "Jurassic World - 6.9")
    if question_id == "F7":
        gt_parts = re.split(r"\s*[-–]\s*", gt, maxsplit=1)
        gt_movie = gt_parts[0].strip().lower()
        resp_norm_val = normalize(resp)
        name_correct = gt_movie in resp_norm_val or resp_norm_val.startswith(gt_movie)
        # full credit if name correct (number is bonus detail)
        return int(name_correct), float(name_correct), "evaluated"

    # F8 — three numbers: before avg, after avg, difference
    if question_id == "F8":
        gt_nums = extract_numbers(gt)   # [8.44, 8.26, 0.18]
        resp_nums = extract_numbers(resp)
        if not gt_nums:
            return 0, 0.0, "evaluated"
        matched = 0
        for gn in gt_nums:
            if any(numbers_close(rn, gn, tol=0.1) for rn in resp_nums):
                matched += 1
        f1 = round(matched / len(gt_nums), 4)
        exact = int(matched == len(gt_nums))
        return exact, f1, "evaluated"

    # F9 — genre name + total runtime (e.g. "Drama - 1608 minutes")
    if question_id == "F9":
        gt_parts = re.split(r"\s*[-–]\s*", gt, maxsplit=1)
        gt_genre = gt_parts[0].strip().lower()
        gt_num = extract_numbers(gt_parts[1]) if len(gt_parts) > 1 else []
        resp_genre_match = re.match(r"([a-zA-Z\-/]+)", resp.strip())
        resp_genre = resp_genre_match.group(1).strip().lower() if resp_genre_match else resp_norm
        resp_num = extract_numbers(resp)

        name_correct = gt_genre in resp_genre or resp_genre in gt_genre
        num_correct = bool(gt_num and resp_num and numbers_close(resp_num[0], gt_num[0], tol=2.0))

        f1 = round((0.5 * name_correct) + (0.5 * num_correct), 4)
        exact = int(name_correct and num_correct)
        return exact, f1, "evaluated"

    # fallback
    exact = int(normalize(gt) == normalize(resp))
    return exact, float(exact), "evaluated"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def rescore_file(path: str) -> None:
    p = Path(path)
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    fieldnames = rows[0].keys() if rows else []

    print(f"\n{p}")
    for row in rows:
        qid = row["question_id"]
        gt_raw = row["ground_truth"]
        response = row["model_response"]

        exact, f1, status = score(qid, gt_raw, response)

        old_f1 = row.get("row_f1", "")
        row["exact_match"] = exact
        row["row_f1"] = f1
        row["precision"] = f1
        row["recall"] = f1
        row["eval_status"] = status

        print(f"  {qid}: f1={f1:.2f} exact={exact}  (was: {old_f1 or 'empty'})")

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_f1 = sum(float(r["row_f1"]) for r in rows) / len(rows) if rows else 0
    exact_n = sum(int(r["exact_match"]) for r in rows)
    print(f"  → Avg F1: {avg_f1:.3f} | Exact: {exact_n}/{len(rows)}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    for f in RESULTS_FILES:
        rescore_file(f)
