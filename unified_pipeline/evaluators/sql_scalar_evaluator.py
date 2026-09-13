"""SQL scalar evaluator — for plain-text ground truth with SQL detour.

Used when SQL detour runs against a multi-table SQLite database (e.g.,
F1 Racing, Music) whose ground truth answers are plain text scalars or
names, not CSV tables.

Extracts the first data row from the SQL result CSV, normalises each
value, then compares against the normalised ground truth.

Single-column result  → numeric or string match against GT.
Multi-column result   → all SQL values must appear in the GT string
                        (handles "track name with 490,780,198 streams").
"""

import csv
import io
import re

from unified_pipeline.base import BaseEvaluator, EvalResult


_UNIT_RE = re.compile(
    r"\s+(weeks?|tracks?|streams?|points?|laps?|races?|seasons?)\s*$",
    re.IGNORECASE,
)


class SqlScalarEvaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        gt_raw = open(ground_truth_path, encoding="utf-8").read().strip()
        values = self._extract_row_values(response_text)

        if not values:
            return EvalResult(0, 0.0, 0.0, 0.0, "empty_result")

        gt_norm = self._normalize(gt_raw)

        if len(values) == 1:
            match = self._compare(self._normalize(values[0]), gt_norm)
        else:
            # Multi-column: all SQL values must appear somewhere in GT
            match = all(self._normalize(v) in gt_norm for v in values)

        score = float(match)
        return EvalResult(int(match), score, score, score, "evaluated")

    # ------------------------------------------------------------------

    def _extract_row_values(self, csv_text: str) -> list[str]:
        rows = list(csv.reader(io.StringIO(csv_text.strip())))
        if len(rows) < 2:
            return []
        return [cell.strip() for cell in rows[1] if cell.strip()]

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[$%']", "", text)   # strip currency / percent / single-quotes
        text = text.replace(",", "")         # remove all commas (thousands separators)
        text = _UNIT_RE.sub("", text)        # strip trailing unit words
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compare(self, pred: str, gt: str) -> bool:
        if pred == gt:
            return True
        try:
            # Numeric comparison — tolerate small floating-point drift
            return abs(float(pred) - float(gt)) < 0.02
        except (ValueError, TypeError):
            return False
