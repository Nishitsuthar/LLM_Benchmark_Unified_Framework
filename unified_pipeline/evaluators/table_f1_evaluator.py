"""Table F1 evaluator — Krittika.

Row-level F1 for multi-row CSV ground truth tables. Parses both
the ground truth CSV and the model response as CSV tables, then
computes precision/recall/F1 via multiset row intersection.

Identical logic to RowF1Evaluator; separate class so config.py
can register it under the 'table_f1' key without aliasing.
"""

import csv
import io
from collections import Counter

from unified_pipeline.base import BaseEvaluator, EvalResult


class TableF1Evaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        predicted = self._to_counter(response_text)
        expected  = self._to_counter(open(ground_truth_path, encoding="utf-8").read())

        if not predicted and not expected:
            return EvalResult(1, 1.0, 1.0, 1.0, "evaluated")
        if not predicted or not expected:
            return EvalResult(0, 0.0, 0.0, 0.0, "evaluated")

        overlap   = sum((predicted & expected).values())
        precision = overlap / sum(predicted.values())
        recall    = overlap / sum(expected.values())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = int(predicted == expected)

        return EvalResult(exact, round(f1, 6), round(precision, 6), round(recall, 6), "evaluated")

    def _to_counter(self, csv_text: str) -> Counter:
        # Extract the CSV block from the model response — take the last
        # contiguous CSV-looking section to skip any preamble prose.
        csv_text = self._extract_csv_block(csv_text)
        rows = list(csv.reader(io.StringIO(csv_text.strip())))
        if len(rows) < 2:
            return Counter()
        return Counter(tuple(cell.strip() for cell in row) for row in rows[1:])  # skip header

    def _extract_csv_block(self, text: str) -> str:
        """Return the last run of lines that look like CSV (contain commas)."""
        lines = text.strip().splitlines()
        # Walk from end to find the last CSV-like block
        block = []
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                if block:
                    break  # blank line ends the block
                continue
            block.append(line)
        if not block:
            return text
        return "\n".join(reversed(block))
