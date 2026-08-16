"""Row-level F1 evaluator — Krittika.

Treats each CSV row as a unit. Computes precision, recall, and F1
using multiset intersection so duplicate rows are counted correctly.
Ported from compute_counter_metrics() in evaluate_results.py.
"""

import csv
import io
from collections import Counter

from unified_pipeline.base import BaseEvaluator, EvalResult


class RowF1Evaluator(BaseEvaluator):
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
        rows = list(csv.reader(io.StringIO(csv_text.strip())))
        if len(rows) < 2:
            return Counter()
        return Counter(tuple(cell.strip() for cell in row) for row in rows[1:])  # skip header
