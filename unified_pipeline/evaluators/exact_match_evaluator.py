"""Normalised exact match evaluator — Satwik.

Strips currency symbols, percent signs, and thousands separators before
comparing. Returns 1 if the normalised strings match exactly, 0 otherwise.
Ported from Satwik's canonical answer normaliser in Sprint 3-4.
"""

import re
from unified_pipeline.base import BaseEvaluator, EvalResult


class ExactMatchEvaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        expected  = self._normalize(open(ground_truth_path, encoding="utf-8").read())
        predicted = self._normalize(response_text)
        match = int(predicted == expected)
        return EvalResult(match, float(match), float(match), float(match), "evaluated")

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[$,%]", "", text)           # strip currency / percent
        text = re.sub(r"(\d),(\d)", r"\1\2", text)  # remove thousands separator
        text = re.sub(r"\s+", " ", text)             # collapse whitespace
        return text
