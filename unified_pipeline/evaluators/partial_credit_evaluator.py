"""Partial credit evaluator — Arpitha.

Reads a score tier from the ground truth file and maps it to a numeric
score: correct=1.0, partial=0.5, incorrect=0.0.
Ported from Arpitha's Sprint 4 manual scoring approach.

Ground truth CSV format expected:
    answer_text,score_tier
    "some expected answer",correct
"""

import csv
from unified_pipeline.base import BaseEvaluator, EvalResult


class PartialCreditEvaluator(BaseEvaluator):
    SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        with open(ground_truth_path, encoding="utf-8") as f:
            gt = next(csv.DictReader(f))

        tier  = gt.get("score_tier", "incorrect").strip().lower()
        score = self.SCORES.get(tier, 0.0)
        exact = int(tier == "correct")

        return EvalResult(exact, score, score, score, "evaluated")
