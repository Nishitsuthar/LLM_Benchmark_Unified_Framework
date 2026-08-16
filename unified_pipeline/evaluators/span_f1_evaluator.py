"""Token-level span F1 evaluator — Nishit.

Splits both the predicted answer and the ground truth into tokens and
computes precision, recall, and F1 over the token overlap.
Ported from Nishit's my_eval.py in Sprint 3.
"""

from unified_pipeline.base import BaseEvaluator, EvalResult


class SpanF1Evaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        expected  = open(ground_truth_path, encoding="utf-8").read().strip().lower().split()
        predicted = response_text.strip().lower().split()

        if not predicted and not expected:
            return EvalResult(1, 1.0, 1.0, 1.0, "evaluated")
        if not predicted or not expected:
            return EvalResult(0, 0.0, 0.0, 0.0, "evaluated")

        common    = set(predicted) & set(expected)
        precision = len(common) / len(predicted)
        recall    = len(common) / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = int(predicted == expected)

        return EvalResult(exact, round(f1, 6), round(precision, 6), round(recall, 6), "evaluated")
