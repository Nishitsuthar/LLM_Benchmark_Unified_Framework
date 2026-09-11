"""HotpotQA F1 evaluator — Sushma.

Implements the standard HotpotQA / SQuAD evaluation protocol:
normalise both strings (lowercase, strip punctuation and articles),
then compute token-overlap F1, precision, recall, and exact match.

Ported from Sushma's src/metrics.py in the hotpot-structured-benchmark.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from unified_pipeline.base import BaseEvaluator, EvalResult


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


class HotpotEvaluator(BaseEvaluator):
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        gold      = _normalize(open(ground_truth_path, encoding="utf-8").read())
        predicted = _normalize(response_text)

        gold_tokens = gold.split()
        pred_tokens = predicted.split()

        exact = int(predicted == gold)

        if not pred_tokens and not gold_tokens:
            return EvalResult(1, 1.0, 1.0, 1.0, "evaluated")
        if not pred_tokens or not gold_tokens:
            return EvalResult(0, 0.0, 0.0, 0.0, "evaluated")

        common    = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        precision = common / len(pred_tokens)
        recall    = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return EvalResult(exact, round(f1, 6), round(precision, 6), round(recall, 6), "evaluated")
