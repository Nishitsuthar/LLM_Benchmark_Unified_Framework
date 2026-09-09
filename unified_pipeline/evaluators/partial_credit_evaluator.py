"""Partial-credit evaluator — Arpitha.

Evaluation strategy for the IMDb-20 visual benchmark.

Exact normalized matches are scored automatically as correct (1.0).

Non-exact responses are not automatically classified as partial or
incorrect because the benchmark contains different answer types
(numerical answers, movie lists, genre aggregations, etc.).

Instead, non-exact answers are marked as "needs_manual_review".
A human reviewer can later assign:
    correct   = 1.0
    partial   = 0.5
    incorrect = 0.0
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from unified_pipeline.base import BaseEvaluator, EvalResult


class PartialCreditEvaluator(BaseEvaluator):
    """Hybrid exact-match + manual partial-credit evaluator."""

    SCORES = {
        "correct": 1.0,
        "partial": 0.5,
        "incorrect": 0.0,
    }

    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        """Evaluate a model response against the expected answer.

        Exact normalized matches are automatically scored as correct.

        Any non-exact response is marked for manual review rather than
        automatically guessing whether it deserves partial credit.
        """

        expected = self._read_expected_answer(
            ground_truth_path
        )

        predicted_normalized = self._normalize(
            response_text
        )

        expected_normalized = self._normalize(
            expected
        )

        if predicted_normalized == expected_normalized:
            return EvalResult(
                content_exact_match=1,
                row_f1=1.0,
                precision=1.0,
                recall=1.0,
                evaluation_status="evaluated_exact",
            )

        return EvalResult(
            content_exact_match=0,
            row_f1=None,
            precision=None,
            recall=None,
            evaluation_status="needs_manual_review",
        )

    @staticmethod
    def _read_expected_answer(
        ground_truth_path: str,
    ) -> str:
        """Read the answer_text field from the ground-truth CSV."""

        path = Path(
            ground_truth_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {path}"
            )

        with path.open(
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            row = next(
                reader,
                None,
            )

        if row is None:
            raise ValueError(
                f"Ground truth file is empty: {path}"
            )

        if "answer_text" not in row:
            raise ValueError(
                f"Ground truth file {path} must contain "
                "an 'answer_text' column."
            )

        return (
            row["answer_text"]
            or ""
        ).strip()

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """Apply conservative normalization before exact comparison."""

        text = str(
            text or ""
        ).strip().lower()

        # Treat different dash characters consistently.
        text = (
            text.replace("–", "-")
            .replace("—", "-")
        )

        # Collapse repeated whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Normalize spacing around semicolon-separated lists.
        text = re.sub(
            r"\s*;\s*",
            "; ",
            text,
        )

        # Remove a final full stop only.
        text = text.rstrip(".")

        return text