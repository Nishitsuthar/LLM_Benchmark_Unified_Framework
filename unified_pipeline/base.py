"""Shared contracts for the unified pipeline.

Nobody modifies this file once agreed upon. Every evidence builder and
evaluator inherits from the abstract classes defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvidenceResult:
    text: str        # prompt-ready evidence string (empty for visual PDF)
    metadata: dict   # chunk count, row count, image list, etc.


@dataclass
class ModelResponse:
    final_text: str        # the answer (or SQL in sql_detour mode)
    reasoning_text: str    # <think> blocks if any
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float


@dataclass
class EvalResult:
    content_exact_match: int       # 1 or 0
    row_f1: float | None
    precision: float | None
    recall: float | None
    evaluation_status: str         # "evaluated", "not_evaluated_...", etc.


class BaseEvidenceBuilder(ABC):
    """Each team member implements exactly this one method."""

    @abstractmethod
    def build(self, question: str, config: dict) -> EvidenceResult:
        """Turn raw files into a prompt-ready evidence string."""
        ...


class BaseEvaluator(ABC):
    """Each team member implements exactly this one method."""

    @abstractmethod
    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
    ) -> EvalResult:
        """Score a model response against ground truth."""
        ...
