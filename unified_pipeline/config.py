"""Registry: maps YAML keys to concrete classes and model aliases.

Nishit updates this file as each team member finishes their module.
run.py never changes — it only reads these dictionaries.
"""

from unified_pipeline.evidence.csv_evidence import CsvEvidenceBuilder
# from unified_pipeline.evidence.prose_evidence import ProseEvidenceBuilder       # Satwik adds
from unified_pipeline.evidence.rag_evidence import RagEvidenceBuilder            # Nishit adds
# from unified_pipeline.evidence.visual_pdf_evidence import VisualPdfEvidenceBuilder  # Arpitha adds

from unified_pipeline.evaluators.row_f1_evaluator import RowF1Evaluator
# from unified_pipeline.evaluators.exact_match_evaluator import ExactMatchEvaluator   # Satwik adds
from unified_pipeline.evaluators.span_f1_evaluator import SpanF1Evaluator           # Nishit adds
# from unified_pipeline.evaluators.partial_credit_evaluator import PartialCreditEvaluator  # Arpitha adds


EVIDENCE_BUILDERS = {
    "csv":        CsvEvidenceBuilder,
    # "prose_docs": ProseEvidenceBuilder,
    "pdf_rag":    RagEvidenceBuilder,
    # "pdf_visual": VisualPdfEvidenceBuilder,
}

EVALUATORS = {
    "row_f1":         RowF1Evaluator,
    # "exact_match":    ExactMatchEvaluator,
    "span_f1":        SpanF1Evaluator,
    # "partial_credit": PartialCreditEvaluator,
}

# Friendly short names → provider model IDs.
# If --model is not found here, it is used as-is (passthrough).
MODEL_ALIASES = {
    "llama_70b":     "meta-llama/llama-3.3-70b-instruct",
    "gemini_flash":  "google/gemini-3.1-flash-lite",
    "deepseek_r1":   "deepseek/deepseek-r1",
    "qwen_72b":      "qwen/qwen-2.5-72b-instruct",
}
