"""Registry: maps YAML keys to concrete classes and model aliases.

Nishit updates this file as each team member finishes their module.
run.py never changes — it only reads these dictionaries.
"""

# from unified_pipeline.evidence.csv_evidence import CsvEvidenceBuilder  # Krittika — not merged yet
from unified_pipeline.evidence.prose_evidence import ProseEvidenceBuilder
# from unified_pipeline.evidence.rag_evidence import RagEvidenceBuilder            # Nishit adds
# from unified_pipeline.evidence.visual_pdf_evidence import VisualPdfEvidenceBuilder  # Arpitha adds

from unified_pipeline.evaluators.row_f1_evaluator import RowF1Evaluator
from unified_pipeline.evaluators.exact_match_evaluator import ExactMatchEvaluator
# from unified_pipeline.evaluators.span_f1_evaluator import SpanF1Evaluator           # Nishit adds
# from unified_pipeline.evaluators.partial_credit_evaluator import PartialCreditEvaluator  # Arpitha adds


EVIDENCE_BUILDERS = {
    # "csv":     CsvEvidenceBuilder,    # Krittika — not merged yet
    "prose_docs": ProseEvidenceBuilder,
    # "pdf_rag":    RagEvidenceBuilder,
    # "pdf_visual": VisualPdfEvidenceBuilder,
}

EVALUATORS = {
    "row_f1":         RowF1Evaluator,
    "exact_match":    ExactMatchEvaluator,
    # "span_f1":        SpanF1Evaluator,
    # "partial_credit": PartialCreditEvaluator,
}

# Friendly short names → provider model IDs.
# If --model is not found here, it is used as-is (passthrough).
MODEL_ALIASES = {
    "llama_70b":       "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "gemini_flash":    "google/gemini-flash-1.5",
    "deepseek_r1":     "deepseek-ai/DeepSeek-R1",
    "qwen_72b":        "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "nemotron_ultra":  "nvidia/nemotron-3-ultra-550b-a55b",
    # Add these once you have the correct model IDs:
    # "gemini_flash_31": "...",   # needs Google AI Studio key, not NVIDIA
    # "mistral_small_4": "...",   # check Mistral's own API or an aggregator
}
