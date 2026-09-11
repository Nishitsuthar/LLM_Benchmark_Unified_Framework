"""Registry: maps YAML keys to concrete classes and model aliases.

Nishit updates this file as each team member finishes their module.
run.py never changes — it only reads these dictionaries.
"""

from unified_pipeline.evidence.csv_full_evidence import CsvFullEvidenceBuilder  # Krittika
from unified_pipeline.evidence.prose_evidence import ProseEvidenceBuilder        # Satwik
from unified_pipeline.evidence.rag_evidence import RagEvidenceBuilder            # Nishit
from unified_pipeline.evidence.hotpot_evidence import HotpotEvidenceBuilder      # Sushma
from unified_pipeline.evidence.visual_pdf_evidence import VisualPdfEvidenceBuilder  # Arpitha

from unified_pipeline.evaluators.table_f1_evaluator import TableF1Evaluator      # Krittika
from unified_pipeline.evaluators.exact_match_evaluator import ExactMatchEvaluator  # Satwik
from unified_pipeline.evaluators.span_f1_evaluator import SpanF1Evaluator        # Nishit
from unified_pipeline.evaluators.hotpot_evaluator import HotpotEvaluator          # Sushma
from unified_pipeline.evaluators.partial_credit_evaluator import PartialCreditEvaluator  # Arpitha
from unified_pipeline.evaluators.sql_evaluator import SqlTableF1Evaluator         # sql_detour


EVIDENCE_BUILDERS = {
    "csv_full":       CsvFullEvidenceBuilder,
    "prose_docs":     ProseEvidenceBuilder,
    "pdf_rag":        RagEvidenceBuilder,
    "hotpot_context": HotpotEvidenceBuilder,
    "pdf_visual":     VisualPdfEvidenceBuilder,
}


EVALUATORS = {
    "table_f1":       TableF1Evaluator,
    "exact_match":    ExactMatchEvaluator,
    "span_f1":        SpanF1Evaluator,
    "hotpot_f1":      HotpotEvaluator,
    "partial_credit": PartialCreditEvaluator,
    "sql_table_f1":   SqlTableF1Evaluator,
}


# Friendly short names -> provider model IDs.
# If --model is not found here, it is used as-is (passthrough).
MODEL_ALIASES = {
    "llama_70b":     "meta-llama/llama-3.3-70b-instruct",
    "gemini_flash":  "google/gemini-3.1-flash-lite",
    "deepseek_r1":   "deepseek/deepseek-r1",
    "qwen_72b":      "qwen/qwen-2.5-72b-instruct",
    "nemotron":      "nvidia/nemotron-3-ultra-550b-a55b:free",
    "mistral_small": "mistralai/mistral-small-2603",
}
