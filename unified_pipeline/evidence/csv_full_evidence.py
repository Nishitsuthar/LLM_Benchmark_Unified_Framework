"""CSV full-context evidence builder — Krittika.

Serializes the entire CSV file into the prompt as-is, so the model
sees all 878 rows. Contrasts with CsvEvidenceBuilder which does
retrieval-based row filtering.
"""

from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class CsvFullEvidenceBuilder(BaseEvidenceBuilder):
    def build(self, question: str, config: dict) -> EvidenceResult:
        data_path = Path(config["data_file"])
        if not data_path.exists():
            raise FileNotFoundError(f"CSV data file not found: {data_path}")

        csv_text = data_path.read_text(encoding="utf-8")
        row_count = csv_text.count("\n")  # approximate row count (includes header)

        return EvidenceResult(
            text=csv_text,
            metadata={"row_count": row_count, "data_file": str(data_path)},
        )
