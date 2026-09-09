"""CSV evidence builder — Krittika.

Serializes the full shared CSV as plain text so the model can reason
directly over the tabular data. Ported from dataframe_to_prompt_csv()
in run_experiment.py.
"""

from __future__ import annotations

import pandas as pd
from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class CsvEvidenceBuilder(BaseEvidenceBuilder):
    def build(self, question: str, config: dict) -> EvidenceResult:
        df = pd.read_csv(config["evidence_path"])

        max_rows = config.get("max_rows", 0)
        df_for_prompt = df if max_rows <= 0 else df.head(max_rows)
        csv_text = df_for_prompt.to_csv(index=False)

        if max_rows > 0 and len(df) > max_rows:
            csv_text += f"\n[Only the first {max_rows} rows shown out of {len(df)} total.]\n"

        return EvidenceResult(
            text=csv_text,
            metadata={"rows": len(df), "columns": list(df.columns)},
        )
