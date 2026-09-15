"""HotpotQA context evidence builder — Sushma.

Reads context documents from the HotpotQA distractor-split JSON file and
formats them as plain numbered text for injection into the prompt.

The distractor JSON must be downloaded once via:
    python ../hotpot-structured-benchmark/scripts/download_distractor.py
and placed at the path specified by `hotpot_data_file` in the dataset YAML.

Ported from Sushma's Sprint 3/4 hotpot-structured-benchmark pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class HotpotEvidenceBuilder(BaseEvidenceBuilder):
    def __init__(self):
        self._index: dict[str, str] = {}  # source_id → formatted context
        self._id_map: dict[str, str] = {}  # question_id (BQ*) → source_id
        self._loaded_data_file: str = ""

    def build(self, question: str, config: dict) -> EvidenceResult:
        data_file = config["hotpot_data_file"]
        if data_file != self._loaded_data_file:
            self._load(data_file, config["questions_file"])
            self._loaded_data_file = data_file

        question_id = config.get("_question_id", "")
        source_id   = self._id_map.get(question_id, "")
        context     = self._index.get(source_id, "")

        if not context:
            raise KeyError(
                f"No HotpotQA context found for question_id={question_id!r} "
                f"(source_id={source_id!r}). "
                f"Is '{data_file}' downloaded? Run scripts/download_distractor.py."
            )

        return EvidenceResult(
            text=context,
            metadata={"question_id": question_id, "source_id": source_id},
        )

    def _load(self, data_file: str, questions_file: str) -> None:
        path = Path(data_file)
        if not path.exists():
            raise FileNotFoundError(
                f"HotpotQA data file not found: {path}\n"
                "Download it with: python scripts/download_distractor.py "
                "(from the hotpot-structured-benchmark directory)"
            )

        hotpot_data: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        self._index = {item["id"]: self._format_context(item) for item in hotpot_data}

        questions: list[dict] = json.loads(Path(questions_file).read_text(encoding="utf-8"))
        self._id_map = {q["id"]: q["source_id"] for q in questions}

    @staticmethod
    def _format_context(item: dict) -> str:
        ctx = item.get("context", {})
        titles     = ctx.get("title", [])
        sent_groups = ctx.get("sentences", [])
        parts = []
        for title, sents in zip(titles, sent_groups):
            parts.append(f"[{title}]")
            for i, s in enumerate(sents):
                parts.append(f"  [{i}] {s}")
        return "\n".join(parts)
