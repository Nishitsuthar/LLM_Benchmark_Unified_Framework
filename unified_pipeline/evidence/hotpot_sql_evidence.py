"""HotpotQA SQL-detour evidence builder — Sushma.

Queries a pre-built SQLite fact database (hotpot_facts.db) instead of
passing raw Wikipedia documents.  Facts are stored as subject/relation/object
triples extracted from HotpotQA context, so the answering model reasons over
structured data rather than free-form paragraphs.

Comparison purpose: tests whether structured retrieval (SQL) beats LLM reading
comprehension (hotpot_context) on multi-hop HotpotQA questions.

Dataset YAML must supply:
  hotpot_facts_db: data/hotpot_facts.db
  questions_file:  evidence/hotpot/questions.json
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult

_QUERY = """
SELECT d.title,
       f.subject,
       f.relation,
       f.object,
       f.evidence
FROM   facts f
JOIN   documents d ON f.document_id = d.document_id
WHERE  d.question_id = :question_id
ORDER  BY d.document_order, f.sentence_index
"""


class HotpotSqlEvidenceBuilder(BaseEvidenceBuilder):
    def __init__(self):
        self._id_map: dict[str, str] = {}   # BQ* question_id → HotpotQA source_id
        self._loaded_questions_file: str = ""
        self._db_path: str = ""
        self._conn: sqlite3.Connection | None = None

    def build(self, question: str, config: dict) -> EvidenceResult:
        db_path       = config["hotpot_facts_db"]
        questions_file = config["questions_file"]

        if questions_file != self._loaded_questions_file:
            self._load_id_map(questions_file)
            self._loaded_questions_file = questions_file

        if db_path != self._db_path:
            if self._conn:
                self._conn.close()
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._db_path = db_path

        question_id = config.get("_question_id", "")
        source_id   = self._id_map.get(question_id, "")

        if not source_id:
            raise KeyError(
                f"No source_id mapping found for question_id={question_id!r}. "
                f"Check {questions_file}."
            )

        rows = self._conn.execute(_QUERY, {"question_id": source_id}).fetchall()

        if not rows:
            raise ValueError(
                f"No facts in DB for question_id={question_id!r} "
                f"(source_id={source_id!r}). "
                f"Run build_fact_database.py to populate {db_path}."
            )

        context = self._format_facts(rows)
        return EvidenceResult(
            text=context,
            metadata={
                "question_id": question_id,
                "source_id":   source_id,
                "fact_count":  len(rows),
            },
        )

    def _load_id_map(self, questions_file: str) -> None:
        questions = json.loads(Path(questions_file).read_text(encoding="utf-8"))
        self._id_map = {q["id"]: q["source_id"] for q in questions}

    @staticmethod
    def _format_facts(rows) -> str:
        parts: list[str] = []
        current_title = None
        for row in rows:
            if row["title"] != current_title:
                current_title = row["title"]
                parts.append(f"\n[{current_title}]")
            parts.append(
                f"  {row['subject']} | {row['relation']} | {row['object']}"
                f"\n    Evidence: {row['evidence']}"
            )
        return "\n".join(parts).strip()
