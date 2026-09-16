from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class ProseEvidenceBuilder(BaseEvidenceBuilder):
    """
    Loads .txt documents from the evidence folder.

    If 'question_doc_file' is set in config (injected by run.py per-question),
    loads only that single file — avoids passing hundreds of docs to the model.
    Falls back to loading all .txt files if not specified.

    Used by: music.yaml, f1_racing.yaml  (evidence_type: prose_docs)
    """

    def build(self, question: str, config: dict) -> EvidenceResult:
        evidence_path = Path(config["evidence_path"])

        # Per-question single-document mode (set by run.py from questions.json doc_file field)
        doc_file = config.get("question_doc_file", "")
        if doc_file:
            target = evidence_path / doc_file
            if target.exists():
                return EvidenceResult(
                    text=target.read_text(encoding="utf-8").strip(),
                    metadata={"file_count": 1, "files": [doc_file]},
                )

        # Fallback: load all .txt files in the folder
        txt_files = sorted(evidence_path.glob("*.txt"))

        if not txt_files:
            return EvidenceResult(text="", metadata={"file_count": 0, "files": []})

        sections = [f.read_text(encoding="utf-8").strip() for f in txt_files]
        combined = "\n\n---\n\n".join(sections)

        return EvidenceResult(
            text=combined,
            metadata={
                "file_count": len(txt_files),
                "files": [f.name for f in txt_files],
            },
        )
