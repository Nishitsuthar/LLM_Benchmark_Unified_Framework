from pathlib import Path

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class ProseEvidenceBuilder(BaseEvidenceBuilder):
    """
    Loads all .txt documents from the evidence folder and concatenates them
    into a single text block separated by '---' dividers.

    Used by: music.yaml, f1_racing.yaml  (evidence_type: prose_docs)
    """

    def build(self, question: str, config: dict) -> EvidenceResult:
        evidence_path = Path(config["evidence_path"])
        txt_files = sorted(evidence_path.glob("*.txt"))

        if not txt_files:
            return EvidenceResult(
                text="",
                metadata={"file_count": 0, "files": []},
            )

        sections = [f.read_text(encoding="utf-8").strip() for f in txt_files]
        combined = "\n\n---\n\n".join(sections)

        return EvidenceResult(
            text=combined,
            metadata={
                "file_count": len(txt_files),
                "files": [f.name for f in txt_files],
            },
        )
