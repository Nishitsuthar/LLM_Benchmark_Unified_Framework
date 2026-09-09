"""Visual PDF evidence builder — Arpitha.

Renders the IMDb screenshot PDF into JPEG evidence images for
multimodal model input.

The focused IMDb benchmark has 40 PDF pages:
- metadata batch: pages 1, 3, 5, ..., 39
- cast batch:     pages 2, 4, 6, ..., 40

Every two selected PDF pages are combined vertically, giving
10 evidence images per batch.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


BATCH_PAGE_INDICES = {
    "metadata": list(range(0, 40, 2)),
    "cast": list(range(1, 40, 2)),
}


class VisualPdfEvidenceBuilder(BaseEvidenceBuilder):
    """Prepare screenshot-PDF pages for multimodal model input."""

    def __init__(self) -> None:
        # Cache rendered images because several questions use the same batch.
        self._cache: dict[tuple, list[str]] = {}

    def build(self, question: str, config: dict) -> EvidenceResult:
        pdf_path = Path(config["evidence_path"])

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Visual PDF evidence not found: {pdf_path}"
            )

        batch_name = config.get("_source_pdf")

        if batch_name not in BATCH_PAGE_INDICES:
            raise ValueError(
                f"Unknown visual evidence batch: {batch_name!r}. "
                f"Expected one of {list(BATCH_PAGE_INDICES)}."
            )

        render_scale = float(config.get("render_scale", 1.0))
        jpeg_quality = int(config.get("jpeg_quality", 75))

        cache_key = (
            str(pdf_path.resolve()),
            batch_name,
            render_scale,
            jpeg_quality,
        )

        if cache_key not in self._cache:
            self._cache[cache_key] = self._prepare_images(
                pdf_path=pdf_path,
                page_indices=BATCH_PAGE_INDICES[batch_name],
                render_scale=render_scale,
                jpeg_quality=jpeg_quality,
            )

        images = self._cache[cache_key]

        return EvidenceResult(
            text="",
            metadata={
                "images": images,
                "batch": batch_name,
                "image_count": len(images),
                "source_pdf": str(pdf_path),
            },
        )

    @staticmethod
    def _render_page(
        document: pymupdf.Document,
        page_index: int,
        render_scale: float,
    ) -> Image.Image:
        """Render one PDF page as an RGB PIL image."""

        if page_index >= document.page_count:
            raise ValueError(
                f"Page index {page_index} is outside the PDF. "
                f"PDF contains {document.page_count} pages."
            )

        page = document.load_page(page_index)

        matrix = pymupdf.Matrix(
            render_scale,
            render_scale,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        return Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )

    @staticmethod
    def _combine_two_pages(
        first: Image.Image,
        second: Image.Image,
    ) -> Image.Image:
        """Combine two rendered pages vertically."""

        width = max(first.width, second.width)
        gap = 20
        height = first.height + gap + second.height

        combined = Image.new(
            "RGB",
            (width, height),
            "white",
        )

        first_x = (width - first.width) // 2
        second_x = (width - second.width) // 2

        combined.paste(
            first,
            (first_x, 0),
        )

        combined.paste(
            second,
            (second_x, first.height + gap),
        )

        return combined

    @staticmethod
    def _image_to_data_url(
        image: Image.Image,
        jpeg_quality: int,
    ) -> str:
        """Encode a PIL image as a base64 JPEG data URL."""

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    def _prepare_images(
        self,
        pdf_path: Path,
        page_indices: list[int],
        render_scale: float,
        jpeg_quality: int,
    ) -> list[str]:
        """Render selected pages and combine every two into one image."""

        document = pymupdf.open(str(pdf_path))

        try:
            rendered_pages = [
                self._render_page(
                    document=document,
                    page_index=page_index,
                    render_scale=render_scale,
                )
                for page_index in page_indices
            ]
        finally:
            document.close()

        if len(rendered_pages) % 2 != 0:
            raise ValueError(
                "Visual evidence page count must be even so pages "
                "can be combined in pairs."
            )

        images: list[str] = []

        for index in range(0, len(rendered_pages), 2):
            combined = self._combine_two_pages(
                rendered_pages[index],
                rendered_pages[index + 1],
            )

            images.append(
                self._image_to_data_url(
                    combined,
                    jpeg_quality=jpeg_quality,
                )
            )

        return images