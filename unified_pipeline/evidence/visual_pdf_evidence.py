"""Visual PDF evidence builder — Arpitha.

Renders visual PDF pages into JPEG evidence images for multimodal model input.

Historical IMDb benchmark:
- alternating metadata/cast pages
- every two selected pages combined vertically

Final45 benchmark:
- metadata-only pages
- optionally combines every four pages into a 2x2 grid
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

from unified_pipeline.base import BaseEvidenceBuilder, EvidenceResult


class VisualPdfEvidenceBuilder(BaseEvidenceBuilder):
    """Prepare visual-PDF pages for multimodal model input."""

    def __init__(self) -> None:
        # Cache rendered images because several questions use the same evidence.
        self._cache: dict[tuple, list[str]] = {}

    def build(self, question: str, config: dict) -> EvidenceResult:
        pdf_path = Path(config["evidence_path"])

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Visual PDF evidence not found: {pdf_path}"
            )

        pdf_page_count = int(config["pdf_page_count"])
        page_layout = config.get("page_layout", "alternating")

        if page_layout not in {"alternating", "metadata_only"}:
            raise ValueError(
                f"Unknown visual PDF page_layout: {page_layout!r}"
            )

        pages_per_image = int(config.get("pages_per_image", 2))

        if pages_per_image not in {1, 2, 4}:
            raise ValueError(
                "pages_per_image must be 1, 2, or 4"
            )

        if pdf_page_count <= 0:
            raise ValueError(
                f"pdf_page_count must be positive; got {pdf_page_count}."
            )

        if page_layout == "alternating" and pdf_page_count % 2 != 0:
            raise ValueError(
                "Alternating visual PDF layout requires an even "
                f"pdf_page_count; got {pdf_page_count}."
            )

        batch_page_indices = {
            "metadata": list(range(0, pdf_page_count, 2)),
            "cast": list(range(1, pdf_page_count, 2)),
        }

        if page_layout == "metadata_only":
            batch_page_indices = {
                "metadata": list(range(pdf_page_count))
            }

        batch_name = config.get("_source_pdf")

        if batch_name not in batch_page_indices:
            raise ValueError(
                f"Unknown visual evidence batch: {batch_name!r}. "
                f"Expected one of {list(batch_page_indices)}."
            )

        render_scale = float(config.get("render_scale", 1.0))
        jpeg_quality = int(config.get("jpeg_quality", 75))

        cache_key = (
            str(pdf_path.resolve()),
            batch_name,
            pdf_page_count,
            render_scale,
            jpeg_quality,
            page_layout,
            pages_per_image,
        )

        if cache_key not in self._cache:
            self._cache[cache_key] = self._prepare_images(
                pdf_path=pdf_path,
                page_indices=batch_page_indices[batch_name],
                pdf_page_count=pdf_page_count,
                render_scale=render_scale,
                jpeg_quality=jpeg_quality,
                pages_per_image=pages_per_image,
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
    def _combine_four_pages_grid(
        pages: list[Image.Image],
    ) -> Image.Image:
        """Combine four rendered pages into a 2x2 grid."""

        if len(pages) != 4:
            raise ValueError(
                "Exactly four pages are required for a 2x2 grid."
            )

        gap = 20

        left_width = max(pages[0].width, pages[2].width)
        right_width = max(pages[1].width, pages[3].width)

        top_height = max(pages[0].height, pages[1].height)
        bottom_height = max(pages[2].height, pages[3].height)

        width = left_width + gap + right_width
        height = top_height + gap + bottom_height

        combined = Image.new(
            "RGB",
            (width, height),
            "white",
        )

        positions = [
            (0, 0),
            (left_width + gap, 0),
            (0, top_height + gap),
            (left_width + gap, top_height + gap),
        ]

        for image, position in zip(pages, positions):
            combined.paste(image, position)

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
        pdf_page_count: int,
        render_scale: float,
        jpeg_quality: int,
        pages_per_image: int = 2,
    ) -> list[str]:
        """Render selected PDF pages and group them into model images."""

        document = pymupdf.open(str(pdf_path))

        try:
            if document.page_count < pdf_page_count:
                raise ValueError(
                    f"Configured pdf_page_count is {pdf_page_count}, "
                    f"but the actual PDF contains only "
                    f"{document.page_count} pages."
                )

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

        if pages_per_image == 1:
            return [
                self._image_to_data_url(page, jpeg_quality)
                for page in rendered_pages
            ]

        if len(rendered_pages) % pages_per_image != 0:
            raise ValueError(
                f"Selected page count ({len(rendered_pages)}) "
                f"must be divisible by pages_per_image "
                f"({pages_per_image})."
            )

        images: list[str] = []

        if pages_per_image == 2:
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

        # pages_per_image == 4
        for index in range(0, len(rendered_pages), 4):
            combined = self._combine_four_pages_grid(
                rendered_pages[index:index + 4]
            )

            images.append(
                self._image_to_data_url(
                    combined,
                    jpeg_quality=jpeg_quality,
                )
            )

        return images