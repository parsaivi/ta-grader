"""
Smart PDF Extractor - Hybrid Text + Vision Approach

This module extracts text from PDFs using direct text extraction (no LLM cost),
and only uses LLM for describing images/tables/graphs that cannot be extracted as text.

Key features:
- Extracts text blocks with their positions using PyMuPDF
- Detects images and their positions
- Detects tables using PyMuPDF's table detection
- Merges all elements by Y-position to maintain correct reading order
- Only calls vision LLM for non-text elements (images, complex tables, graphs)
"""

import base64
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz
from tqdm import tqdm

from .config import VISION_MODEL, get_client, student_id_for

# Arabic, Persian and Hebrew ranges, plus Arabic presentation forms.
_RTL_CHARS = re.compile(
    r"[֐-׿؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)


def is_rtl_line(line: str) -> bool:
    """True if the line is predominantly right-to-left script."""
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    rtl = sum(1 for c in letters if _RTL_CHARS.match(c))
    return rtl > len(letters) / 2


def fix_rtl_text(text: str) -> str:
    """
    Repair word order in right-to-left lines.

    Some PDFs store an RTL line in visual order, so reading it left-to-right
    yields the words backwards: "به نام خدا" comes out as "خدا نام به". That
    wrecks both the LLM's comprehension and our question-heading detection.

    Reversing the word order of predominantly-RTL lines restores logical order.
    Latin runs and numbers inside the line keep their internal spelling, since
    only whole tokens are reordered.
    """
    fixed_lines = []
    for line in text.splitlines():
        if is_rtl_line(line):
            fixed_lines.append(" ".join(reversed(line.split())))
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


class ElementType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class PageElement:
    """Represents an element on a PDF page with its position."""
    element_type: ElementType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    content: str = ""  # For text elements, this is the text itself
    image_bytes: Optional[bytes] = None  # For images, the raw bytes
    table_data: Optional[List[List[str]]] = None  # For tables, the cell data

    @property
    def y_position(self) -> float:
        """Return the top Y position for sorting."""
        return self.bbox[1]

    @property
    def x_position(self) -> float:
        """Return the left X position."""
        return self.bbox[0]


@dataclass
class ExtractionStats:
    """Statistics about the extraction process."""
    total_pages: int = 0
    text_blocks: int = 0
    images_found: int = 0
    images_described: int = 0
    tables_found: int = 0
    tables_described: int = 0
    llm_calls: int = 0


class SmartPDFExtractor:
    """
    Extracts content from PDFs using a hybrid approach:
    - Text is extracted directly (no LLM cost)
    - Images/tables/graphs are described using vision LLM
    - All elements are merged by position to maintain reading order
    """

    def __init__(self, vision_model: str = VISION_MODEL, fix_rtl: bool = True):
        self.client = get_client()
        self.vision_model = vision_model
        # Persian submissions are the norm for this course, and many PDFs store
        # RTL lines in visual order. Repair them unless told otherwise.
        self.fix_rtl = fix_rtl
        self.stats = ExtractionStats()

    def extract_pdf(self, pdf_path: str, describe_visuals: bool = True) -> str:
        """
        Extract content from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            describe_visuals: Whether to use LLM to describe images/tables

        Returns:
            Extracted text with image/table descriptions in correct positions
        """
        self.stats = ExtractionStats()
        doc = fitz.open(pdf_path)
        self.stats.total_pages = len(doc)

        all_content_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements = self._extract_page_elements(page, page_num, doc)

            # Sort elements by Y position (top to bottom), then X (left to right)
            page_elements.sort(key=lambda e: (e.y_position, e.x_position))

            # Process each element
            page_content = self._process_page_elements(
                page_elements,
                page_num,
                describe_visuals
            )

            if page_content.strip():
                all_content_parts.append(f"--- Page {page_num + 1} ---\n{page_content}")

        doc.close()

        return "\n\n".join(all_content_parts)

    def _extract_page_elements(
        self,
        page: fitz.Page,
        page_num: int,
        doc: fitz.Document
    ) -> List[PageElement]:
        """Extract all elements (text, images, tables) from a page with positions."""
        elements = []

        # 1. Extract text blocks with positions
        text_elements = self._extract_text_blocks(page, page_num)
        elements.extend(text_elements)

        # 2. Extract images with positions
        image_elements = self._extract_images(page, page_num, doc)
        elements.extend(image_elements)

        # 3. Detect and extract tables
        table_elements = self._extract_tables(page, page_num)
        elements.extend(table_elements)

        # 4. Remove text elements that overlap with tables (tables take precedence)
        elements = self._remove_overlapping_text(elements)

        return elements

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[PageElement]:
        """Extract text blocks with their bounding boxes."""
        elements = []

        # Get text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")

        for block in blocks:
            # The block_type flag lives at index 6, so the tuple must actually
            # have 7 entries before we read it.
            if len(block) < 7 or block[6] != 0:  # block_type 0 = text
                continue

            x0, y0, x1, y1 = block[:4]
            text = block[4]

            if not text.strip():
                continue

            if self.fix_rtl:
                text = fix_rtl_text(text)

            self.stats.text_blocks += 1
            elements.append(PageElement(
                element_type=ElementType.TEXT,
                bbox=(x0, y0, x1, y1),
                page_num=page_num,
                content=text.strip()
            ))

        return elements

    def _extract_images(
        self,
        page: fitz.Page,
        page_num: int,
        doc: fitz.Document
    ) -> List[PageElement]:
        """Extract images with their bounding boxes."""
        elements = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                # Get image bounding box on the page
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                # Use the first occurrence of this image
                rect = img_rects[0]
                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

                # Skip very small images (likely icons or artifacts)
                width = rect.x1 - rect.x0
                height = rect.y1 - rect.y0
                if width < 50 or height < 50:
                    continue

                # Extract image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                self.stats.images_found += 1
                elements.append(PageElement(
                    element_type=ElementType.IMAGE,
                    bbox=bbox,
                    page_num=page_num,
                    image_bytes=image_bytes
                ))

            except Exception:
                continue

        # Also check for drawings/vector graphics that might be diagrams
        drawings = page.get_drawings()
        if len(drawings) > 20:  # Significant number of vector elements
            # This might be a diagram - render that region
            drawing_rects = [d["rect"] for d in drawings if "rect" in d]
            if drawing_rects:
                # Find bounding box of all drawings
                combined_rect = self._get_combined_rect(drawing_rects)
                if combined_rect:
                    # Check if this region overlaps significantly with existing images
                    if not self._overlaps_with_elements(combined_rect, elements):
                        # Render this region as an image
                        clip_rect = fitz.Rect(combined_rect)
                        try:
                            pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(2, 2))
                            image_bytes = pix.tobytes("png")

                            self.stats.images_found += 1
                            elements.append(PageElement(
                                element_type=ElementType.IMAGE,
                                bbox=combined_rect,
                                page_num=page_num,
                                image_bytes=image_bytes
                            ))
                        except Exception:
                            pass

        return elements

    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[PageElement]:
        """Extract tables using PyMuPDF's table detection."""
        elements = []

        try:
            # PyMuPDF 1.23+ has find_tables()
            tables = page.find_tables()

            for table in tables:
                bbox = table.bbox  # (x0, y0, x1, y1)

                # Extract table data as list of lists
                table_data = []
                for row in table.extract():
                    cleaned_row = [cell if cell else "" for cell in row]
                    table_data.append(cleaned_row)

                # Check if table has meaningful content
                has_content = any(
                    any(cell.strip() for cell in row)
                    for row in table_data
                )

                if has_content:
                    self.stats.tables_found += 1
                    elements.append(PageElement(
                        element_type=ElementType.TABLE,
                        bbox=bbox,
                        page_num=page_num,
                        table_data=table_data
                    ))

        except AttributeError:
            # Older PyMuPDF version without find_tables()
            pass
        except Exception:
            pass

        return elements

    def _remove_overlapping_text(self, elements: List[PageElement]) -> List[PageElement]:
        """Remove text elements that overlap significantly with tables."""
        table_rects = [
            fitz.Rect(e.bbox) for e in elements
            if e.element_type == ElementType.TABLE
        ]

        if not table_rects:
            return elements

        filtered = []
        for element in elements:
            if element.element_type == ElementType.TEXT:
                text_rect = fitz.Rect(element.bbox)
                overlaps = False
                for table_rect in table_rects:
                    # Check if text is inside or mostly overlaps with table
                    intersection = text_rect & table_rect
                    if intersection.is_empty:
                        continue
                    overlap_ratio = intersection.get_area() / text_rect.get_area()
                    if overlap_ratio > 0.5:  # More than 50% overlap
                        overlaps = True
                        break

                if not overlaps:
                    filtered.append(element)
            else:
                filtered.append(element)

        return filtered

    def _process_page_elements(
        self,
        elements: List[PageElement],
        page_num: int,
        describe_visuals: bool
    ) -> str:
        """Process elements and generate text output."""
        parts = []

        for element in elements:
            if element.element_type == ElementType.TEXT:
                parts.append(element.content)

            elif element.element_type == ElementType.IMAGE:
                if describe_visuals and element.image_bytes:
                    description = self._describe_image(element.image_bytes)
                    if description:
                        self.stats.images_described += 1
                        parts.append(f"\n[IMAGE DESCRIPTION]: {description}\n")
                else:
                    parts.append("\n[IMAGE: Visual content present]\n")

            elif element.element_type == ElementType.TABLE:
                # First, try to format table as text
                table_text = self._format_table_as_text(element.table_data)

                if table_text:
                    parts.append(f"\n[TABLE]:\n{table_text}\n")
                elif describe_visuals:
                    # If table is complex, use LLM to describe it
                    # Render the table region as image and describe
                    self.stats.tables_described += 1
                    parts.append("\n[TABLE: Complex table present]\n")

        return "\n".join(parts)

    def _format_table_as_text(self, table_data: List[List[str]]) -> Optional[str]:
        """Format table data as readable text."""
        if not table_data:
            return None

        # Calculate column widths
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = 0
            for row in table_data:
                if col_idx < len(row):
                    max_width = max(max_width, len(str(row[col_idx])))
            col_widths.append(min(max_width, 50))  # Cap at 50 chars

        # Format rows
        lines = []
        for row_idx, row in enumerate(table_data):
            cells = []
            for col_idx, cell in enumerate(row):
                cell_str = str(cell)[:50] if cell else ""
                if col_idx < len(col_widths):
                    cells.append(cell_str.ljust(col_widths[col_idx]))
                else:
                    cells.append(cell_str)
            lines.append(" | ".join(cells))

            # Add separator after header row
            if row_idx == 0:
                separator = "-+-".join("-" * w for w in col_widths)
                lines.append(separator)

        return "\n".join(lines)

    def _describe_image(self, image_bytes: bytes) -> str:
        """Send image to vision LLM and get description."""
        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Detect image type
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            elif image_bytes[:2] == b'\xff\xd8':
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"

            self.stats.llm_calls += 1

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are describing images from a student's homework/exam paper.
Describe what you see concisely but completely, focusing on:
- Mathematical equations, formulas, or expressions (write them out)
- Graphs, charts: describe axes, labels, data points, trends
- Diagrams: describe components and their relationships
- Handwritten answers or solutions
- Any text visible in the image

Be precise and technical. Keep description under 200 words."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this image from a homework submission:"
                            }
                        ]
                    }
                ],
                max_tokens=400
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error describing image: {e}")
            return ""

    def _get_combined_rect(
        self,
        rects: List[fitz.Rect]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get the bounding box that contains all given rectangles."""
        if not rects:
            return None

        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)

        return (x0, y0, x1, y1)

    def _overlaps_with_elements(
        self,
        bbox: Tuple[float, float, float, float],
        elements: List[PageElement]
    ) -> bool:
        """Check if bbox overlaps significantly with existing elements."""
        rect = fitz.Rect(bbox)

        for element in elements:
            if element.element_type == ElementType.IMAGE:
                element_rect = fitz.Rect(element.bbox)
                intersection = rect & element_rect
                if not intersection.is_empty:
                    overlap = intersection.get_area() / min(
                        rect.get_area(),
                        element_rect.get_area()
                    )
                    if overlap > 0.3:
                        return True

        return False

    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return {
            "total_pages": self.stats.total_pages,
            "text_blocks": self.stats.text_blocks,
            "images_found": self.stats.images_found,
            "images_described": self.stats.images_described,
            "tables_found": self.stats.tables_found,
            "tables_described": self.stats.tables_described,
            "llm_calls": self.stats.llm_calls
        }


def extract_all_pdfs(
    input_folder: str,
    output_folder: str,
    describe_visuals: bool = True,
    vision_model: str = VISION_MODEL,
    fix_rtl: bool = True
) -> Dict[str, Any]:
    """
    Extract text from all PDFs in a folder using smart hybrid extraction.

    Args:
        input_folder: Folder containing PDF files
        output_folder: Folder to save extracted text files
        describe_visuals: Whether to use LLM for images/tables
        vision_model: Model to use for vision tasks
        fix_rtl: Repair word order in right-to-left (Persian) lines

    Returns:
        Summary statistics
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = SmartPDFExtractor(vision_model=vision_model, fix_rtl=fix_rtl)

    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    total_stats = {
        "processed": 0,
        "failed": 0,
        "total_llm_calls": 0,
        "total_images": 0,
        "total_tables": 0
    }

    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        try:
            content = extractor.extract_pdf(str(pdf_file), describe_visuals=describe_visuals)

            output_file = output_path / f"{student_id_for(pdf_file)}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            stats = extractor.get_stats()
            total_stats["processed"] += 1
            total_stats["total_llm_calls"] += stats["llm_calls"]
            total_stats["total_images"] += stats["images_found"]
            total_stats["total_tables"] += stats["tables_found"]

        except Exception as e:
            print(f"\nError processing {pdf_file.name}: {e}")
            total_stats["failed"] += 1

    print("\n=== Extraction Complete ===")
    print(f"Processed: {total_stats['processed']} files")
    print(f"Failed: {total_stats['failed']} files")
    print(f"Total LLM calls: {total_stats['total_llm_calls']}")
    print(f"Total images found: {total_stats['total_images']}")
    print(f"Total tables found: {total_stats['total_tables']}")

    return total_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart PDF Extractor - Hybrid text + vision extraction"
    )
    parser.add_argument(
        "--input", "-i",
        default="students_pdf",
        help="Input folder containing PDF files"
    )
    parser.add_argument(
        "--output", "-o",
        default="students_extracted",
        help="Output folder for extracted text files"
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip LLM vision calls (only extract text)"
    )
    parser.add_argument(
        "--model",
        default=VISION_MODEL,
        help=f"Vision model to use (default: {VISION_MODEL})"
    )
    parser.add_argument(
        "--single",
        help="Process a single PDF file instead of folder"
    )
    parser.add_argument(
        "--no-rtl-fix",
        action="store_true",
        help="Do not reverse word order on right-to-left lines"
    )

    args = parser.parse_args()

    if args.single:
        # Process single file
        extractor = SmartPDFExtractor(vision_model=args.model, fix_rtl=not args.no_rtl_fix)
        print(f"Processing: {args.single}")

        content = extractor.extract_pdf(args.single, describe_visuals=not args.no_vision)

        # Save output
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{Path(args.single).stem}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Saved to: {output_file}")
        stats = extractor.get_stats()
        print(f"Stats: {stats}")
    else:
        # Process folder
        extract_all_pdfs(
            input_folder=args.input,
            output_folder=args.output,
            describe_visuals=not args.no_vision,
            vision_model=args.model,
            fix_rtl=not args.no_rtl_fix
        )
