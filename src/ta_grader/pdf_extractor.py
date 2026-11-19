import os
import base64
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .smart_pdf_extractor import fix_rtl_text


from .config import VISION_MODEL, get_client, student_id_for


@dataclass
class PageContent:
    page_num: int
    text: str
    image_descriptions: List[str]


class PDFExtractor:
    """Extracts text and describes images/tables/graphs from PDFs using vision LLM."""
    
    def __init__(self, fix_rtl: bool = True):
        self.client = get_client()
        # Persian PDFs often store right-to-left lines in visual order.
        self.fix_rtl = fix_rtl
    
    def extract_pdf(self, pdf_path: str, describe_images: bool = True) -> str:
        """
        Extract full content from PDF.
        - Extracts text from each page
        - Extracts images and sends them to vision LLM for description
        - Returns combined text with image descriptions inline
        """
        doc = fitz.open(pdf_path)
        pages_content: List[PageContent] = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text("text")
            if self.fix_rtl:
                text = fix_rtl_text(text)
            
            # Extract and describe images
            image_descriptions = []
            if describe_images:
                images = self._extract_page_images(page, doc)
                for img_data in images:
                    desc = self._describe_image(img_data)
                    if desc:
                        image_descriptions.append(desc)
            
            pages_content.append(PageContent(
                page_num=page_num + 1,
                text=text,
                image_descriptions=image_descriptions
            ))
        
        doc.close()
        
        # Combine into single text
        return self._combine_content(pages_content)
    
    def _extract_page_images(self, page, doc) -> List[bytes]:
        """Extract images from a PDF page as bytes."""
        images = []
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
            except Exception:
                continue
        
        # Also render the page as image to catch tables/graphs that aren't separate images
        # Only if page has significant non-text content
        if self._page_has_complex_content(page):
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality
                images.append(pix.tobytes("png"))
            except Exception:
                pass
        
        return images
    
    def _page_has_complex_content(self, page) -> bool:
        """Check if page likely has tables, diagrams, or complex layouts."""
        # Check for drawings (vector graphics)
        drawings = page.get_drawings()
        if len(drawings) > 10:  # Likely has diagrams/tables
            return True
        
        # Check text block layout complexity
        blocks = page.get_text("blocks")
        if len(blocks) > 5:  # Multiple text blocks might indicate tables
            # Check if blocks are arranged in grid-like pattern
            x_positions = set()
            for block in blocks:
                x_positions.add(round(block[0], -1))  # Round to nearest 10
            if len(x_positions) > 2:  # Multiple columns
                return True
        
        return False
    
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
                mime_type = "image/png"  # Default
            
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are describing images from a student's homework/exam paper.
Describe what you see in detail, focusing on:
- Mathematical equations, formulas, or expressions
- Graphs, charts, and their data points/trends
- Tables and their content
- Diagrams and their components
- Handwritten answers or solutions
- Any text visible in the image

Be precise and technical. If it's a math problem, write out the equations.
If it's a graph, describe axes, labels, and key points.
Keep description concise but complete."""
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
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error describing image: {e}")
            return ""
    
    def _combine_content(self, pages: List[PageContent]) -> str:
        """Combine all page content into a single string."""
        parts = []
        
        for page in pages:
            page_text = f"--- Page {page.page_num} ---\n"
            page_text += page.text.strip()
            
            if page.image_descriptions:
                page_text += "\n\n[IMAGES/FIGURES ON THIS PAGE:]"
                for i, desc in enumerate(page.image_descriptions, 1):
                    page_text += f"\n\n[Figure {i}]: {desc}"
            
            parts.append(page_text)
        
        return "\n\n".join(parts)


def extract_all_pdfs(input_folder: str, output_folder: str, describe_images: bool = True):
    """
    Extract text from all PDFs in a folder.
    Saves extracted text as .txt files with same name.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFExtractor()
    
    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            content = extractor.extract_pdf(str(pdf_file), describe_images=describe_images)
            
            output_file = output_path / f"{student_id_for(pdf_file)}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  Saved: {output_file.name}")
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Example usage
    extract_all_pdfs("students_pdf", "students_extracted", describe_images=True)
