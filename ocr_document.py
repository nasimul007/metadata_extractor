import fitz  # pymupdf
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


class PDFTextExtractor:
    def __init__(self):
        pass

    def extract_text(self, pdf_path):
        text = self._extract_digital_text(pdf_path)

        # If enough text found, return it
        if len(text.strip()) > 100:
            return text

        print("Scanned PDF detected. Running OCR...")
        return self._extract_ocr_text(pdf_path)

    def _extract_digital_text(self, pdf_path):
        text = ""

        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

        return text

    def _extract_ocr_text(self, pdf_path):
        text = ""

        images = convert_from_path(
            pdf_path,
            dpi=300
        )

        for page_no, image in enumerate(images, start=1):
            page_text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

            text += f"\n\n--- PAGE {page_no} ---\n"
            text += page_text

        return text
    

def get_extracted_text():
    pdf_file_path = "/home/nasimul/Documents/Personal/AI worksop/TASK metadata extractor/metadata_extractor/docs/sample-pdf-invoice.pdf"
    extractor = PDFTextExtractor()
    text = extractor.extract_text(pdf_file_path)

    return text