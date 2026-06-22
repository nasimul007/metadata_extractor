import os
import fitz  # pymupdf
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


class DocumentTextExtractor:

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp"
    }

    def extract_text(self, file_path):

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            return self._extract_pdf_text(file_path)

        elif extension in self.IMAGE_EXTENSIONS:
            return self._extract_image_text(file_path)

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    def _extract_pdf_text(self, pdf_path):

        text = self._extract_digital_pdf_text(pdf_path)

        # If text exists, return it
        if len(text.strip()) > 100:
            return text

        # Otherwise OCR
        return self._extract_pdf_ocr_text(pdf_path)

    def _extract_digital_pdf_text(self, pdf_path):

        text = ""

        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

        return text

    def _extract_pdf_ocr_text(self, pdf_path):

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

    def _extract_image_text(self, image_path):

        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )

        return text
    

def get_extracted_text():
    file_path = "/home/nasimul/Documents/Personal/AI worksop/TASK metadata extractor/metadata_extractor/docs/photo_2026-06-22_08-50-29.jpg"
    extractor = DocumentTextExtractor()
    text = extractor.extract_text(file_path)

    return text