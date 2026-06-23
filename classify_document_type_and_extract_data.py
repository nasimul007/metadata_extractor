import os
import fitz  # pymupdf
import pytesseract
from pytesseract import Output
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

from extract_metadata import extract_metadata_from_digital_scanned_doc
from extraction_using_llm import extract_metadata_using_vision_llm


class DocumentTypeClassifier:
    def __init__(self, file_path):
        self.file_path = file_path

    def is_digital_pdf(self, pdf_path, min_chars=100):
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()

            return len(text.strip()) >= min_chars

        except Exception:
            return False
        

    def get_ocr_confidence(self, image_path):
        image = Image.open(image_path)
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--oem 3 --psm 6"
        )

        confidences = []

        for conf in data["conf"]:
            try:
                conf = float(conf)
                if conf > 0:
                    confidences.append(conf)
            except:
                pass

        if not confidences:
            return 0
        
        return sum(confidences) / len(confidences)


    def route_document(self, file_path):
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            if self.is_digital_pdf(file_path):
                return "digital_pdf"
            
            return "scanned_pdf"

        confidence = self.get_ocr_confidence(file_path)
        print("confidence", confidence)

        if confidence < 60:
            return "vision_llm"

        return "tesseract"


    def check_document_type(self):
        doc_type = self.route_document(self.file_path)
        print("data retrieval type: ", doc_type)

        if doc_type == "digital_pdf" or doc_type == "scanned_pdf" or doc_type == "tesseract":
            # extract metadata from digital/scanned documents using LLMs and OCR
            text = extract_metadata_from_digital_scanned_doc(self.file_path)
        elif doc_type == "vision_llm":
            # extract metadata from scanned + handwritten documents using VISION LLMs
            text = extract_metadata_using_vision_llm(self.file_path)

        return text
