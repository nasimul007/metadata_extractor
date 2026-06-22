from extract_metadata import extract_metadata_from_digital_scanned_doc
from extraction_using_llm import extract_metadata_using_vision_llm

def main():
    # extract metadata from digital/scanned documents using LLMs and OCR
    extract_metadata_from_digital_scanned_doc()

    # extract metadata from scanned + handwritten documents using VISION LLMs
    extract_metadata_using_vision_llm()


if __name__ == "__main__":
    main()