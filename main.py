from classify_document_type_and_extract_data import DocumentTypeClassifier


def main():
    file_path = "/home/nasimul/Documents/Personal/AI worksop/TASK metadata extractor/metadata_extractor/docs/08252018154809.jpg"
    doc_classifier = DocumentTypeClassifier(file_path)
    doc_classifier.check_document_type()


if __name__ == "__main__":
    main()