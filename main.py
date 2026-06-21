from extract_metadata import get_metadata_values
from ocr_document import get_extracted_text
from generate_meta_field_description_and_alias import get_meta_field_description_and_alias

def main():
    # Step 1: Extract raw text
    extracted_text = get_extracted_text()

    # Step 2: Generate metadata field descriptions and aliases
    meta_field_description = get_meta_field_description_and_alias()

    #Step 3: Extract metadata from the document using the generated descriptions and aliases
    extracted_metadata = get_metadata_values(extracted_text, meta_field_description)

    print(f"Extracted Metadata: {extracted_metadata}")


if __name__ == "__main__":
    main()