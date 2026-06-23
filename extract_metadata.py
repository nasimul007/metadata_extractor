import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from generate_meta_field_description_and_alias import get_meta_field_description_and_alias
from ocr_document import get_extracted_text


load_dotenv()


llm_model = os.environ.get("LLM_MODEL")
llm = ChatOllama(model=llm_model)


metadata_extraction_system_message = """
You are an intelligent document metadata extraction engine. 
Your task is to extract metadata values from OCR-extracted document text.

You will receive:
1. Metadata definitions

   * title
   * displayname
   * aliases
   * description

2. OCR extracted document text.

Extraction Rules:
* Use the field description as the primary source of understanding.
* Use aliases as alternative labels that may appear in the document.
* Do not rely only on exact keyword matching.
* Use semantic understanding and nearby context to identify values.
* OCR text may contain spelling mistakes, missing spaces, broken lines, or noisy characters.
* Infer the correct value when reasonably confident.

Confidence Rules:
* confidence = high
  * Explicitly found and clearly associated with the field.

* confidence = medium
  * Value is likely correct based on context but not explicitly labeled.

* confidence = low
  * Best guess only.

* confidence = none
  * No reliable value found.

Output Rules:
* Return JSON only.
* Do not explain.
* Do not add markdown.
* Preserve the original metadata title.
* If a value cannot be found, set value to null.
* Always return every metadata field.

Output Format:
{
    "metadata": 
    [
        {
            "title": "",
            "displayname": "",
            "value": "",
            "confidence": "",
            "evidence": ""
        }
    ]
}

"""


metadata_extraction_human_message = """
Metadata Definitions:
{{ metadata_definitions }}

OCR Extracted Text:
{{ document_text }}

Extract the metadata values and return JSON only.
"""


# meta fields will be fetched from DB
meta_fields = [
    {"title": "inv_no", "displayname": "Invoice Number"},
    {"title": "inv_to", "displayname": "Invoiced To"},
    {"title": "inv_amnt", "displayname": "Invoice Amount"},
    {"title": "inv_date", "displayname": "Invoice Date"}
]


def get_metadata_values(extracted_text, meta_field_description):
    meta_field_generation_messages = [
        SystemMessage(content=metadata_extraction_system_message),
        HumanMessage(content=metadata_extraction_human_message.replace(
            "{{ metadata_definitions }}", str(meta_field_description)
            ).replace("{{ document_text }}", extracted_text))
    ]

    ai_msg = llm.invoke(meta_field_generation_messages)
    extracted_metadata = ai_msg.content.strip()

    return extracted_metadata


def extract_metadata_from_digital_scanned_doc(file_path):
  
  # Step 1: Extract raw text
  extracted_text = get_extracted_text(file_path)
  # print(f"extracted_text: {extracted_text}")

  # Step 2: Generate metadata field descriptions and aliases
  meta_field_description = get_meta_field_description_and_alias(meta_fields)

  #Step 3: Extract metadata from the document using the generated descriptions and aliases
  extracted_metadata = get_metadata_values(extracted_text, meta_field_description)


  print(f"------------------Extracted Metadata using OCR + LLM----------------")
  print(extracted_metadata)