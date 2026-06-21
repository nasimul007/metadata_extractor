import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from ocr_document import get_extracted_text
load_dotenv()


llm_model = os.environ.get("LLM_MODEL")
llm = ChatOllama(model=llm_model)


"""
meta_fields = [
    {"title": "inv_no", "displayname": "Invoice Number"},
    {"title": "inv_to", "displayname": "Invoiced To"},
    {"title": "inv_amount", "displayname": "Invoice Amount"},
    {"title": "inv_date", "displayname": "Invoice Date"}
]
"""


meta_field_generation_system_message = """
You are a document metadata expert.

Your task is to analyze metadata fields and enrich them with:

1. aliases
   - Generate 5-10 realistic alternative names that may appear in business documents.
   - Include abbreviations, common OCR variations, synonyms, and industry terms.
   - Do not generate unrelated aliases.

2. description
   - Generate a concise business description (1 sentence).
   - Explain what information this field typically contains.
   - The description should help another AI understand the meaning of the field during document extraction.

Rules:
- Preserve all existing keys.
- Add only:
  - aliases
  - description
- Return valid JSON only.
- Return the same list structure as the input.
- Do not wrap the response in markdown.
- Do not add explanations outside the JSON.
"""


meta_field_generation_human_message = """
Generate aliases and descriptions for the following metadata fields.

Input:
{{ meta_fields }}

Expected output format:
[
  {
    "title": "...",
    "displayname": "...",
    "aliases": ["...", "..."],
    "description": "..."
  }
]
"""


def get_meta_field_description_and_alias():
    # meta fields will be fetched from DB
    meta_fields = [
        {"title": "inv_no", "displayname": "Invoice Number"},
        {"title": "inv_to", "displayname": "Invoiced To"},
        {"title": "inv_amnt", "displayname": "Invoice Amount"},
        {"title": "inv_date", "displayname": "Invoice Date"}
    ]

    meta_field_generation_messages = [
        SystemMessage(content=meta_field_generation_system_message),
        HumanMessage(content=meta_field_generation_human_message.replace("{{ meta_fields }}", str(meta_fields)))
    ]

    ai_msg = llm.invoke(meta_field_generation_messages)
    answer = ai_msg.content.strip()
    print(f"\nAI Response: {answer}\n\n")

    return meta_fields


def main():
    get_extracted_text()
    get_meta_field_description_and_alias()


if __name__ == "__main__":
    main()