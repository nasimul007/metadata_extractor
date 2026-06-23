import os
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from generate_meta_field_description_and_alias import get_meta_field_description_and_alias


load_dotenv()


llm_model = os.environ.get("LLM_MODEL")
llm = ChatOllama(model=llm_model)

system_message = """
You are an intelligent document understanding and metadata extraction engine.

You will receive:
1. A scanned document image.
2. Metadata field definitions.

The document may contain:
* Printed text
* Handwritten text
* Tables
* Forms
* Signatures
* Stamps
* OCR noise
* Mixed layouts

Your tasks are:
TASK 1: Extract Document Text
* Read all visible content from the document.
* Extract both printed and handwritten text.
* Preserve logical reading order.
* Correct obvious OCR mistakes when highly confident.
* Do not invent missing text.
* If text is unreadable, mark it as [UNCLEAR].

TASK 2: Extract Metadata
For each metadata field:
* Use title, displayname, aliases, and description.
* Use semantic understanding.
* Do not rely solely on exact label matching.
* Consider nearby context.
* Handwritten values may belong to printed labels.
* Infer values only when reasonably confident.

Confidence Levels:
* high
  Explicitly visible and clearly associated.

* medium
  Likely correct based on context.

* low
  Possible value but uncertain.

* none
  No reliable value found.

Output Rules:
* Return JSON only.
* Do not include explanations.
* Do not include markdown.
* Always return every metadata field.
* Use null when no value can be found.

Output Format:
{
    "extracted_text": "",
    "metadata": [
        {
            "title": "",
            "displayname": "",
            "value": null,
            "confidence": "",
            "evidence": ""
        }
    ]
}
"""

human_message_template = """
Metadata Definitions:
{{ metadata_definitions }}

Analyze the attached document image.
Tasks:
1. Extract all visible text from the image.
2. Extract metadata values using the metadata definitions.
3. Return valid JSON only using the specified schema.

Remember:
* The document may contain handwritten values.
* The document may contain OCR noise.
* Use aliases and descriptions to identify metadata fields.
* If a value is not confidently identifiable, return null.
"""

# meta fields will be fetched from DB
meta_fields = [
    {"title": "sender_name", "displayname": "Sender Name"},
    {"title": "sender_add", "displayname": "Sender Address"},
    {"title": "shipper_country", "displayname": "Shipper Country"},
    {"title": "receiver_name", "displayname": "Receiver Name"},
    {"title": "receiver_add", "displayname": "Receiver Address"},
    {"title": "recev_country", "displayname": "Receiver Country"},
    {"title": "inv_date", "displayname": "Invoice Date"},
    {"title": "inv_no", "displayname": "Invoice Number"},
    {"title": "prod_type", "displayname": "Product Type"},
    {"title": "bill_shipping_charge_to", "displayname": "Bill Shipping Charge To"},
    {"title": "bill_duty_tax_to", "displayname": "Bill Duty/Tax To"},
    {"title": "service_lvl", "displayname": "Service Level"}
]


def extract_metadata_using_vision_llm(file_path):
    meta_field_description = get_meta_field_description_and_alias(meta_fields)

    # file_path = "/home/nasimul/Documents/Personal/AI worksop/TASK metadata extractor/metadata_extractor/docs/08252018154809.jpg"

    with open(file_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    human_message = HumanMessage(content=[
        {
            "type": "text", 
            "text": human_message_template.replace("{{ metadata_definitions }}", str(meta_field_description))
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        }
    ])

    messages = [
        SystemMessage(content=system_message),
        human_message
    ]

    ai_msg = llm.invoke(messages)
    print(f"------------------Extracted Metadata using VISION LLM----------------")
    print(ai_msg.content.strip())



"""
---------------Generated Meta Fields aliases and descriptions--------------------------

[
  {
    "title": "sender_name",
    "displayname": "Sender Name",
    "aliases": ["Shipper", "From", "Consignor", "Sender", "Ship From", "Vendor Name", "Exporter"],
    "description": "The legal name of the person or company sending the shipment."
  },
  {
    "title": "sender_add",
    "displayname": "Sender Address",
    "aliases": ["Ship From Address", "Origin Address", "Sender Location", "Consignor Address", "Return Address", "Pickup Address"],
    "description": "The full physical address of the sender, including street, city, and postal code."
  },
  {
    "title": "shipper_country",
    "displayname": "Shipper Country",
    "aliases": ["Country of Origin", "Origin Country", "Shipper Nation", "Export Country", "From Country"],
    "description": "The name or ISO code of the country where the shipment originated."
  },
  {
    "title": "receiver_name",
    "displayname": "Receiver Name",
    "aliases": ["Consignee", "Ship To", "Recipient", "Receiver", "Buyer Name", "Customer Name", "Deliver To"],
    "description": "The legal name of the person or company receiving the shipment."
  },
  {
    "title": "receiver_add",
    "displayname": "Receiver Address",
    "aliases": ["Ship To Address", "Destination Address", "Delivery Address", "Consignee Address", "Shipping Destination"],
    "description": "The full physical address where the shipment is being delivered."
  },
  {
    "title": "recev_country",
    "displayname": "Receiver Country",
    "aliases": ["Destination Country", "Country of Destination", "Ship To Country", "Import Country", "To Country"],
    "description": "The name or ISO code of the country where the shipment is being delivered."
  },
  {
    "title": "inv_date",
    "displayname": "Invoice Date",
    "aliases": ["Billing Date", "Invoice Dt", "Date of Issue", "Doc Date", "Invoice Issuance Date"],
    "description": "The calendar date on which the invoice was officially issued."
  },
  {
    "title": "inv_no",
    "displayname": "Invoice Number",
    "aliases": ["Invoice #", "Inv No", "Bill Number", "Invoice Reference", "Inv #", "Document Number"],
    "description": "The unique alphanumeric identifier assigned to the specific invoice."
  },
  {
    "title": "prod_type",
    "displayname": "Product Type",
    "aliases": ["Item Category", "Commodity", "Goods Type", "Product Class", "Material Type", "SKU Type"],
    "description": "The classification or category of the goods being shipped."
  },
  {
    "title": "bill_shipping_charge_to",
    "displayname": "Bill Shipping Charge To",
    "aliases": ["Freight Charges To", "Shipping Bill To", "Freight Payer", "Carriage Paid By", "Freight Bill To"],
    "description": "The entity or party responsible for paying the transportation and shipping costs."
  },
  {
    "title": "bill_duty_tax_to",
    "displayname": "Bill Duty/Tax To",
    "aliases": ["Duty Payer", "Tax Bill To", "Customs Charges To", "VAT Payer", "Duty/Tax Responsibility"],
    "description": "The party responsible for the payment of customs duties and import taxes."
  },
  {
    "title": "service_lvl",
    "displayname": "Service Level",
    "aliases": ["Shipping Method", "Service Type", "Delivery Speed", "Transit Class", "Freight Level", "Shipping Tier"],
    "description": "The specific shipping speed or quality of service selected, such as Express, Standard, or Overnight."
  }
]
"""""

""""
--------------output for file = 08252018154808_001.jpg------------------

{
    "extracted_text": "SHIPPER'S UPS ACCOUNT NO. RIP 218\nFOR CUSTOMS PURPOSES (V.A.T. No, etc.)\nUPS WORLDWIDE SERVICES WAYBILL (non-negotiable)\nNAME OF SENDER SHAKIB\nTELEPHONE NO. (VERY IMPORTANT)\nCOMPANY NAME AND ADDRESS (Include Postal Code)\nGH S COMPOSITE KNITTING INDULT\nSHARDANGONT, KASHMIRUIR, J&K\nGAZIYPUR\nSERVICE LEVEL (Please mark \"X\". Select one level only. Refer to the appropriate service guide for the level selected.)\nEXPRESS PLUS 1+\nEXPRESS 1\nEXPRESS FREIGHT MIDDAY 1P\nEXPRESS FREIGHT X\nEXPRESS SAVER X\nEXPEDITED 2\nPOSTAL CODE COUNTRY/TERRITORY\nUPS Waybill No. 1S V031 4180 053 145\nTracking No.\nSPECIAL INSTRUCTIONS 58 x 39 x 32 (4) 5\nRECEIVER'S UPS ACCOUNT NO.\nRECEIVER'S IDENTIFICATION NO. FOR CUSTOMS PURPOSES (E.I.N., V.A.T., Import No., R.F.C. No., etc.)\nNAME OF CONTACT PERSON CHRISTINA\nTELEPHONE NO. (VERY IMPORTANT)\nCOMPANY NAME AND ADDRESS (Include Postal Code/ZIP Code)\nE BESTSELLER VERO MODA\nHELGE NIELSENS ALLE 11,\n8723, LOSNING,\nPOSTAL CODE COUNTRY/TERRITORY DANMARK\nSHIPPER RECEIPT (TRACKING NO.) 1/5\nW511 5504 209 ups\nSHIPPER RECEIPT (TRACKING NO.) 2/5\nW511 5504 218 ups\nSHIPPER RECEIPT (TRACKING NO.) 3/5\nW511 5504 227 ups\nSHIPPER RECEIPT (TRACKING NO.) 4/5\nW511 5504 236 ups\nSHIPMENT INFORMATION\nNO. OF PACKAGES/PALLETS TOTAL ACTUAL WEIGHT TOTAL CHARGEABLE WEIGHT ZONE\nIN SHIPMENT OF SHIPMENT OF SHIPMENT OF appliccable\n05 STAG 7410145 15\nNAME OF ALL PRODUCT/S MARK \"X\" IF LARGE PACKAGE MARK \"X\" IF HAZARDOUS HANDLING\n(DESCRIBE SIZE & WEIGHT) (OVER 68\" OR OVER 150 LBS) (CONTAINS DANGEROUS GOODS) CHARGE SHIPPER TO PAY DUTIES\nENV 10KG BOX 25KG XKN\nDESCRIPTION OF GOODS T-SHIRT = 321 PCS\nINVOICE NO. OF DOCUMENTS\n(if any) W511 5504 236\nPAYMENT OF CHARGES\nBILL SHIPPING CHARGES TO:\nSHIPPER (S) Account No. in Section 1\nRECEIVER (R) Account No. in Section 2\nTHIRD PARTY (T)\nCREDIT CARD CHECK THIRD PARTY COMPANY NAME:\nENTER THIRD PARTY'S UPS ACCOUNT NO. OR SHIPPER'S MAJOR CREDIT CARD NO.\nTHIRD PARTY COUNTRY / TERRITORY OR EXPIRATION DATE\nDECLARED VALUE OF SHIPMENT FOR CARRIAGE ONLY (Specify Currency)\nDECLARED VALUE OF SHIPMENT FOR CUSTOMER ONLY (Specify Currency)\nCURRENCY REFERENCE NO. 1 AMOUNT CURRENCY AMOUNT\nREFERENCE NO. 2\nBILL DUTIES AND TAXES TO (OPTIONAL SHIPMENTS ONLY):\nSHIPPER (S) Account No. in Section 1\nRECEIVER (R) Account No. in Section 2\nTHIRD PARTY (T)\nTHIRD PARTY COMPANY NAME:\nTHIRD PARTY ACCOUNT NO. THIRD PARTY COUNTRY / TERRITORY:\nSPECIAL OFFER CODE\nOTHER\nTOTAL CHARGES\nUnless a greater value for carriage is declared in the space provided on this waybill, the carrier's liability is limited to the amount provided by the Conditions of Contract and any amendments thereto, and the shipper's receipt of the shipment constitutes acceptance of these terms. The shipper agrees to the Terms and Conditions on the reverse of this waybill or the shipper's copy of this waybill.\nDATE OF SHIPMENT 25/8/18\nCOUNTRY / TERRITORY OF ORIGIN (MANUFACTURE) OF GOODS (BD)\nRECEIVED FOR UPS BY: DATE: TIME:\nSiva 25/8/18 19.50\n02578015803",
    "metadata": [
        {
            "title": "sender_name",
            "displayname": "Sender Name",
            "value": "SHAKIB GH S COMPOSITE KNITTING INDULT",
            "confidence": "high",
            "evidence": "NAME OF SENDER SHAKIB COMPANY NAME AND ADDRESS (Include Postal Code) GH S COMPOSITE KNITTING INDULT"
        },
        {
            "title": "sender_add",
            "displayname": "Sender Address",
            "value": "SHARDANGONT, KASHMIRUIR, J&K GAZIYPUR",
            "confidence": "high",
            "evidence": "SHARDANGONT, KASHMIRUIR, J&K GAZIYPUR"
        },
        {
            "title": "shipper_country",
            "displayname": "Shipper Country",
            "value": "BD",
            "confidence": "medium",
            "evidence": "COUNTRY / TERRITORY OF ORIGIN (MANUFACTURE) OF GOODS (BD)"
        },
        {
            "title": "receiver_name",
            "displayname": "Receiver Name",
            "value": "CHRISTINA E BESTSELLER VERO MODA",
            "confidence": "high",
            "evidence": "NAME OF CONTACT PERSON CHRISTINA COMPANY NAME AND ADDRESS (Include Postal Code/ZIP Code) E BESTSELLER VERO MODA"
        },
        {
            "title": "receiver_add",
            "displayname": "Receiver Address",
            "value": "HELGE NIELSENS ALLE 11, 8723, LOSNING",
            "confidence": "high",
            "evidence": "HELGE NIELSENS ALLE 11, 8723, LOSNING"
        },
        {
            "title": "recev_country",
            "displayname": "Receiver Country",
            "value": "DANMARK",
            "confidence": "high",
            "evidence": "COUNTRY/TERRITORY DANMARK"
        },
        {
            "title": "inv_date",
            "displayname": "Invoice Date",
            "value": "25/8/18",
            "confidence": "medium",
            "evidence": "DATE OF SHIPMENT 25/8/18"
        },
        {
            "title": "inv_no",
            "displayname": "Invoice Number",
            "value": "W511 5504 236",
            "confidence": "medium",
            "evidence": "INVOICE NO. OF DOCUMENTS (if any) W511 5504 236"
        },
        {
            "title": "prod_type",
            "displayname": "Product Type",
            "value": "T-SHIRT",
            "confidence": "high",
            "evidence": "DESCRIPTION OF GOODS T-SHIRT = 321 PCS"
        },
        {
            "title": "bill_shipping_charge_to",
            "displayname": "Bill Shipping Charge To",
            "value": "Shipper",
            "confidence": "medium",
            "evidence": "BILL SHIPPING CHARGES TO: SHIPPER (S) [marked with X]"
        },
        {
            "title": "bill_duty_tax_to",
            "displayname": "Bill Duty/Tax To",
            "value": "Shipper",
            "confidence": "medium",
            "evidence": "BILL DUTIES AND TAXES TO (OPTIONAL SHIPMENTS ONLY): SHIPPER (S) [marked with X]"
        },
        {
            "title": "service_lvl",
            "displayname": "Service Level",
            "value": "EXPRESS SAVER",
            "confidence": "high",
            "evidence": "EXPRESS SAVER X"
        }
    ]
}

"""

"""
--------------output for file = 08252018154809.jpg------------------

{
    "extracted_text": "SHIPPER'S UPS ACCT NO. SHIPPER'S IDENTIFICATION NO. FOR CUSTOMS PURPOSES (V.A.T. No., etc.)\nAQV608 C\nUPS WORLDWIDE SERVICES WAYBILL (non-negotiable) ups\nUPS Waybill Tracking No. V031 4038 912\nS NAME OF SENDER TELEPHONE NO. (VERY IMPORTANT)\nI QBAL 01863363996\nH COMPANY NAME AND ADDRESS (include Postal Code)\nINFINITY OUTFIT LTD\n1323 UTTAR KHAILKUR\nNATIONAL UNIVERSITY\nGAZIPUR 1704\nPOSTAL CODE 1230 COUNTRY/TERRITOR BANGLADESH\nSERVICE LEVEL (Please mark \"X\". Select one level only. Refer to the appropriate service guide on the back of the waybill.)\n4 EXPRESS PLUS 1+\nEXPRESS 1\nEXPRESS FREIGHT MIDDAY 1P\nEXPRESS SAVER X 1P\nEXPEDITED 2\n2 RECEIVER'S UPS ACCT NO. RECEIVER'S IDENTIFICATION NO. FOR CUSTOMS PURPOSES (E.I.N., V.A.T., Importer's No., R.F.C No., etc.)\nS NAME OF CONTACT PERSON TELEPHONE NO. (VERY IMPORTANT)\nKAREL BOSMA 31 645810570\nI COMPANY NAME AND ADDRESS (include Postal/ZIP Code)\nPOCKIES\nCYCLOON POST\nENERGIEWEG 12, 9743 AN\nGRONINGEN THE NETHERLAND\nPOSTAL CODE [UNCLEAR] COUNTRY/TERRITORY NETHERLAND\nS SHIPMENT INFORMATION\nNUMBER OF PACKAGES/PALLETS IN SHIPMENT TOTAL ACTUAL WEIGHT OF SHIPMENT TOTAL DIMENSIONAL WEIGHT OF SHIPMENT (if applicable)\nOL\n[UNCLEAR] BOX 10KG [UNCLEAR] BOX 25KG\nSPECIFY BOX OR\nS\nMAKE SURE ALL PACKAGES/PALLETS ARE SAFE AND SECURE\nMAKE SURE ALL PACKAGE LABELS APPLY TO EACH PACKAGE\nMAKE SURE ALL ADDRESS WARNING CHARGES APPLY TO SEND PACKAGE\nDESCRIPTION OF GOODS\nSHORT RIMS-2 AS\nZONE 5\nTRANSPORTATION\nDECLARED VALUE CHARGE\nOTHER\nOTHER\nTOTAL CHARGES\n3 PAYMENT OF CHARGES\nBILL SHIPPING CHARGES TO:\nSHIPPER (S) X RECEIVER (R) THIRD PARTY (T)\nAccount No. in Section 1 Account No. in Section 2\nCREDIT CARD CHECK THIRD PARTY COMPANY NAME:\nENTER THIRD PARTY'S UPS ACCOUNT NO. OR SHIPPER'S MAJOR CREDIT CARD NO.\nTHIRD PARTY COUNTRY/ TERRITORY OR EXPIRATION DATE\nDECLARED VALUE OF SHIPMENT FOR CARRIAGE ONLY (Specify Currency)\nCURRENCY REFERENCE NO. 1 AMOUNT\nCURRENCY REFERENCE NO. 2\nC\nL\nDECLARED VALUE OF SHIPMENT FOR CUSTOMS ONLY (Specify Currency)\nAMOUNT\nSPECIAL OFFER CODE\nN\nBILL DUTIES AND TAXES TO (DUTIABLE SHIPMENTS ONLY):\nSHIPPER (S) X RECEIVER (R) THIRD PARTY (T)\nAccount No. in Section 1 Account No. in Section 2\nTHIRD PARTY COMPANY NAME:\nTHIRD PARTY ACCOUNT NO. THIRD PARTY COUNTRY / TERRITORY:\n6 COUNTRY/TERRITORY OF ORIGIN (MANUFACTURE) OF GOODS\nBD\n7 DATE OF SHIPMENT 25/08/18 SHIPPER'S SIGNATURE\nRECEIVED UPS BY [UNCLEAR] DATE [UNCLEAR] TIME [UNCLEAR]\nARIF 24/8/18 20:07\nUPS COPY",
    "metadata": [
        {
            "title": "sender_name",
            "displayname": "Sender Name",
            "value": "INFINITY OUTFIT LTD",
            "confidence": "high",
            "evidence": "COMPANY NAME AND ADDRESS (include Postal Code) INFINITY OUTFIT LTD"
        },
        {
            "title": "sender_add",
            "displayname": "Sender Address",
            "value": "1323 UTTAR KHAILKUR NATIONAL UNIVERSITY GAZIPUR 1704",
            "confidence": "high",
            "evidence": "1323 UTTAR KHAILKUR NATIONAL UNIVERSITY GAZIPUR 1704"
        },
        {
            "title": "shipper_country",
            "displayname": "Shipper Country",
            "value": "BANGLADESH",
            "confidence": "high",
            "evidence": "COUNTRY/TERRITOR BANGLADESH"
        },
        {
            "title": "receiver_name",
            "displayname": "Receiver Name",
            "value": "POCKIES",
            "confidence": "high",
            "evidence": "COMPANY NAME AND ADDRESS (include Postal/ZIP Code) POCKIES"
        },
        {
            "title": "receiver_add",
            "displayname": "Receiver Address",
            "value": "CYCLOON POST ENERGIEWEG 12, 9743 AN GRONINGEN THE NETHERLAND",
            "confidence": "high",
            "evidence": "CYCLOON POST ENERGIEWEG 12, 9743 AN GRONINGEN THE NETHERLAND"
        },
        {
            "title": "recev_country",
            "displayname": "Receiver Country",
            "value": "NETHERLAND",
            "confidence": "high",
            "evidence": "COUNTRY/TERRITORY NETHERLAND"
        },
        {
            "title": "inv_date",
            "displayname": "Invoice Date",
            "value": null,
            "confidence": "none",
            "evidence": null
        },
        {
            "title": "inv_no",
            "displayname": "Invoice Number",
            "value": null,
            "confidence": "none",
            "evidence": null
        },
        {
            "title": "prod_type",
            "displayname": "Product Type",
            "value": "SHORT RIMS-2 AS",
            "confidence": "high",
            "evidence": "DESCRIPTION OF GOODS SHORT RIMS-2 AS"
        },
        {
            "title": "bill_shipping_charge_to",
            "displayname": "Bill Shipping Charge To",
            "value": "SHIPPER",
            "confidence": "high",
            "evidence": "BILL SHIPPING CHARGES TO: SHIPPER (S) X"
        },
        {
            "title": "bill_duty_tax_to",
            "displayname": "Bill Duty/Tax To",
            "value": "SHIPPER",
            "confidence": "high",
            "evidence": "BILL DUTIES AND TAXES TO (DUTIABLE SHIPMENTS ONLY): SHIPPER (S) X"
        },
        {
            "title": "service_lvl",
            "displayname": "Service Level",
            "value": "EXPRESS SAVER",
            "confidence": "high",
            "evidence": "EXPRESS SAVER X 1P"
        }
    ]
}

"""