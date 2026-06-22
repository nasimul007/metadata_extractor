from paddleocr import PaddleOCR

ocr = PaddleOCR(
    # use_angle_cls=True,
    lang='en'
)

result = ocr.predict("/home/nasimul/Documents/Personal/AI worksop/TASK metadata extractor/metadata_extractor/docs/08252018154808_001.jpg")

print("OCR Result:", result)