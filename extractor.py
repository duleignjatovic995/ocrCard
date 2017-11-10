"""
This file is combining all OCR and NLP functionalities and extracting 
values from cards such as insurer, member id and member name.
"""
import ocr
import json
import nlp


def extract_image(image_path):
    data = {}
    data['insurer'] = ocr.get_insurer(image_path)
    text = ocr.get_text(image_path)
    data['member name'] = nlp.get_person(text)
    data['member id'] = nlp.get_id(text)

    return json.dumps(data)
