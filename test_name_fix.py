import cv2
import pytesseract
import os
import json
from main import extract_front, load_image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_extraction(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    img = load_image(image_path)
    # Simulate qr_data being None as requested
    data = extract_front(img, qr_data=None)
    
    output = {
        "image": image_path,
        "name_en": data.get('name_en'),
        "name_am": data.get('name_am'),
        "id_number": data.get('id_number')
    }
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"Results written to test_results.json")

if __name__ == "__main__":
    test_extraction("front_id.jpg")
