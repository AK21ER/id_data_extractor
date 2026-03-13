import cv2
import pytesseract
import json
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def debug_full_ocr(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    img = cv2.imread(path)
    if img is None: 
        print(f"Failed to load image: {path}")
        return
    
    # Normalize
    target_width = 1400
    ratio = target_width / img.shape[1]
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (target_width, new_height))
    
    # Full OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    data = pytesseract.image_to_data(gray, lang="eng+amh", output_type=pytesseract.Output.DICT)
    
    with open("debug_ocr_dump.txt", "w", encoding="utf-8") as f:
        for i in range(len(data['text'])):
            t = data['text'][i].strip()
            if t:
                f.write(f"Txt: {t} | X: {data['left'][i]} | Y: {data['top'][i]} | W: {data['width'][i]}\n")

if __name__ == "__main__":
    debug_full_ocr("front_id.jpg")
    print("Done. Check debug_ocr_dump.txt")
