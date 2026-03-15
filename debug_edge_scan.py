import cv2
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def debug_edge_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    
    # Check left 10% and right 10%
    sides = {
        "left": img[:, :int(w*0.1)],
        "right": img[:, int(w*0.9):]
    }
    
    for side_name, crop in sides.items():
        # Rotate to horizontal for OCR
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        data = pytesseract.image_to_data(rotated, lang="eng+amh", output_type=pytesseract.Output.DICT)
        
        print(f"--- Scan results for {side_name} edge of {image_path} ---")
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if txt:
                # Coordinate in the rotated image:
                # 'left' is distance from top in original image
                # 'top' is distance from left in original image (sort of)
                # Actually, in ROTATE_90_CLOCKWISE:
                # new_x = h - old_y
                # new_y = old_x
                # So old_y = h - new_x
                # old_x = new_y
                
                orig_y = h - (data['left'][i] + data['width'][i]//2)
                orig_x = data['top'][i] + data['height'][i]//2 # Relative to edge
                
                print(f"Text: '{txt}' at normalized Y: {orig_y/h:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    for f in ["front_id.jpg", "id_image.jpg", "front_1.jpg"]:
        if os.path.exists(f):
            debug_edge_ocr(f)
            break
