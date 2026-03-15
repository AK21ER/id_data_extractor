import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def scan_full_strip(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: Could not load {image_path}")
        return
    h, w = img.shape[:2]
    
    # Take the left 10% as a vertical strip
    strip = img[:, :int(w*0.10)]
    
    # Try both rotations
    for rot_name, rot_op in [("CW", cv2.ROTATE_90_CLOCKWISE), ("CCW", cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        rotated = cv2.rotate(strip, rot_op)
        data = pytesseract.image_to_data(rotated, output_type=pytesseract.Output.DICT)
        
        print(f"--- OCR Analysis ({rot_name}) for {image_path} ---")
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                y_orig = data['left'][i] if rot_name == "CW" else h - data['left'][i]
                print(f"Text: '{text}' at Y-pos: {y_orig} ({y_orig/h:.4f} height)")

if __name__ == "__main__":
    for f in ["front_id.jpg", "id_image.jpg"]:
        if os.path.exists(f):
            scan_full_strip(f)
