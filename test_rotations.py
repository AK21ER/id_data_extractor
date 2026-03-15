import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_rotations(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    strip = img[:, :int(w*0.10)]
    
    # Try 90 CW
    cw = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"strip_cw_{os.path.basename(image_path)}", cw)
    
    # Try 90 CCW
    ccw = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"strip_ccw_{os.path.basename(image_path)}", ccw)
    
    print(f"Saved rotations for {image_path}")

if __name__ == "__main__":
    for f in ["front_id.jpg", "id_image.jpg"]:
        if os.path.exists(f):
            test_rotations(f)
