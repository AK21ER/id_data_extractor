import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def check_strip(path):
    img = cv2.imread(path)
    if img is None: return
    # Rotate 90 deg clockwise to make vertical text horizontal for OCR
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    text = pytesseract.image_to_string(rotated, lang="eng+amh", config='--psm 6')
    print(f"--- Text in {path} (rotated) ---")
    print(text)
    print("-------------------------------")

if __name__ == "__main__":
    check_strip("debug_left_strip.jpg")
    check_strip("debug_right_strip.jpg")
