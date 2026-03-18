import pytesseract
from PIL import Image
import os

# Try common paths if default fails
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    print(f"Tesseract Version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    print(f"Tesseract not in PATH: {e}")
    # Try looking for it
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser('~') + r'\AppData\Local\Tesseract-OCR\tesseract.exe'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            print(f"Found tesseract at: {p}")
            pytesseract.pytesseract.tesseract_cmd = p
            print(f"Tesseract Version: {pytesseract.get_tesseract_version()}")
            break
    else:
        print("Tesseract executable NOT FOUND in common locations.")
