import pdfplumber
import sys
import os

pdf_path = "3077.pdf"
if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found in {os.getcwd()}")
    sys.exit(1)

try:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"--- Page {i} ---")
            text = page.extract_text()
            if text:
                print(text)
            else:
                print("No text found on this page.")
except Exception as e:
    print(f"Error: {e}")
