import pdfplumber
import pypdfium2 as pdfium
from pdf_to_card import detect_is_v1, parse_fields_v1, parse_fields_flexible

pdf_path = "efayda_ Aby Yilma Astatke.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

with pdfplumber.open(pdf_path) as pdf:
    raw_text = pdf.pages[0].extract_text() or ""
    print("--- PDFPLUMBER TEXT ---")
    print(raw_text[:1000])
    
    # Try pypdfium2 text as fallback
    doc = pdfium.PdfDocument(pdf_bytes)
    page = doc[0]
    textpage = page.get_textpage()
    raw_text_p = textpage.get_text_range()
    print("--- PYPDFIUM2 TEXT ---")
    print(raw_text_p[:1000])
    
    # Use better text for field parsing
    combined_text = raw_text + "\n" + raw_text_p
    
    is_v1 = detect_is_v1(combined_text)
    print(f"Is V1: {is_v1}")
    
    if is_v1:
        fields = parse_fields_v1(combined_text)
    else:
        fields = parse_fields_flexible(combined_text)
    
    # Check QR
    from pdf_to_card import crop_qr_with_fallback, render_page, parse_qr_data
    full_page = render_page(pdf_bytes)
    qr_pil = crop_qr_with_fallback(full_page)
    qr_data = {}
    if qr_pil:
        import numpy as np
        from pyzbar.pyzbar import decode
        decoded = decode(qr_pil)
        if decoded:
            qr_data = parse_qr_data(decoded[0].data.decode())

    print("--- QR DATA ---")
    for k, v in qr_data.items():
        print(f"{k}: {v}")

    print("--- EXTRACTED FIELDS ---")
    for k, v in fields.items():
        if any(x in k for x in ["date", "expiry", "issue", "dob"]):
            print(f"{k}: {v}")
