import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def visualize_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    
    # Run OCR with data
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    with open(f"ocr_coords_{os.path.basename(image_path)}.txt", "w") as f:
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                f.write(f"Text: '{text}' | X: {x}, Y: {y}, W: {bw}, H: {bh}\n")
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
    cv2.imwrite(f"ocr_vis_{os.path.basename(image_path)}", img)
    print(f"Saved OCR visualization for {image_path}")

if __name__ == "__main__":
    for f in ["front_id.jpg", "id_image.jpg"]:
        if os.path.exists(f):
            visualize_ocr(f)
