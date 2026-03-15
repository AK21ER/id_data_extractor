import cv2
import os
import base64
from main import extract_front

def test_extraction(image_path):
    print(f"\n--- Testing {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return
    print(f"Dimensions: {img.shape[1]}x{img.shape[0]}")
    
    try:
        data = extract_front(img)
        
        # Check if encoded images exist
        for k in ["issue_greg_encoded", "issue_eth_encoded"]:
            if k in data and data[k] is not None:
                # Save for visual check
                filename = f"out_{k}_{os.path.basename(image_path)}.png"
                with open(filename, "wb") as f:
                    f.write(data[k])
                print(f"Saved {filename} (Size: {len(data[k])} bytes)")
            else:
                print(f"MISSING: {k}")
        
        # Check OCR text
        print(f"OCR Greg: {data.get('issue_greg', '—')}")
        print(f"OCR Eth: {data.get('issue_eth', '—')}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if os.path.exists("front_id.jpg"): test_extraction("front_id.jpg")
    if os.path.exists("id_image.jpg"): test_extraction("id_image.jpg")
