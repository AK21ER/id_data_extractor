import cv2
import os

def capture_full_strips(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    
    # Capture 5% to 10% width on both sides, full height
    left_strip = img[:, int(w*0.04):int(w*0.10)]
    right_strip = img[:, int(w*0.90):int(w*0.96)]
    
    cv2.imwrite("full_left_strip.jpg", left_strip)
    cv2.imwrite("full_right_strip.jpg", right_strip)
    print(f"Saved full strips for {image_path}")

if __name__ == "__main__":
    for f in ["front_id.jpg", "id_image.jpg", "front_1.jpg"]:
        if os.path.exists(f):
            capture_full_strips(f)
            break
