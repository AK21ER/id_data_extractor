import cv2
import numpy as np
import os

def debug_strips(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image {image_path} size: {w}x{h}")
    
    # Left strip (0-10%)
    left_strip = img[:, :int(w*0.1)]
    cv2.imwrite("debug_left_strip.jpg", left_strip)
    
    # Right strip (90-100%)
    right_strip = img[:, int(w*0.9):]
    cv2.imwrite("debug_right_strip.jpg", right_strip)
    
    print(f"Saved debug strips for {image_path}")

if __name__ == "__main__":
    # Check what files are available
    files = ["front_id.jpg", "kf.jpg", "id_image.jpg", "front_1.jpg"]
    found = False
    for f in files:
        if os.path.exists(f):
            debug_strips(f)
            found = True
            break
    if not found:
        print("No sample images found to debug.")
