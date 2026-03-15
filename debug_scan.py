import cv2
import os

def scan_strips(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    
    # Save strips 1-5% and 5-10% from left
    cv2.imwrite("debug_left_1.jpg", img[:, int(w*0.01):int(w*0.04)])
    cv2.imwrite("debug_left_2.jpg", img[:, int(w*0.04):int(w*0.08)])
    
    # Save strips 90-95% and 95-99% from right
    cv2.imwrite("debug_right_1.jpg", img[:, int(w*0.90):int(w*0.94)])
    cv2.imwrite("debug_right_2.jpg", img[:, int(w*0.94):int(w*0.98)])
    
    print(f"Scanned strips for {image_path}")

if __name__ == "__main__":
    if os.path.exists("front_id.jpg"):
        scan_strips("front_id.jpg")
    elif os.path.exists("id_image.jpg"):
        scan_strips("id_image.jpg")
