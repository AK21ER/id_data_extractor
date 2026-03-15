import cv2
import numpy as np

def analyze_strip_content(image_path):
    img = cv2.imread(image_path)
    if img is None: return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate vertical profile (sum pixels across each row)
    profile = np.sum(thresh, axis=1)
    
    h = img.shape[0]
    print(f"--- Vertical Profile for {image_path} ---")
    
    # Find segments with content
    in_content = False
    start_y = 0
    segments = []
    
    threshold_val = 5 * 255 # At least 5 pixels wide
    
    for y in range(h):
        if profile[y] > threshold_val:
            if not in_content:
                start_y = y
                in_content = True
        else:
            if in_content:
                segments.append((start_y, y))
                in_content = False
    
    if in_content:
        segments.append((start_y, h))
        
    for start, end in segments:
        length = end - start
        if length > 10: # Ignore tiny noise
            print(f"Segment: {start/h:.4f} to {end/h:.4f} (height: {length})")
            # Save small crops of these segments for visual confirmation
            crop = img[start:end, :]
            cv2.imwrite(f"segment_{start}_{end}.jpg", crop)

if __name__ == "__main__":
    analyze_strip_content("debug_left_2.jpg")
