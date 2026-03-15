import cv2
import numpy as np

def check_variance(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    return np.var(img)

if __name__ == "__main__":
    for side in ["left_1", "left_2", "right_1", "right_2"]:
        p = f"debug_{side}.jpg"
        v = check_variance(p)
        print(f"Variance in {p}: {v:.2f}")
