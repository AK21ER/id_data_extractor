import cv2
import numpy as np

def debug_sides(image_path):
    print(f"\n--- Debugging {image_path} ---")
    image = cv2.imread(image_path)
    if image is None: return
    
    h_img, w_img = image.shape[:2]
    is_portrait = h_img > w_img
    
    if is_portrait:
        areas = [
            {'x0': int(w_img * 0.82), 'x1': int(w_img * 0.96), 'side': 'right'},
            {'x0': int(w_img * 0.01), 'x1': int(w_img * 0.15), 'side': 'left'}
        ]
    else:
        areas = [
            {'x0': int(w_img * 0.01), 'x1': int(w_img * 0.15), 'side': 'left'},
            {'x0': int(w_img * 0.85), 'x1': int(w_img * 0.99), 'side': 'right'}
        ]
        
    for area in areas:
        test_crop = image[int(h_img*0.15):int(h_img*0.75), area['x0']:area['x1']]
        if test_crop.size > 0:
            var = np.var(cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY))
            print(f"Side: {area['side']} | X: {area['x0']}-{area['x1']} | Variance: {var:.2f}")
            cv2.imwrite(f"debug_side_{area['side']}_{image_path}", test_crop)

if __name__ == "__main__":
    debug_sides("front_id.jpg")
    debug_sides("id_image.jpg")
