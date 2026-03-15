import os

def find_target_coords(coords_file, targets):
    if not os.path.exists(coords_file): return
    print(f"--- FInding coordinates in {coords_file} ---")
    with open(coords_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for t in targets:
            if t.lower() in line.lower():
                print(line.strip())
                break

if __name__ == "__main__":
    targets = ['2026', 'Mar', '13', '18/07', '/04', 'Date', 'Issue']
    find_target_coords('ocr_coords_front_id.jpg.txt', targets)
    find_target_coords('ocr_coords_id_image.jpg.txt', targets)
