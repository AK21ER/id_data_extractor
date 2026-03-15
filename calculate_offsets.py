def parse_coords(coords_file):
    import os
    if not os.path.exists(coords_file): return
    with open(coords_file, 'r') as f:
        lines = f.readlines()
    
    # Target tokens
    targets = ['2026', 'Mar', '13', 'Date', 'Issue']
    
    results = []
    for line in lines:
        if '|' not in line or "'" not in line: continue
        for t in targets:
            if t.lower() in line.lower():
                try:
                    parts = line.split('|')
                    text = parts[0].split("'")[1]
                    coords = parts[1].split(',')
                    x = int(coords[0].split(':')[1].strip())
                    y = int(coords[1].split(':')[1].strip())
                    w = int(coords[2].split(':')[1].strip())
                    h = int(coords[3].split(':')[1].strip())
                    results.append((text, x, y, w, h))
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
                break
    return results

if __name__ == "__main__":
    # front_id.jpg (1280x591)
    results_f = parse_coords('ocr_coords_front_id.jpg.txt')
    print("--- front_id.jpg (Vertical Screenshot) ---")
    for text, x, y, w, h in results_f:
        print(f"Text: {text} | X: {x/591:.4f}, Y: {y/1280:.4f}, W: {w/591:.4f}, H: {h/1280:.4f}")
        
    # id_image.jpg (325x1092)
    results_i = parse_coords('ocr_coords_id_image.jpg.txt')
    print("\n--- id_image.jpg (Landscape Crop) ---")
    for text, x, y, w, h in results_i:
        print(f"Text: {text} | X: {x/1092:.4f}, Y: {y/325:.4f}, W: {w/1092:.4f}, H: {h/325:.4f}")
