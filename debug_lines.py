import main
import pytesseract
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')
img = main.load_image('back_id.jpg')
full_back_proc = main.preprocess(img)
ocr = pytesseract.image_to_data(full_back_proc, lang='eng+amh', output_type=pytesseract.Output.DICT)

address_label_y = None
for i in range(len(ocr['text'])):
    txt = ocr['text'][i].strip().lower()
    if "አድራሻ" in txt or "address" in txt or "አድራ" in txt:
        address_label_y = ocr['top'][i]
        break

elements = []
if address_label_y:
    for i in range(len(ocr['text'])):
        t = ocr['text'][i].strip()
        y = ocr['top'][i]
        if t and y > address_label_y + 20: 
            elements.append({'text': t, 'top': y, 'left': ocr['left'][i]})

elements.sort(key=lambda x: x['top'])
lines = []
if elements:
    curr_line = [elements[0]]
    for e in elements[1:]:
        if e['top'] - curr_line[-1]['top'] < 20:  # 20px tolerance
            curr_line.append(e)
        else:
            lines.append(" ".join([x['text'] for x in sorted(curr_line, key=lambda x: x['left'])]))
            curr_line = [e]
    lines.append(" ".join([x['text'] for x in sorted(curr_line, key=lambda x: x['left'])]))

print(json.dumps(lines, ensure_ascii=False, indent=2))
