import cv2
import pytesseract
import numpy as np
from pyzbar.pyzbar import decode
from jinja2 import Template
import re
import os



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1. LOAD IMAGE
# -----------------------------
def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise Exception("Image not found: " + path)

    return image


# -----------------------------
# 2. NORMALIZE IMAGE SIZE
# -----------------------------
def normalize(image):

    target_width = 1400
    ratio = target_width / image.shape[1]

    new_height = int(image.shape[0] * ratio)

    resized = cv2.resize(image, (target_width, new_height))

    return resized


# -----------------------------
# 3. PREPROCESS FOR OCR
# -----------------------------
def preprocess(region):
    if region is None or region.size == 0:
        return None

    # Check if image is already grayscale
    if len(region.shape) == 2:
        gray = region
    else:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Resize to improve OCR on small text
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Adaptive thresholding or OTSU
    thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh


# -----------------------------
# 4. OCR FUNCTION
# -----------------------------
def read_text(region, psm=6, lang="eng+amh"):
    """
    Performs OCR with Amharic and English support.
    """
    processed = preprocess(region)
    if processed is None:
        return ""

    # Custom configuration for better OCR
    custom_config = f'--oem 3 --psm {psm}'
    text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)

    clean_text = text.strip()

    # Basic cleanup to remove common labels if they appear in the crop
    labels_to_remove = [
        "Full Name", "Date of Birth", "Sex", "Date of Expiry", "Phone Number", 
        "Nationality", "Address", "Ethiopian Digital ID", "National ID",
        "|", ":", "=", ">", "*", "APPR", "Nebomelty"
    ]
    for label in labels_to_remove:
        # Case insensitive removal
        clean_text = re.sub(re.escape(label), "", clean_text, flags=re.IGNORECASE)

    return clean_text.strip()


# -----------------------------
# 5. QR CODE PARSER
# -----------------------------
def parse_qr_data(qr_string):
    """
    Parses Ethiopian Digital ID QR data.
    Format: DLT:Name:V:4:G:Sex:A:FIN:D:DOB:SIGN:...
    """
    if not qr_string:
        return {}

    parts = qr_string.split(':')
    data = {}

    try:
        # Fayda QR format can vary; we check indices carefully
        if len(parts) > 2:
            data["name"] = parts[2]
            # Check if name itself contains Amharic (rare in QR but possible)
            if any(0x1200 <= ord(c) <= 0x137F for c in parts[2]):
                data["name_am"] = parts[2]
        
        # Some versions have Amharic fields later or in different parts
        for p in parts:
            if any(0x1200 <= ord(c) <= 0x137F for c in p) and len(p) > 2:
                if "name_am" not in data: data["name_am"] = p
        
        if len(parts) > 6:
            data["sex"] = "Female" if parts[6] == "F" else "Male" if parts[6] == "M" else parts[6]
        if len(parts) > 8:
            data["fin"] = parts[8]
        if len(parts) > 10:
            data["dob"] = parts[10]
    except Exception as e:
        print(f"Error parsing QR data: {e}")

    return data


# -----------------------------
# 5. QR CODE READER
# -----------------------------
# -----------------------------
# 5.5 TRANSLITERATE ENGLISH TO AMHARIC
# -----------------------------
# -----------------------------
# 5.5 TRANSLITERATE ENGLISH TO AMHARIC
# -----------------------------
def transliterate_to_amharic(text):
    """
    Phonetic transliteration from English to Amharic.
    Used for name fields when Amharic OCR fails.
    """
    if not text or text == "—": return "—"
    text = re.sub(r'[^a-zA-Z\s]', '', text).upper().strip()

    # Form mapping: Ge'ez(e), Ka'eb(u), Salis(i), Rabi(a), Hamis(ay/e), Sadis(silent), Sabi(o)
    forms = {
        'B': ['በ', 'ቡ', 'ቢ', 'ባ', 'ቤ', 'ብ', 'ቦ'],
        'C': ['ከ', 'ኩ', 'ኪ', 'ካ', 'ኬ', 'ክ', 'ኮ'],
        'D': ['ደ', 'ዱ', 'ዲ', 'ዳ', 'ዴ', 'ድ', 'ዶ'],
        'F': ['ፈ', 'ፉ', 'ፊ', 'ፋ', 'ፌ', 'ፍ', 'ፎ'],
        'G': ['ገ', 'ጉ', 'ጊ', 'ጋ', 'ጌ', 'ግ', 'ጎ'],
        'H': ['ሀ', 'ሁ', 'ሂ', 'ሃ', 'ሄ', 'ህ', 'ሆ'], 
        'J': ['ጀ', 'ጁ', 'ጂ', 'ጃ', 'ጄ', 'ጅ', 'ጆ'],
        'K': ['ከ', 'ኩ', 'ኪ', 'ካ', 'ኬ', 'ክ', 'ኮ'],
        'L': ['ለ', 'ሉ', 'ሊ', 'ላ', 'ሌ', 'ል', 'ሎ'],
        'M': ['መ', 'ሙ', 'ሚ', 'ማ', 'ሜ', 'ም', 'ሞ'],
        'N': ['ነ', 'ኑ', 'ኒ', 'ና', 'ኔ', 'ን', 'ኖ'],
        'P': ['ፐ', 'ፑ', 'ፒ', 'ፓ', 'ፔ', 'ፕ', 'ፖ'],
        'Q': ['ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ'],
        'R': ['ረ', 'ሩ', 'ሪ', 'ራ', 'ሬ', 'ር', 'ሮ'],
        'S': ['ሰ', 'ሱ', 'ሲ', 'ሳ', 'ሴ', 'ስ', 'ሶ'],
        'T': ['ተ', 'ቱ', 'ቲ', 'ታ', 'ቴ', 'ት', 'ቶ'],
        'V': ['ቨ', 'ቩ', 'ቪ', 'ቫ', 'ቬ', 'ቭ', 'ቮ'],
        'W': ['ወ', 'ዉ', 'ዊ', 'ዋ', 'ዌ', 'ው', 'ዎ'],
        'Y': ['የ', 'ዩ', 'ዪ', 'ያ', 'ዬ', 'ይ', 'ዮ'],
        'Z': ['ዘ', 'ዙ', 'ዚ', 'ዛ', 'ዜ', 'ዝ', 'ዞ'],
        'CH': ['ቸ', 'ቹ', 'ቺ', 'ቻ', 'ቼ', 'ች', 'ቾ'],
        'SH': ['ሸ', 'ሹ', 'ሺ', 'ሻ', 'ሼ', 'ሽ', 'ሾ'],
        'PH': ['ፈ', 'ፉ', 'ፊ', 'ፋ', 'ፌ', 'ፍ', 'ፎ'],
        'TH': ['ተ', 'ቱ', 'ቲ', 'ታ', 'ቴ', 'ት', 'ቶ'],
        'TS': ['ጸ', 'ጹ', 'ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ'],
        'NY': ['ኘ', 'ኙ', 'ኚ', 'ኛ', 'ኜ', 'ኝ', 'ኞ'],
        'HH': ['ሐ', 'ሑ', 'ሒ', 'ሓ', 'ሔ', 'ሕ', 'ሖ'], # Used for strict mapping
        'X':  ['ኃ', 'ኁ', 'ኂ', 'ኃ', 'ኄ', 'ኅ', 'ኆ']   # Used for strict mapping
    }
    
    # Map vowel to form index (0-6)
    v_map = {'E': 0, 'U': 1, 'I': 2, 'A': 3, 'O': 6}

    # High-Priority Exact Root Overrides
    text = text.replace("YOHANNES", "YOHHNIS")  # Forces ዮሐንስ instead of ዮሃነስ
    text = text.replace("HAILU", "XAYLU")       # Forces ኃይሉ instead of ሀይሉ
    text = text.replace("AKAKI", "AQAQI")       # Forces አቃቂ instead of አካኪ

    # Phonetic Pre-processing (Curious Logic)
    # 1. Words starting with KA- often use ቃ (Q) in Ethiopia (e.g., Kaleab, Kasahun, Kality)
    text = re.sub(r'\bKA', 'QA', text)
    # Words starting with WO map to ወ
    text = re.sub(r'\bWO', 'W', text)
        
    # 2. Collapse double consonants (Amharic gemination isn't written explicitly)
    text = re.sub(r'([BCDFGHJKLMNPQRSTVWYZ])\1+', r'\1', text)
    
    # 3. Handle specific clusters without collapsing them entirely
    text = text.replace("AI", "AY")  # Hailu -> Haylu
    text = text.replace("OU", "U")   # Ousman -> Usman
    text = text.replace("EI", "I")   # Hussein -> Husin (after SS->S)
    text = text.replace("IE", "I")
    
    # Map trailing 'Y' to 'I' for natural suffix sounds (Kality -> Kaliti)
    text = re.sub(r'Y\b', 'I', text)
    
    # EA in Kaleab is two syllables (ቃ-ለ-አብ), not one (ቃ-ሌ-ብ)
    text = text.replace("EE", "I").replace("OO", "U")

    final_words = []
    for word in text.split():
        if not word: continue
        if any(0x1200 <= ord(c) <= 0x137F for c in word):
            final_words.append(word); continue
        w_res = ""; wi = 0
        while wi < len(word):
            c_key = ""
            if wi + 1 < len(word) and word[wi:wi+2] in forms:
                c_key = word[wi:wi+2]; wi += 2
            elif word[wi] in forms:
                c_key = word[wi]; wi += 1
            
            if c_key:
                if wi < len(word) and word[wi] in v_map:
                    v_idx = v_map[word[wi]]
                    # Special Rule: 'E' followed by 'A' (like in Kaleab)
                    # The 'E' should be Ge'ez (0), not Hamis (4)
                    if word[wi] == 'E' and wi + 1 < len(word) and word[wi+1] == 'A':
                        v_idx = 0
                    w_res += forms[c_key][v_idx]
                    wi += 1
                else: 
                    # Default to Sadis (silent) if no vowel follows
                    w_res += forms[c_key][5]
            else:
                v = word[wi]
                # Map lone vowels to their distinct Amharic characters
                w_res += {'A': 'አ', 'U': 'ኡ', 'I': 'ኢ', 'O': 'ኦ', 'E': 'እ'}.get(v, '')
                wi += 1
        final_words.append(w_res)
    return " ".join(final_words)

def extract_dates_smart(text, is_dob=True):
    """
    Robustly extracts Ethiopic and Gregorian dates from text.
    Uses year ranges and script markers to distinguish between calendars.
    """
    if not text or len(text) < 5: return "—", "—"
    
    amh_months = ["መስከረም", "ጥቅምት", "ኅዳር", "ታኅሣሥ", "ጥር", "የካቲት", "መጋቢት", "ሚያዝያ", "ግንቦት", "ሰኔ", "ሐምሌ", "ነሐሴ", "ጳጉሜን"]
    eng_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                  "January", "February", "March", "April", "June", "July", "August", "September", "October", "November", "December"]
    
    # Improved regex to handle various delimiters and month names
    # Specifically tightened to catch DD/MM/YYYY and YYYY/MM/DD variations
    pattern = r'(\d{1,4}[-/.\s]*(?:[A-Za-z]{3,10}|[\u1200-\u137F]{2,10}|\d{1,4})[-/.\s]*\d{1,4}(?:[-/.\s]*\d{2,4})?)'
    
    candidates = re.findall(pattern, text)
    processed = []
    
    # Filter for realistic years
    # DOB: 1940-2025, Expiry: 2024-2045
    min_y = 1930 if is_dob else 2015
    max_y = 2025 if is_dob else 2045
    
    for cand in candidates:
        cand_clean = cand.strip().strip("|").strip(" :")
        if not cand_clean or len(cand_clean) < 6: continue
        
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', cand_clean)
        if not year_match: continue
        
        year = int(year_match.group(1))
        if not (min_y <= year <= max_y): continue
            
        is_greg_marker = any(m.lower() in cand_clean.lower() for m in eng_months)
        is_amh_marker = any(m in cand_clean for m in amh_months)
        
        processed.append({
            'original': cand_clean,
            'year': year,
            'is_greg': is_greg_marker,
            'is_amh': is_amh_marker
        })
    
    # --- GLOBAL FALLBACK FOR NUMERIC DATES ---
    # If we missed numeric pairs like 2026/07/01 | 2034/Mar/10
    numeric_pattern = r'(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b)'
    num_candidates = re.findall(numeric_pattern, text)
    
    for nc in num_candidates:
        if any(nc == p['original'] for p in processed): continue
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', nc)
        if year_match:
            year = int(year_match.group(1))
            if min_y <= year <= max_y:
                processed.append({
                    'original': nc,
                    'year': year,
                    'is_greg': False, # Default to False, will compare later
                    'is_amh': False
                })

    eth_date, greg_date = "—", "—"
    
    if len(processed) >= 1:
        # Separate by explicit markers first
        gregs = [p for p in processed if p['is_greg']]
        amhs = [p for p in processed if p['is_amh']]
        
        if gregs: greg_date = gregs[0]['original']
        if amhs: eth_date = amhs[0]['original']
        
        # Numeric comparison for the rest
        remaining = [p for p in processed if p['original'] not in [eth_date, greg_date]]
        remaining.sort(key=lambda x: x['year'])
        
        if eth_date == "—" and greg_date == "—" and len(remaining) >= 2:
            # If we have two numeric candidates, the smaller year is Ethiopic
            eth_date = remaining[0]['original']
            greg_date = remaining[-1]['original']
        elif len(remaining) >= 1:
            # Try to fill whichever is missing
            for cand in remaining:
                if eth_date == "—" and (greg_date == "—" or cand['year'] < (int(re.search(r'\d{4}', greg_date).group(0)) if re.search(r'\d{4}', greg_date) else 9999)):
                    eth_date = cand['original']
                elif greg_date == "—" and (eth_date == "—" or cand['year'] > (int(re.search(r'\d{4}', eth_date).group(0)) if re.search(r'\d{4}', eth_date) else 0)):
                    greg_date = cand['original']
                    
    # Final cleanup: if they are the same and not markers, one is likely wrong
    if eth_date == greg_date and eth_date != "—":
        if not any(m.lower() in eth_date.lower() for m in eng_months) and not any(m in eth_date for m in amh_months):
            # Numeric only and same, check year
            y = int(re.search(r'\d{4}', eth_date).group(0))
            if is_dob:
                if y <= 2018: greg_date = "—" # Ethiopic candidate
                else: eth_date = "—" # Gregorian candidate
            else:
                if y <= 2026: greg_date = "—"
                else: eth_date = "—"
            
    return eth_date, greg_date

def read_qr(image):

    qr_codes = decode(image)

    if len(qr_codes) > 0:
        return qr_codes[0].data.decode("utf-8")

    return None


# -----------------------------
# 6. CROP SAFE REGION
# -----------------------------
def crop(img, y1,y2,x1,x2):

    h,w = img.shape[:2]

    y1 = max(0,min(h,y1))
    y2 = max(0,min(h,y2))
    x1 = max(0,min(w,x1))
    x2 = max(0,min(w,x2))

    return img[y1:y2,x1:x2]


# -----------------------------
# 7. EXTRACT ASSETS (PORTRAIT & QR)
# -----------------------------
def extract_assets(qr_id_path):
    """
    Crops portrait and QR from qr_id.jpg.
    Returns (portrait_path, qr_path).
    """
    img = load_image(qr_id_path)
    
    # Adjusted slightly higher from 305 to 280
    portrait = crop(img, 280, 580, 150, 430)
    cv2.imwrite("portrait.png", portrait)
    
    # QR crop - keeping exactly as it was in the "perfect" version
    qr_img = crop(img, 725, 1050, 150, 426)
    cv2.imwrite("qr_code.png", qr_img)
    
    return "portrait.png", "qr_code.png"


# -----------------------------
# 8. EXTRACT FRONT DATA
# -----------------------------
def extract_front(image, qr_data=None):
    """
    Extracts data from the front of the ID card.
    Uses a dynamic layout-aware approach to handle varied positions and phone screenshots.
    """
    data = {
        "name_am": "—", "name_en": "—", 
        "dob_eth": "—", "dob_greg": "—", 
        "sex_am": "—", "sex_en": "—", 
        "id_number": "—", "fan": "—", 
        "barcode_data": "—"
    }

    # 1. PREPARE FULL IMAGE
    # -----------------------------
    # Use the full preprocessed image for initial layout analysis
    proc_full = preprocess(image)
    if proc_full is None: return data
    h_proc, w_proc = proc_full.shape[:2]
    
    # First pass: find anchors on the high-quality preprocessed image
    ocr_data = pytesseract.image_to_data(proc_full, lang="eng+amh", output_type=pytesseract.Output.DICT)

    # 1.5 BARCODE DETECTION (Front)
    # -----------------------------
    barcodes = decode(image)
    if barcodes:
        data["barcode_data"] = barcodes[0].data.decode('utf-8')
        # If barcode data looks like a FIN/FAN, use it as fallback
        if len(data["barcode_data"]) > 8:
            data["id_number"] = data["barcode_data"]
    
    # 2. FIND LAYOUT ANCHORS
    # -----------------------------
    id_top_y = 0
    header_found = False
    
    for i in range(len(ocr_data['text'])):
        txt = ocr_data['text'][i].strip().lower()
        if any(h in txt for h in ["ethiopian", "digital", "id", "national", "federal"]):
            id_top_y = max(id_top_y, ocr_data['top'][i])
            header_found = True
    
    if not header_found:
        id_top_y = 50 # Lower threshold since image is scaled 2x
    else:
        id_top_y += 50 # Buffer

    # 3. GROUP WORDS INTO LINES (Below Header)
    # -----------------------------
    raw_elements = []
    for i in range(len(ocr_data['text'])):
        t = ocr_data['text'][i].strip()
        y = ocr_data['top'][i]
        if t and y > id_top_y:
            raw_elements.append({
                'text': t, 
                'top': y, 
                'left': ocr_data['left'][i], 
                'w': ocr_data['width'][i], 
                'h': ocr_data['height'][i]
            })
    
    if not raw_elements: 
        return data
        
    raw_elements.sort(key=lambda x: x['top'])
    lines = []
    curr_l = [raw_elements[0]]
    for e in raw_elements[1:]:
        # Higher tolerance since image is 2x
        if e['top'] - curr_l[-1]['top'] < 35:
            curr_l.append(e)
        else:
            lines.append(sorted(curr_l, key=lambda x: x['left']))
            curr_l = [e]
    lines.append(sorted(curr_l, key=lambda x: x['left']))

    # 4. SEQUENCE-AWARE BUCKETING
    # -----------------------------
    label_map = {
        "name": ["name", "full", "ስም", "ሙሉ", "ሙለ", "ሙታ", "ሰም", "ጳም"],
        "dob": ["birth", "date", "የውልድ", "ትውልድ", "ቀን", "የልደት", "birt", "dob"],
        "sex": ["sex", "ፆታ", "ባታ", "ፃታ"],
        "expiry": ["expiry", "ያሚያበቃበት", "የሚቆይበት", "የሚያበቃበት", "expir", "expire", "doe"],
        "issue": ["issue", "የተሰጠበት"],
        "fcn": ["fan", "ካርድ", "ቁጥር", "fcn", "fina", "fin"]
    }
    
    buckets = {"name": [], "dob": [], "sex": [], "expiry": [], "issue": [], "fcn": []}
    active_key = "name"
    
    for line in lines:
        line_txt_low = " ".join([e['text'].lower() for e in line])
        found_new_field = False
        label_end_x = 0
        
        for key, tokens in label_map.items():
            if any(t in line_txt_low for t in tokens):
                # Find the rightmost edge of the label part
                for e in line:
                    if any(t in e['text'].lower() for t in tokens):
                        label_end_x = max(label_end_x, e['left'] + e['w'])
                active_key = key
                found_new_field = True
                break
        
        # Words to the right of label (or all words if no label) belong to active_key
        for e in line:
            if e['left'] > label_end_x:
                buckets[active_key].append(e)

    # 5. PROCESS BUCKETS (Dual-Pass for Name)
    # -----------------------------
    # proc_full is already prepared at 2x scale
    h_proc, w_proc = proc_full.shape[:2]

    # --- Name Processing ---
    name_words = buckets["name"]
    if name_words:
        name_words.sort(key=lambda x: x['top'])
        
        # --- NEW LOGIC: Explicit Amharic Bounding Box Crop ---
        # 1. Find the bottom of the "Full Name" label logic (or start of bucket)
        label_bottom_y = name_words[0]['top'] - 10 
        
        # 2. Find the top of the English name (first line that is purely English)
        eng_top_y = h_proc # Default to bottom of image
        for w in name_words:
            # If word is mostly English letters and has decent length
            if len(re.sub(r'[^a-zA-Z]', '', w['text'])) > 3:
                eng_top_y = w['top']
                break
                
        # 3. Define the bounding box for Amharic (between label and English)
        am_box_top = max(0, label_bottom_y)
        am_box_bottom = eng_top_y
        
        # Only proceed with this targeted crop if we have a reasonable vertical gap
        if am_box_bottom - am_box_top > 15:
            # Find horizontal bounds from all words in this vertical gap
            am_cand_words = [w for w in name_words if am_box_top <= w['top'] <= am_box_bottom]
            if am_cand_words:
                min_x = max(0, min(w['left'] for w in am_cand_words) - 15)
                max_x = min(w_proc, max(w['left'] + w['w'] for w in am_cand_words) + 15)
                
                if max_x - min_x > 50:
                    am_roi = proc_full[am_box_top:am_box_bottom, min_x:max_x]
                    
                    # Optional: Save crop for debugging
                    # cv2.imwrite("debug_amh_crop.png", am_roi)
                    
                    # Clean and OCR the specific crop
                    roi_z = cv2.resize(am_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    th1 = cv2.adaptiveThreshold(roi_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
                    raw_am_crop = pytesseract.image_to_string(th1, lang="amh", config='--psm 6').strip()
                    
                    # Clean the result
                    clean_crop = re.sub(r'[a-zA-Z0-9\(\)\{\}\[\]\.\,\|\!\?\-\_\/\\\»]', ' ', raw_am_crop)
                    clean_crop = re.sub(r'[\u1369-\u1371]', ' ', clean_crop) 
                    am_words = [w for w in clean_crop.replace("፣", " ").replace("።", " ").split() if sum(1 for c in w if 0x1200 <= ord(c) <= 0x137F) / max(1, len(w)) > 0.6]
                    
                    if am_words and len(" ".join(am_words)) > 3:
                        data["name_am"] = " ".join(am_words[:4])
                        # Filter out garbage
                        garbage_tokens = ["ርዐ", "ከህከ", "ፚ", "ቺጅ", "ጩወ", "ፍነ", "ርን", "ቺ", "ጩ", "ኚ", "ጯ"] 
                        if any(g in data["name_am"] for g in garbage_tokens):
                            data["name_am"] = "—"

        # --- Standard Line-by-Line Processing (Fallback for English and missing Amharic) ---
        n_lines = []
        cl = [name_words[0]]
        for w in name_words[1:]:
            if w['top'] - cl[-1]['top'] < 25: cl.append(w)
            else:
                n_lines.append(sorted(cl, key=lambda x: x['left']))
                cl = [w]
        n_lines.append(sorted(cl, key=lambda x: x['left']))
        
        am_parts, en_parts = [], []
        for nl in n_lines:
            l_top = min(w['top'] for w in nl)
            l_bot = max(w['top'] for w in nl) + 50
            l_left = min(w['left'] for w in nl) - 15
            l_right = max(w['left'] for w in nl) + 400
            
            y1, y2 = max(0, l_top - 60), min(h_proc, l_bot + 20)
            x1, x2 = max(0, l_left), min(w_proc, l_right)
            if (y2-y1) < 20 or (x2-x1) < 40: continue
            
            roi_g = proc_full[y1:y2, x1:x2]
            roi_z = cv2.resize(roi_g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Extract English
            en_n = pytesseract.image_to_string(roi_z, lang="eng", config='--psm 7').strip()
            en_words = [w for w in en_n.split() if int(len(re.findall(r'[a-z]', w.lower()))) > 1]
            en_words = [w for w in en_words if w.lower() not in ["name", "full", "id", "card", "national"]]
            if en_words: en_parts.append(" ".join(en_words))
            
            # Fallback Amharic extraction if our precise crop failed
            if data["name_am"] == "—":
                th1 = cv2.adaptiveThreshold(roi_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
                _, th2 = cv2.threshold(roi_z, 140, 255, cv2.THRESH_BINARY)
                pass1 = pytesseract.image_to_string(th1, lang="amh", config='--psm 6').strip()
                pass2 = pytesseract.image_to_string(th2, lang="amh", config='--psm 6').strip()
                am_n = pass1 + " " + pass2
                am_c = re.sub(r'[a-zA-Z0-9\(\)\{\}\[\]\.\,\|\!\?\-\_\/\\\»]', ' ', am_n)
                am_words = [w for w in am_c.replace("፣", " ").replace("።", " ").split() if len(w) > 1 and sum(1 for c in w if 0x1200 <= ord(c) <= 0x137F) / len(w) > 0.6]
                if am_words: am_parts.append(" ".join(am_words))

        # Compile Amharic fallback if needed
        if data["name_am"] == "—" and am_parts:
            seen = set()
            u_am = []
            for p in am_parts:
                for w in p.split():
                    if w not in seen and not any(noise in w for noise in ["ርዐ", "ቦዩህ", "ነሃ", "ልከ", "ከህ", "ከህከ"]):
                        u_am.append(w); seen.add(w)
            data["name_am"] = " ".join(u_am[:4])
            
            garbage_tokens = ["ርዐ", "ከህከ", "ፚ", "ቺጅ", "ጩወ", "ፍነ", "ርን", "ቺ", "ጩ", "ኚ", "ጯ"] 
            if any(g in data["name_am"] for g in garbage_tokens) or re.search(r'[፠-⠿]', data["name_am"]) or len(data["name_am"]) < 3:
                data["name_am"] = "—"
        
        # Compile English
        if en_parts:
            seen = set()
            u_en = []
            for p in en_parts:
                for w in p.split():
                    if w.lower() not in seen:
                        u_en.append(w); seen.add(w.lower())
            data["name_en"] = " ".join(u_en[:4])

    # --- Other Fields ---
    for key in ["dob", "sex", "expiry", "issue", "fcn"]:
        if buckets[key] or key in ["dob", "expiry", "issue"]: # Always attempt dates if key exists
            val_str = " ".join([e['text'] for e in sorted(buckets[key], key=lambda x: x['left'])])
            
            if key == "dob":
                data["dob_eth"], data["dob_greg"] = extract_dates_smart(val_str, is_dob=True)
            elif key == "sex":
                s_low = val_str.lower()
                if any(m in s_low for m in ["ወንድ", "male", "m"]): data["sex_am"], data["sex_en"] = "ወንድ", "Male"
                elif any(f in s_low for f in ["ሴት", "female", "f"]): data["sex_am"], data["sex_en"] = "ሴት", "Female"
            elif key == "expiry":
                data["expiry_eth"], data["expiry_greg"] = extract_dates_smart(val_str, is_dob=False)
            elif key == "issue":
                # We extract it to avoid it polluting other fields
                data["issue_eth"], data["issue_greg"] = extract_dates_smart(val_str, is_dob=False)
            elif key == "fcn":
                fcn_c = re.sub(r'[^0-9]', '', val_str)
                if 10 <= len(fcn_c) <= 16:
                    data["fan"] = fcn_c
                    if data["id_number"] == "—": data["id_number"] = fcn_c

    # --- GLOBAL FALLBACK SEARCH (Dates, FAN, Barcode) ---
    all_text = " ".join(ocr_data['text'])
    
    # 1. FAN Fallback
    if data["fan"] == "—":
        fan_match = re.search(r'\b(\d{10,16})\b', all_text)
        if fan_match: data["fan"] = fan_match.group(1)

    # 2. Date Fallback (Geometric Proximity Search)
    found_dates = []
    # Combined patterns
    date_patterns = [
        r'\b\d{1,4}[-/.\s]*(?:[A-Za-z]{3,10}|[\u1200-\u137F]{2,10}|\d{1,4})[-/.\s]*\d{1,4}(?:[-/.\s]*\d{2,4})?\b',
        r'(\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b)'
    ]
    
    # Extract with geometric positions
    all_cands = []
    for p in date_patterns:
        # We need the original OCR coordinates, so we'll look for strings that match patterns in groups
        # However, for global fallback, it's easier to just find all tokens that look like dates 
        # and use their original OCR indices.
        pass

    # Better approach: Iterate over OCR tokens and find those that contribute to a date
    char_to_y = {}
    curr_cursor = 0
    for i in range(len(ocr_data['text'])):
        t = ocr_data['text'][i]
        for _ in range(len(t)):
            char_to_y[curr_cursor] = ocr_data['top'][i]
            curr_cursor += 1
        # Space added by " ".join()
        char_to_y[curr_cursor] = ocr_data['top'][i]
        curr_cursor += 1

    all_found = []
    for p in date_patterns:
        for m in re.finditer(p, all_text):
            d_str = m.group(0).strip().strip("|").strip(" :")
            if len(d_str) < 6: continue
            ym = re.search(r'\b(19\d{2}|20\d{2})\b', d_str)
            if ym:
                y_coord = char_to_y.get(m.start(), 0)
                all_found.append({'val': d_str, 'y_year': int(ym.group(1)), 'y_coord': y_coord})

    # Identify label Y-coordinates
    label_y = {}
    for l_key, l_terms in label_map.items():
        for i in range(len(ocr_data['text'])):
            t = ocr_data['text'][i].strip().lower()
            if any(term in t for term in l_terms):
                label_y[l_key] = ocr_data['top'][i]
                break

    def get_best_pair(l_key, min_year, max_year):
        if l_key not in label_y: return "—", "—"
        target_y = label_y[l_key]
        # Broader range for vertical screenshots
        matches = [f for f in all_found if target_y - 50 <= f['y_coord'] <= target_y + 180]
        matches = [f for f in matches if min_year <= f['y_year'] <= max_year]
        
        if not matches: return "—", "—"
        
        # Sort by distance to label Y
        matches.sort(key=lambda x: abs(x['y_coord'] - target_y))
        
        # Take the top ones that are on similar Y-levels (horizontal pairs)
        closest_y = matches[0]['y_coord']
        pair = [f for f in matches if abs(f['y_coord'] - closest_y) < 40]
        
        # Sort indices by year to distinguish Ethiopic / Gregorian
        pair.sort(key=lambda x: x['y_year'])
        
        if len(pair) >= 2:
            return pair[0]['val'], pair[-1]['val']
        elif len(pair) == 1:
            if pair[0]['y_year'] <= (2018 if l_key == "dob" else 2027): return pair[0]['val'], "—"
            else: return "—", pair[0]['val']
        return "—", "—"

    # Re-assign using proximity
    if data["dob_eth"] == "—" or data["dob_greg"] == "—":
        d_eth, d_greg = get_best_pair("dob", 1930, 2025)
        if data["dob_eth"] == "—": data["dob_eth"] = d_eth
        if data["dob_greg"] == "—": data["dob_greg"] = d_greg

    if data["expiry_eth"] == "—" or data.get("expiry_greg", "—") == "—":
        e_eth, e_greg = get_best_pair("expiry", 2016, 2045)
        if data["expiry_eth"] == "—": data["expiry_eth"] = e_eth
        if data.get("expiry_greg", "—") == "—": data["expiry_greg"] = e_greg

    # 3. Specific fix for numeric OCR "2026/ Mar/ 12 2026"
    for k in ["dob_eth", "dob_greg", "expiry_eth", "expiry_greg"]:
        v = data.get(k, "—")
        if v != "—":
            # Remove redundant year repeats like "2026 ... 2026"
            parts = v.split()
            if len(parts) > 1 and parts[0] == parts[-1] and len(parts[0]) == 4:
                data[k] = " ".join(parts[:-1])

    # 6. QR OVERRIDES
    if qr_data:
        if qr_data.get("name"): data["name_en"] = qr_data["name"]
        if qr_data.get("name_am"): data["name_am"] = qr_data["name_am"]
        if qr_data.get("dob"): data["dob_greg"] = qr_data["dob"]
        if qr_data.get("sex"):
            data["sex_en"] = qr_data["sex"]
            data["sex_am"] = "ወንድ" if qr_data["sex"] == "Male" else "ሴት"
        if qr_data.get("fin"): data["id_number"] = qr_data["fin"]

    # 7. FINAL NAME TRANSLITERATION (Smart Fallback)
    if data["name_en"] != "—":
        trans_am = transliterate_to_amharic(data["name_en"])
        
        # If Amharic is missing, use transliteration
        if data["name_am"] == "—" or len(data["name_am"]) < 3:
            data["name_am"] = trans_am
        else:
            # If we have both, verify the OCR is likely a real name
            en_word_count = len(data["name_en"].split())
            am_word_count = len(data["name_am"].split())
            
            # If word counts differ, or OCR has suspicious repeat chars, use transliteration
            if en_word_count != am_word_count:
                data["name_am"] = trans_am
            # Double check for very common OCR failures again
            elif any(x in data["name_am"] for x in ["ክ", "ች", "ሽ", "ዝ", "ጅ"]):
                 # If Amharic OCR contains too many Sadis (6th) markers it might be noise
                 # but this is risky, so we just stick to word count mostly.
                 pass
            # Otherwise, we keep the OCR as requested ("exact as image")
            # unless the OCR has suspicious characters
    
    return data


# -----------------------------
# 9. EXTRACT BACK DATA
# -----------------------------
def extract_back(image):
    data = {"phone": "—", "reg_am": "—", "zone_am": "—", "woreda_am": "—", 
            "reg_en": "—", "zone_en": "—", "woreda_en": "—", "fin": "—"}

    full_back_proc = preprocess(image)
    if full_back_proc is None: return data
    full_back_ocr = pytesseract.image_to_data(full_back_proc, lang="eng+amh", output_type=pytesseract.Output.DICT)

    # 1. Phone Number
    phone_label_y = None
    for i in range(len(full_back_ocr['text'])):
        txt = full_back_ocr['text'][i].strip().lower()
        if "ስልክ" in txt or "phone" in txt:
            phone_label_y = full_back_ocr['top'][i]
            break

    found_phone = None
    for i in range(len(full_back_ocr['text'])):
        txt_clean = re.sub(r'[^0-9\+]', '', full_back_ocr['text'][i].strip())
        if 9 <= len(txt_clean) <= 13:
            if txt_clean.startswith("0") or txt_clean.startswith("251") or txt_clean.startswith("9"):
                found_phone = txt_clean
                break
    data["phone"] = found_phone or "—"

    # 1.5 FIN Extraction (Back)
    full_text = " ".join(full_back_ocr['text'])
    
    # Primary Search: Next to "FIN" label
    for i in range(len(full_back_ocr['text'])):
        txt = full_back_ocr['text'][i].strip().lower()
        if "fin" in txt or "የመታወቂያ" in txt or "ቁጥር" in txt:
            for j in range(i, min(i+8, len(full_back_ocr['text']))):
                num_cand = re.sub(r'[^0-9]', '', full_back_ocr['text'][j])
                if 12 <= len(num_cand) <= 16:
                    data["fin"] = num_cand
                    break
            if data["fin"] != "—": break

    # Fallback Search: Find any 12+ digit sequence anywhere in the text
    if data["fin"] == "—":
        fin_match = re.search(r'\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b', full_text)
        if fin_match:
            data["fin"] = re.sub(r'[^0-9]', '', fin_match.group(1))

    # 2. Address (Full-page label-anchored extraction)
    address_label_y = None
    for i in range(len(full_back_ocr['text'])):
        txt = full_back_ocr['text'][i].strip().lower()
        if "አድራሻ" in txt or "address" in txt or "አድራ" in txt:
            address_label_y = full_back_ocr['top'][i]
            break

    # If not found, fall back to address_crop
    if address_label_y is None:
        processed_addr = preprocess(address_crop)
        addr_ocr = pytesseract.image_to_data(processed_addr, lang="eng+amh", output_type=pytesseract.Output.DICT)
        elements = []
        for i in range(len(addr_ocr['text'])):
            t = addr_ocr['text'][i].strip()
            if t:
                elements.append({'text': t, 'top': addr_ocr['top'][i], 'left': addr_ocr['left'][i]})
    else:
        # Collect all tokens below the address label
        elements = []
        for i in range(len(full_back_ocr['text'])):
            t = full_back_ocr['text'][i].strip()
            y = full_back_ocr['top'][i]
            if t and y > address_label_y + 20:  # skip the label itself
                elements.append({'text': t, 'top': y, 'left': full_back_ocr['left'][i]})

    elements.sort(key=lambda x: x['top'])

    # Group into lines by Y coordinate tolerance
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

    # Categorize lines into Amharic and English with keyword awareness
    am_parts_map = {"reg": None, "zone": None, "woreda": None}
    en_parts_map = {"reg": None, "zone": None, "woreda": None}
    
    am_list, en_list = [], []
    skip_labels = {"Region", "Zone", "Woreda", "Address", "ክልል", "ዞን", "ወረዳ", "ቁጥር", "አድራሻ"}

    for l in lines:
        l_low = l.lower()
        # Clean leading/trailing symbols
        l_clean = re.sub(r'^[^a-zA-Z0-9\u1200-\u137F]+', '', l)
        l_clean = re.sub(r'[^a-zA-Z0-9\u1200-\u137F\s]+$', '', l_clean).strip()

        if len(re.sub(r'[^a-zA-Z0-9\u1200-\u137F]', '', l_clean)) < 2:
            continue

        # Strip keyword from the line if it exists to get the pure data
        data_part = l_clean
        for lbl in skip_labels:
            if lbl.lower() in l_low:
                # Use regex to strip the label word (with some boundary flexibility)
                data_part = re.sub(rf'\b{re.escape(lbl)}\b', '', data_part, flags=re.IGNORECASE).strip()
        
        # If after stripping labels we have nothing, skip
        if not data_part or len(data_part) < 2:
            continue

        # Categorize
        if any(0x1200 <= ord(c) <= 0x137F for c in data_part):
            am_val = re.sub(r'[a-zA-Z%@&\|]+', '', data_part).strip()
            if am_val:
                # Map based on keyword or sequence
                if "ክልል" in l or "region" in l_low: am_parts_map["reg"] = am_val
                elif "ዞን" in l or "zone" in l_low: am_parts_map["zone"] = am_val
                elif "ወረዳ" in l or "woreda" in l_low: am_parts_map["woreda"] = am_val
                else: am_list.append(am_val)
        elif any(c.isalpha() for c in data_part):
            en_val = data_part
            if "region" in l_low: en_parts_map["reg"] = en_val
            elif "zone" in l_low: en_parts_map["zone"] = en_val
            elif "woreda" in l_low: en_parts_map["woreda"] = en_val
            else: en_list.append(en_val)

    # Fill mapping from lists if still None
    if not am_parts_map["reg"] and am_list: am_parts_map["reg"] = am_list.pop(0)
    if not am_parts_map["zone"] and am_list: am_parts_map["zone"] = am_list.pop(0)
    if not am_parts_map["woreda"] and am_list: am_parts_map["woreda"] = am_list.pop(0)
    
    if not en_parts_map["reg"] and en_list: en_parts_map["reg"] = en_list.pop(0)
    if not en_parts_map["zone"] and en_list: en_parts_map["zone"] = en_list.pop(0)
    if not en_parts_map["woreda"] and en_list: en_parts_map["woreda"] = en_list.pop(0)

    data["reg_en"] = en_parts_map["reg"] or "—"
    data["zone_en"] = en_parts_map["zone"] or "—"
    data["woreda_en"] = en_parts_map["woreda"] or "—"

    # 3. Final Address Transliteration (Syncing & Fallback)
    # We prioritize OCR from the image if it's legitimate
    for key in ["reg", "zone", "woreda"]:
        en_val = data[f"{key}_en"]
        am_val = am_parts_map.get(key) or "—"
        
        # If we have English, we compute the transliteration as a sync candidate
        trans_am = transliterate_to_amharic(en_val) if en_val != "—" else "—"
        
        # Word count parity check
        en_word_count = len(en_val.split())
        am_word_count = len(am_val.split())
        
        # Priority logic: 
        # 1. Use image's Amharic if it's clean, doesn't look like noise, AND matches English word count
        if am_val != "—" and len(am_val) >= 2 and en_word_count == am_word_count:
            # Basic noise check for addresses (symbols, repeating chars)
            if not any(n in am_val for n in ["ርዐ", "ከህከ", "ፚ"]):
                data[f"{key}_am"] = am_val
            else:
                data[f"{key}_am"] = trans_am
        else:
            # 2. Otherwise (e.g. OCR truncated the Amharic address), use the full high-quality transliteration
            data[f"{key}_am"] = trans_am

    return data




# -----------------------------
# 9. DETECT WHICH SIDE
# -----------------------------
def detect_type(image):
    """
    Detects if the image is front, back, or side-by-side.
    """
    h, w = image.shape[:2]
    
    # 1. Check aspect ratio
    if w > h * 2.5:
        return "combined"
    
    # 2. Check content via OCR
    text = pytesseract.image_to_string(image)
    
    if "Full Name" in text or "Date of Birth" in text:
        return "front"
    
    if "Phone Number" in text or "Nationality" in text:
        return "back"

    return "unknown"


# -----------------------------
# 10. MERGE RESULTS
# -----------------------------
def merge(front=None,back=None,qr=None):

    result = {}

    if front:
        result.update(front)

    if back:
        result.update(back)

    if qr:
        result["qr_data"] = qr

    return result


# -----------------------------
# 10. EXPORT HTML
# -----------------------------
def export_html(data):
    """
    Renders the result.html template with the extracted data.
    """
    try:
        with open("result.html", "r", encoding="utf-8") as f:
            template_content = f.read()
    except:
        print("result.html not found, falling back to simple output.")
        return

    # Smart fallback for the 16-digit number
    actual_id = "—"
    for k in ["id_number", "fan", "barcode_data"]:
        val = data.get(k)
        if val and val != "—" and len(val) >= 10:
            actual_id = val
            break

    # Mock cards data for result.html
    # It expects: cards: [{ photo: '', photo_warm: '', qr: '', data: { ... } }]
    cards = [{
        "photo": "portrait.png",
        "photo_warm": "portrait.png",
        "qr": "qr_code.png",
        "data": {
            "name_en": data.get("name_en", ""),
            "name_am": data.get("name_am", "—"),
            "dob_greg": data.get("dob_greg", ""),
            "dob_eth": data.get("dob_eth", "—"),
            "sex_en": data.get("sex_en", ""),
            "sex_am": data.get("sex_am", ""),
            "expiry_greg": data.get("expiry_greg") or data.get("expiry") or data.get("expiry_eth") or "—",
            "expiry_eth": data.get("expiry_eth", "—"),
            "fcn": actual_id,
            "fin": data.get("fin", "—"),
            "barcode_data": actual_id,
            "phone": data.get("phone", ""),
            "nat_en": "Ethiopian",
            "nat_am": "ኢትዮጵያዊ",
            "reg_en": data.get("reg_en", ""),
            "reg_am": data.get("reg_am", ""),
            "zone_en": data.get("zone_en", ""),
            "zone_am": data.get("zone_am", ""),
            "woreda_en": data.get("woreda_en", ""),
            "woreda_am": data.get("woreda_am", ""),
            "fin_image": "", # Placeholders
            "expiry_image": "",
            "issue_greg_image": "",
            "issue_eth_image": ""
        }
    }]

    template = Template(template_content)
    
    # Mock url_for for Jinja2
    def mock_url_for(endpoint, **values):
        return values.get('filename', '')

    html = template.render(cards=cards, url_for=mock_url_for)

    with open("final_id_result.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Exported to final_id_result.html")


# -----------------------------
# 12. MAIN PROCESS FUNCTION
# -----------------------------
def process_screenshots(front_path, back_path, qr_path):
    """
    Specialized function for front_id, back_id, and qr_id screenshots.
    """
    # 1. Extract assets from qr_id
    portrait, qr_code = extract_assets(qr_path)
    
    # 2. Extract QR data
    qr_img = load_image(qr_path)
    qr_raw = read_qr(qr_img)
    qr_data = parse_qr_data(qr_raw)
    
    # 3. Extract Front
    front_img = load_image(front_path)
    front_data = extract_front(front_img, qr_data)
    
    # 4. Extract Back
    back_img = load_image(back_path)
    back_data = extract_back(back_img)
    
    # Merge results
    final_data = {**front_data, **back_data}
    final_data["qr_raw"] = qr_raw
    
    return final_data


# -----------------------------
# 13. RUN PROGRAM
# -----------------------------
if __name__ == "__main__":

    # Specific screenshots provided by user
    front = "front_id.jpg"
    back = "back_id.jpg"
    qr_id = "qr_id.jpg"

    if all(os.path.exists(f) for f in [front, back, qr_id]):
        data = process_screenshots(front, back, qr_id)
        # Avoid print encoding issues with Amharic
        try:
            print({k: v if isinstance(v, str) and not any(ord(c) > 127 for c in v) else "..." for k, v in data.items()})
        except:
            pass
        export_html(data)
    else:
        # Fallback to general processing if screenshots not found
        image_path = "id_image.jpg"
        if os.path.exists(image_path):
            data = process_image([image_path])
            print(data)
            export_html(data)
