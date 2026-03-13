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
    
    # Sanitize: remove non-alpha (except spaces) for mapping
    orig_text = text
    text = re.sub(r'[^a-zA-Z\s]', '', text).upper().strip()
    
    # Advanced CV mapping
    # Consonants (in Sadis - 6th form)
    cons = {
        'B': 'ብ', 'C': 'ክ', 'D': 'ድ', 'F': 'ፍ', 'G': 'ግ', 'H': 'ህ', 'J': 'ጅ', 'K': 'ክ', 'L': 'ል', 
        'M': 'ም', 'N': 'ን', 'P': 'ፕ', 'Q': 'ቅ', 'R': 'ር', 'S': 'ስ', 'T': 'ት', 'V': 'ቭ', 'W': 'ው', 
        'X': 'ክስ', 'Y': 'ይ', 'Z': 'ዝ', 'CH': 'ች', 'SH': 'ሽ', 'PH': 'ፍ', 'TH': 'ት', 'TS': 'ጽ', 'NY': 'ኝ'
    }
    # Vowel transformations (Ge'ez-based)
    vowels = {
        'A': 0, 'U': 1, 'I': 2, 'O': 6, 'E': 5
    }
    
    # Syllable table (Sadis base + offset)
    def build_syllable(c_sadis, v_char):
        if not v_char or v_char not in vowels: return c_sadis
        base_ord = ord(c_sadis)
        offset = vowels[v_char]
        irregulars = {
            'ሀ': 0, 'ለ': 8, 'ሐ': 16, 'መ': 24, 'ሠ': 32, 'ረ': 40, 'ሰ': 48, 'ሸ': 56, 'ቀ': 64, 
            'በ': 72, 'ተ': 80, 'ቸ': 88, 'ኀ': 96, 'ነ': 104, 'ኘ': 112, 'አ': 120, 'ከ': 128, 
            'ኸ': 136, 'ወ': 144, 'ዐ': 152, 'ዘ': 160, 'ዠ': 168, 'የ': 176, 'ደ': 184, 'ጀ': 192, 
            'ገ': 200, 'ጠ': 208, 'ጨ': 216, 'ጰ': 224, 'ጸ': 232, 'ፀ': 240, 'ፈ': 248, 'ፐ': 256
        }
        geez = None
        for k, v in irregulars.items():
            if base_ord >= 0x1200 + v and base_ord < 0x1200 + v + 7:
                geez = chr(0x1200 + v); break
        if geez:
            if v_char == 'A': return chr(ord(geez) + 3) # Rabi
            if v_char == 'E': return geez # Default to Ge'ez
            if v_char == 'I': return chr(ord(geez) + 2)
            return chr(ord(geez) + offset)
        return c_sadis

    final_words = []
    for word in text.split():
        if not word: continue
        if any(0x1200 <= ord(c) <= 0x137F for c in word):
            final_words.append(word); continue
        w_res = ""; wi = 0
        while wi < len(word):
            c_part = ""
            if wi + 1 < len(word) and word[wi:wi+2] in cons:
                c_part = word[wi:wi+2]; wi += 2
            elif word[wi] in cons:
                c_part = word[wi]; wi += 1
            if c_part:
                if wi < len(word) and word[wi] in vowels:
                    w_res += build_syllable(cons[c_part], word[wi]); wi += 1
                else:
                    w_res += cons[c_part]
            else:
                v = word[wi]
                if v == 'A': w_res += 'አ'
                elif v == 'U': w_res += 'ኡ'
                elif v == 'I': w_res += 'ኢ'
                elif v == 'O': w_res += 'ኦ'
                elif v == 'E': w_res += 'እ'
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
            
            # Expand ROI upwards to catch Amharic if it's slightly higher than English
            y1, y2 = max(0, l_top - 60), min(h_proc, l_bot + 20)
            x1, x2 = max(0, l_left), min(w_proc, l_right)
            if (y2-y1) < 20 or (x2-x1) < 40: continue
            
            roi_g = proc_full[y1:y2, x1:x2]
            # Multiple thresholding styles for better noise rejection
            roi_z = cv2.resize(roi_g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            th1 = cv2.adaptiveThreshold(roi_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
            # Second threshold for thick/darker text
            _, th2 = cv2.threshold(roi_z, 140, 255, cv2.THRESH_BINARY)
            
            # Use multi-pass OCR on the same ROI
            pass1 = pytesseract.image_to_string(th1, lang="amh", config='--psm 6').strip()
            pass2 = pytesseract.image_to_string(th2, lang="amh", config='--psm 6').strip()
            am_n = pass1 + " " + pass2
            
            en_n = pytesseract.image_to_string(roi_z, lang="eng", config='--psm 7').strip()
            
            # Clean Amharic (Reject words with > 30% non-Amharic characters)
            def is_legit_amh(w):
                if len(w) < 2 or len(w) > 13: return False
                amh_chars = sum(1 for c in w if 0x1200 <= ord(c) <= 0x137F)
                if amh_chars / len(w) < 0.7: return False
                # Nonsense repetitions
                if sum(1 for i in range(len(w)-2) if w[i] == w[i+1] == w[i+2]) > 0: return False
                return True

            am_c = re.sub(r'[a-zA-Z0-9\(\)\{\}\[\]\.\,\|\!\?\-\_\/\\\»]', ' ', am_n)
            am_c = re.sub(r'[\u1369-\u1371]', ' ', am_c) 
            am_words = [w for w in am_c.replace("፣", " ").replace("።", " ").split() if is_legit_amh(w)]
            
            if am_words: am_parts.append(" ".join(am_words))
            
            # Clean English
            en_words = [w for w in en_n.split() if int(len(re.findall(r'[a-z]', w.lower()))) > 1]
            en_words = [w for w in en_words if w.lower() not in ["name", "full", "id", "card", "national"]]
            if en_words: en_parts.append(" ".join(en_words))

        if am_parts:
            seen = set()
            u_am = []
            for p in am_parts:
                for w in p.split():
                    if w not in seen:
                        # Stricter noise rejection
                        if not any(noise in w for noise in ["ርዐ", "ቦዩህ", "ነሃ", "ልከ", "ከህ", "ከህከ"]):
                            u_am.append(w); seen.add(w)
            data["name_am"] = " ".join(u_am[:4])
        
        if en_parts:
            seen = set()
            u_en = []
            for p in en_parts:
                for w in p.split():
                    if w.lower() not in seen:
                        u_en.append(w); seen.add(w.lower())
            data["name_en"] = " ".join(u_en[:4])

        # Noise rejection for Amharic
        # We classify as noise if it has too many symbols or known OCR artifacts
        if data["name_am"] != "—":
            noise_chars = ["፡", "፦", "፥", "ርዐ", "ከህ", "ከህከ", "ጩ", "ቺ", "ኚ", "ጯ", "ፚ"]
            if any(n in data["name_am"] for n in noise_chars):
                data["name_am"] = "—"
            # If name contains characters that are almost never in names
            elif re.search(r'[፠-⠿]', data["name_am"]): # Punctuation/Symbols
                data["name_am"] = "—"
            elif len(data["name_am"]) < 3:
                 data["name_am"] = "—"

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
            # If we have both, compare them. 
            # If OCR is completely different length than English words, it might be noise
            en_word_count = len(data["name_en"].split())
            am_word_count = len(data["name_am"].split())
            if abs(en_word_count - am_word_count) > 1:
                data["name_am"] = trans_am
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
    for i in range(len(full_back_ocr['text'])):
        txt = full_back_ocr['text'][i].strip().lower()
        if "fin" in txt or "የመታወቂያ" in txt or "ቁጥር" in txt:
            for j in range(i, min(i+8, len(full_back_ocr['text']))):
                num_cand = re.sub(r'[^0-9]', '', full_back_ocr['text'][j])
                if 10 <= len(num_cand) <= 16:
                    data["fin"] = num_cand
                    break
            if data["fin"] != "—": break

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

    # 3. Final Address Transliteration (Syncing)
    # If English exists, we transliterate to Amharic for consistency and to fix OCR noise
    if data["reg_en"] != "—":
        data["reg_am"] = transliterate_to_amharic(data["reg_en"])
    elif data["reg_am"] == "—":
        data["reg_am"] = "—"

    if data["zone_en"] != "—":
        data["zone_am"] = transliterate_to_amharic(data["zone_en"])
    elif data["zone_am"] == "—":
        data["zone_am"] = "—"

    if data["woreda_en"] != "—":
        data["woreda_am"] = transliterate_to_amharic(data["woreda_en"])
    elif data["woreda_am"] == "—":
        data["woreda_am"] = "—"

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
            "fcn": data.get("fan", "—"),
            "fin": data.get("fin", "—"),
            "barcode_data": data.get("barcode_data", "—"),
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
