import cv2
import pytesseract
import numpy as np
from pyzbar.pyzbar import decode
from jinja2 import Template
import re
import os
from PIL import Image, ImageEnhance



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1. LOAD IMAGE
# -----------------------------
def load_image(path_or_img):
    if isinstance(path_or_img, str):
        image = cv2.imread(path_or_img)
    else:
        image = path_or_img

    if image is None:
        raise Exception("Image not found or invalid: " + str(path_or_img))

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

from datetime import datetime

def calculate_issue_from_expiry(date_str, minus_years=5):
    """
    Takes a date string (YYYY-MM-DD or DD/MM/YYYY), 
    subtracts years, and returns the new string.
    """
    # Try common formats used in your OCR output
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Subtract years by replacing the year attribute
            issue_dt = dt.replace(year=dt.year - minus_years)
            return issue_dt.strftime(fmt)
        except (ValueError, TypeError):
            continue
    return "—"

def transliterate_to_amharic(en_name):
    """Basic fallback transliteration for names if Amharic crop fails."""
    if not en_name or en_name == "—": return "—"
    # Mapping for common sounds/names (very simplified)
    mapping = {
        'a': 'አ', 'b': 'በ', 'c': 'ከ', 'd': 'ደ', 'e': 'ኤ', 'f': 'ፈ', 'g': 'ገ', 'h': 'ሀ',
        'i': 'ኢ', 'j': 'ጀ', 'k': 'ከ', 'l': 'ለ', 'm': 'መ', 'n': 'ነ', 'o': 'ኦ', 'p': 'ፐ',
        'q': 'ቀ', 'r': 'ረ', 's': 'ሰ', 't': 'ተ', 'u': 'ኡ', 'v': 'ቨ', 'w': 'ወ', 'x': 'ክስ',
        'y': 'የ', 'z': 'ዘ', 'sh': 'ሸ', 'ch': 'ቸ', 'gn': 'ኝ', 'nh': 'ኝ'
    }
    am = en_name.lower()
    for en, amh in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        am = am.replace(en, amh)
    return am
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
    
    # Use OTSU thresholding (generally more robust for white-background IDs)
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
    
    return clean_text

def looks_like_name(text):
    """
    Validates if a string looks like a full name with high precision.
    Rejects noise, labels, and poorly formatted OCR artifacts.
    """
    if not text or len(text) > 70 or len(text) < 5:
        return False
    
    # Remove obvious noise and check digits
    clean = re.sub(r'[:|/\\_\[\]\(\)\-\.]', ' ', text).strip()
    
    # Rejection list for common OCR noise/labels (EXACT WORD or MAJOR PART)
    blockwords = {"fullname", "full name", "date", "birth", "sex", "expiry", "issue", "male", "female", "card", "id", "national", "fayda", "digital", "copy", "registration", "federal", "republic", "pan", "pfo", "pfa", "aps", "apo", "pda", "od", "at", "da", "ap", "official", "nationality", "selte", "declared", "birthdate"}
    low_clean = clean.lower()
    words = low_clean.split()
    
    # If any word in the candidate is a known label, reject it
    if any(w.strip(".:") in blockwords for w in words):
        return False

    # Block patterns like "Feb", "Jan", "OCR", "Nationality"
    noise_patterns = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "ocr"]
    if any(p in words for p in noise_patterns):
        return False

    # Strict digit rejection: no digits allowed in a clean name
    if any(c.isdigit() for c in clean):
        return False
    
    # Check word count - must be at least 2 substantiated words
    words_orig = clean.split()
    long_words = [w for w in words_orig if len(w) > 2]
    if len(long_words) < 2:
        return False
    
    # Reject if too many single letters (OCR spam)
    single_letters = [w for w in words_orig if len(w) == 1]
    if len(single_letters) > 1:
        return False
        
    return True

def looks_like_amharic_name(text):
    """
    Validates if a string looks like an Amharic full name.
    """
    if not text or len(text) < 3:
        return False
    
    # Keep only Amharic characters for validation
    am_chars = re.sub(r'[^\u1200-\u137F\s]', ' ', text).strip()
    words = am_chars.split()
    
    # Reject card labels and common noise
    # Ethiopic punctuation to strip: ፡(1361), ።(1362), ፣(1363), ፤(1364), ፥(1365), ፦(1366), ፧(1367), ፨(1368)
    blockwords = {
        "ሙሉ", "ስም", "ፆታ", "ባታ", "ቀን", "ትውልድ", "የውልድ", "ካርድ", 
        "ቁጥር", "ዜግነት", "ኢትዮጵያዊ", "መታወቂያ", "ዲጂታል", "ኢትዮጵያ", "ሰቅ", "ኢቶዮጵያ", "ብሔራዊ"
    }
    
    for w in words:
        # Clean word of Amharic punctuation for matching
        w_clean = re.sub(r'[\u1361-\u1368]', '', w)
        if w_clean in blockwords:
            return False
        
    # Standard Amharic names have 2-4 words
    if not (2 <= len(words) <= 4):
        return False
        
    # Substantiated words (at least 2 chars long)
    long_words = [w for w in words if len(w) >= 2]
    if len(long_words) < 2:
        return False
        
    return True

def score_amharic_name(text):
    """
    Scores Amharic name candidates.
    """
    score = 0
    words = text.strip().split()
    
    # Prefer 3 words
    if len(words) == 3:
        score += 10
    elif len(words) == 2:
        score += 5
        
    # Prefer length of characters (substantially long names)
    score += min(len(text), 20)
    
    return score

def score_name_candidate(text):
    """
    Scoring system that strongly prefers Title Case and 3-word names.
    """
    score = 0
    words = text.strip().split()
    
    # Length bonus: preferred 2-4 words
    if 2 <= len(words) <= 4:
        score += 5
    elif 5 <= len(words) <= 6:
        score += 2
        
    # Title Case bonus (crucial for English names)
    if all(w[0].isupper() for w in words if len(w) > 2):
        score += 10
    elif text.istitle():
        score += 8
        
    # Penalty for very short words
    shorts = [w for w in words if len(w) <= 2]
    score -= len(shorts) * 2
        
    # Bonus for common name-like word counts
    if len(words) == 3:
        score += 3
        
    return score


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


import cv2
import numpy as np
from pyzbar.pyzbar import decode

# -----------------------------
# 7. EXTRACT ASSETS (PORTRAIT & QR)
# -----------------------------

def extract_assets(qr_id_path_or_img):
    img = cv2.imread(qr_id_path_or_img) if isinstance(qr_id_path_or_img, str) else qr_id_path_or_img
    if img is None:
        return None, None

    height, width = img.shape[:2]

    # ---------------------------
    # QR DETECTION
    # ---------------------------
    decoded_qrs = decode(img)
    qr_top = height
    qr_img_crop = None

    if decoded_qrs:
        qr = decoded_qrs[0]
        x, y, w, h = qr.rect
        margin = 15
        y1, y2 = max(0, y - margin), min(height, y + h + margin)
        x1, x2 = max(0, x - margin), min(width, x + w + margin)
        qr_img_crop = img[y1:y2, x1:x2]
        qr_top = y1
    else:
        # Better dynamic fallback for QR if not found
        qr_img_crop = img[int(height*0.6):int(height*0.95), int(width*0.2):int(width*0.8)]
        qr_top = int(height*0.6)

    # ---------------------------
    # FACE DETECTION (Replaces Rectangle Finder)
    # ---------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    portrait_raw = None

    if len(faces) > 0:
        # Pick the face that is highest up (above the QR code) and largest
        valid_faces = [f for f in faces if f[1] < qr_top]
        if valid_faces:
            # Sort by size (area)
            valid_faces = sorted(valid_faces, key=lambda f: f[2]*f[3], reverse=True)
            xf, yf, wf, hf = valid_faces[0]

            # Add padding to get the whole head, not just the face "mask"
            # Balanced ID-style crop
        pad_w = int(wf * 0.40)

        top_pad = int(hf * 0.35)       # keep some space above head
        bottom_pad = int(hf * 1.0)    # include shoulders

        y1 = max(0, yf - top_pad)
        y2 = min(qr_top, yf + hf + bottom_pad)

        x1 = max(0, xf - pad_w)
        x2 = min(width, xf + wf + pad_w)

        portrait_raw = img[y1:y2, x1:x2]

    # Fallback if no face detected
    if portrait_raw is None:
        # Default to the upper-center area where portraits usually sit
        portrait_raw = img[int(height*0.1):int(qr_top*0.9), int(width*0.1):int(width*0.5)]

    # ---------------------------
    # BACKGROUND REMOVE (Clean edges)
    # ---------------------------
    portrait_bgra = cv2.cvtColor(portrait_raw, cv2.COLOR_BGR2BGRA)
    
    # Increased sensitivity for white background removal
    lower_white = np.array([220, 220, 220])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(portrait_raw, lower_white, upper_white)
    portrait_bgra[mask == 255] = (0, 0, 0, 0)

    # ---------------------------
    # ENCODE
    # ---------------------------
    _, p_enc = cv2.imencode('.png', portrait_bgra)
    _, q_enc = cv2.imencode('.png', qr_img_crop)

    return p_enc, q_enc

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

    # header keywords (EXACT WORD MATCHING)
    header_keywords = {"ethiopian", "digital", "id", "card", "national", "federal", "fayda", "republic", "fcn", "fin", "copy", "official", "nationalid", "registration", "name", "full", "fullname", "ሙሉ", "ስም", "ap", "da", "apo", "apf", "aps", "pan", "pfa", "pto", "pfo", "pda", "od", "at"}

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
        if any(h in txt for h in ["ethiopian", "digital", "id", "national", "federal", "fayda", "republic", "federal"]):
            # More aggressive header skip for screenshots
            id_top_y = max(id_top_y, ocr_data['top'][i] + ocr_data['height'][i] + 5)
            header_found = True
    
    if not header_found:
        id_top_y = 10 
    else:
        # Don't skip more than 12% of the height initially to avoid hitting labels
        id_top_y = min(id_top_y, max(100, int(h_proc * 0.12)))

    # 3. GROUP WORDS INTO LINES (Below Header)
    # -----------------------------
    raw_elements = []
    for i in range(len(ocr_data['text'])):
        t = ocr_data['text'][i].strip()
        y = ocr_data['top'][i]
        x = ocr_data['left'][i]
        
        # 75% Width Filter: Ignore right-side "sideways" labels (like Date of Issue) 
        # that interfere with the primary field sequence.
        if x > w_proc * 0.75:
            continue

        if t and y > id_top_y:
            raw_elements.append({
                'text': t, 
                'top': y, 
                'left': x, 
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
        "name": ["fullname", "full name", "ስም", "ሙሉ", "ሙለ", "ሙታ", "ሰም", "ጳም", "ስም:", "ስም፡", "ሙሉስም", "name", "Name", "fullname:", "APF", "AFF", "A-P-F", "Mu"],
        "dob": ["date of birth", "birth", "የውልድ", "ትውልድ", "ቀን", "የልደት", "dob", "ትውልደ", "የልደትቀን"],
        "sex": ["sex", "ፆታ", "ባታ", "ፃታ", "ፆባ", "ባታ", "ፆ", "ታ"],
        "expiry": ["date of expiry", "expiry", "ያሚያበቃበት", "የሚቆይበት", "የሚያበቃበት", "expir", "expire", "expirey"],
        "issue": ["issue date", "issue", "የተሰጠበት"],
        "fcn": ["fan", "ካርድ", "ቁጥር", "fcn", "fina", "fin"]
    }
    
    buckets = {"header": [], "name": [], "dob": [], "sex": [], "expiry": [], "issue": [], "fcn": []}
    active_key = "header"
    
    for line in lines:
        line.sort(key=lambda x: x['left'])
        current_label = active_key
        
        for e in line:
            # Check if this word itself is a label token
            is_label = False
            t_low = e['text'].lower().strip(":| ")
            if len(t_low) < 2: continue # skip noise tokens like | or :

            for key, tokens in label_map.items():
                # Use stricter matching for English labels to avoid name collision
                if any((t == t_low or (len(t) > 3 and t in t_low)) for t in tokens):
                    current_label = key
                    active_key = key 
                    is_label = True
                    break
            
            if not is_label:
                # Extra safety: If we are still in "header" but see something that looks like a name
                # and are below the presumed header Y, we might want to switch to name.
                # But it's safer to wait for the label "Full Name".
                buckets[current_label].append(e)

    # 5. PROCESS BUCKETS (Dual-Pass for Name)
    # -----------------------------
    # proc_full is already prepared at 2x scale
    h_proc, w_proc = proc_full.shape[:2]


    


    # --- Name Processing (Robust Anchor-Based Logic) ---
    name_candidates = []
    
    # Identify DOB anchor
    dob_anchor_y = None
    if buckets.get("dob"):
        dob_anchor_y = min(w['top'] for w in buckets["dob"])
    else:
        # Search globally for "Date of Birth" label if bucket is empty
        for i in range(len(ocr_data['text'])):
            if any(l in ocr_data['text'][i].lower() for l in ["birth", "የውልድ", "ትውልድ"]):
                dob_anchor_y = ocr_data['top'][i]
                break

    # Region of Interest (ROI) for Name
    name_roi_top = 0
    name_roi_bottom = h_proc
    
    if dob_anchor_y:
        # PRIMARY: Catch both Amharic and English above DOB
        name_roi_top = max(id_top_y, dob_anchor_y - 425) 
        name_roi_bottom = dob_anchor_y - 20
    else:
        # FALLBACK: Region between header and first structural field (Sex/Expiry)
        first_field_y = h_proc
        for key in ["sex", "expiry", "fcn"]:
            if buckets.get(key):
                first_field_y = min(first_field_y, min(w['top'] for w in buckets[key]))
        
        name_roi_top = id_top_y
        name_roi_bottom = min(first_field_y - 10, int(h_proc * 0.48))

    # Collect candidates from the spatial region
    spatial_candidates = []
    for i in range(len(ocr_data['text'])):
        txt = ocr_data['text'][i].strip()
        ty = ocr_data['top'][i]
        tx = ocr_data['left'][i]
        
        if name_roi_top <= ty <= name_roi_bottom:
            if tx > w_proc * 0.75:
                continue
            
            if len(txt) <= 1: continue
            
            txt_words = txt.lower().split()
            if not any(w.strip(".:") in header_keywords for w in txt_words):
                spatial_candidates.append({
                    'text': txt, 'top': ty, 'left': tx, 
                    'w': ocr_data['width'][i], 'h': ocr_data['height'][i]
                })

    # Group spatial candidates into lines
    s_lines = []
    if spatial_candidates:
        spatial_candidates.sort(key=lambda x: x['top'])
        cl = [spatial_candidates[0]]
        for w in spatial_candidates[1:]:
            if w['top'] - cl[-1]['top'] < 45: cl.append(w)
            else:
                s_lines.append(sorted(cl, key=lambda x: x['left']))
                cl = [w]
        s_lines.append(sorted(cl, key=lambda x: x['left']))

    # --- Name Processing (Robust Line Association Logic) ---
    extracted_lines = []
    am_anchor_top = None
    am_anchor_text = None
    
    if s_lines:
        for line in s_lines:
            l_top = min(w['top'] for w in line)
            l_text = " ".join([w['text'] for w in line])
            
            # Optimized OCR for each line to catch noisy English names
            try:
                l_bot = max(w['top'] + w['h'] for w in line)
                l_left = min(w['left'] for w in line)
                l_right = max(w['left'] + w['w'] for w in line)
                pad = 12
                line_crop = crop(proc_full, l_top-pad, l_bot+pad, l_left-pad, l_right+pad)
                if line_crop.size > 0:
                    opt_text = pytesseract.image_to_string(line_crop, lang="eng", config="--psm 7").strip()
                    # Apply noise filter to optimized text too (WORD BASED)
                    opt_words = opt_text.lower().split()
                    if not any(w.strip(".:") in header_keywords for w in opt_words):
                        # Only replace if optimized result is 'richer' in letters and looks like a name part
                        if sum(c.isalpha() for c in opt_text) > sum(c.isalpha() for c in l_text) * 0.8:
                            # Clean up common PSM 7 artifacts (trailing dots/marks)
                            opt_text = re.sub(r'[^a-zA-Z\s]', '', opt_text).strip()
                            if len(opt_text.split()) >= len(l_text.split()):
                                l_text = opt_text
            except: pass
            
            extracted_lines.append({'text': l_text, 'top': l_top})
            
    # --- New Robust Amharic Selection ---
    am_cands = []
    for el in extracted_lines:
        if looks_like_amharic_name(el['text']):
            am_cands.append(el)
            
    if am_cands:
        # Pick best Amharic candidate based on score
        best_am_obj = max(am_cands, key=lambda x: score_amharic_name(x['text']))
        am_anchor_top = best_am_obj['top']
        am_anchor_text = best_am_obj['text']
    elif extracted_lines:
        # Fallback to first line with ANY Amharic if no perfect candidate
        for el in extracted_lines:
            am_chars = re.sub(r'[^\u1200-\u137F]', '', el['text'])
            if len(am_chars) >= 5:
                am_anchor_top = el['top']
                am_anchor_text = el['text']
                break

    # 1. Assign Amharic Name
    if am_anchor_text:
        # Keep only Ethiopic script characters (U+1200 to U+137F) and spaces
        am_filtered = re.sub(r'[^\u1200-\u137F\s]', ' ', am_anchor_text)
        data["name_am"] = ' '.join(am_filtered.split()).strip()
    
    # 2. Assign English Name via Association
    best_en = "—"
    if am_anchor_top is not None:
        # Search BELOW (+10 to +120)
        below_cands = []
        for el in extracted_lines:
            diff = el['top'] - am_anchor_top
            if 10 <= diff <= 120:
                # Clean punctuations but keep spaces
                en_clean = re.sub(r'[^a-zA-Z\s]', ' ', el['text']).strip()
                en_clean = ' '.join(en_clean.split())
                if looks_like_name(en_clean): below_cands.append(en_clean)
        
        if below_cands:
            best_en = max(below_cands, key=score_name_candidate)
        else:
            # Search ABOVE (-120 to -10)
            above_cands = []
            for el in extracted_lines:
                diff = am_anchor_top - el['top']
                if 10 <= diff <= 120:
                    en_clean = re.sub(r'[^a-zA-Z\s]', '', el['text']).strip()
                    if looks_like_name(en_clean): above_cands.append(en_clean)
            if above_cands:
                best_en = max(above_cands, key=score_name_candidate)

    # 3. Global Fallback
    if best_en == "—":
        global_cands = []
        for el in extracted_lines:
            en_clean = re.sub(r'[^a-zA-Z\s]', '', el['text']).strip()
            if looks_like_name(en_clean): global_cands.append(en_clean)
        if global_cands:
            best_en = max(global_cands, key=score_name_candidate)
            
    data["name_en"] = best_en

    if data["name_am"] == "—" and data["name_en"] != "—":
        data["name_am"] = transliterate_to_amharic(data["name_en"])

    # --- Image Crop enhancement (Integrated with Spatial Candidates) ---
    if spatial_candidates:
        try:
            # ROI for visual confirmation - using the full anchor-based region to ensure both lines (AM/EN) are captured
            am_top = name_roi_top
            am_bot = name_roi_bottom
            
            # ROI for visual confirmation - using nearly full-width to ensure nothing is cut off
            x0 = max(0, int(w_proc * 0.08)) 
            x1 = min(w_proc, int(w_proc * 0.92))
            
            img_s = cv2.resize(image, (w_proc, h_proc))
            crop_roi = img_s[max(0, am_top):min(h_proc, am_bot), x0:x1]
            if crop_roi.size > 0:
                # 1. High-quality upscale
                roi_z = cv2.resize(crop_roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
                
                # 2. Advanced Denoise (Mild to preserve text edges)
                roi_z = cv2.fastNlMeansDenoisingColored(roi_z, None, 10, 10, 7, 21)
                
                # 3. Visual Polish via PIL (Increased for BOLDNESS)
                pil_roi = Image.fromarray(cv2.cvtColor(roi_z, cv2.COLOR_BGR2RGB))
                pil_roi = ImageEnhance.Contrast(pil_roi).enhance(1.8)
                pil_roi = ImageEnhance.Sharpness(pil_roi).enhance(4.0)
                roi_z = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)

                # 4. Refined Background removal
                gray = cv2.cvtColor(roi_z, cv2.COLOR_BGR2GRAY)
                mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
                
                # --- BOLDNESS ---
                # Erode background (255) to expand foreground text (0)
                kernel_bold = np.ones((2,2), np.uint8)
                mask = cv2.erode(mask, kernel_bold, iterations=1)
                
                # Close/Open mask to remove noise artifacts
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_bold)
                
                bgra = cv2.cvtColor(roi_z, cv2.COLOR_BGR2BGRA)
                bgra[mask == 255] = (0,0,0,0) # Mask matches ID light background
                
                _, buf = cv2.imencode('.png', bgra)
                data["name_am_encoded"] = buf
        except: pass

    # --- Other Fields ---
    eth_exp, greg_exp = "—", "—"  # default values

    for key in ["dob", "sex", "expiry", "issue", "fcn"]:
        if buckets[key] or key in ["dob", "expiry", "issue"]:
          val_str = " ".join([e['text'] for e in sorted(buckets[key], key=lambda x: x['left'])])
        
        if key == "dob":
            data["dob_eth"], data["dob_greg"] = extract_dates_smart(val_str, is_dob=True)
        elif key == "sex":
            low_val = val_str.lower()
            if "female" in low_val or "ሴት" in low_val or "f" == low_val:
                data["sex_en"], data["sex_am"] = "Female", "ሴት"
            elif "male" in low_val or "ወንድ" in low_val or "m" == low_val:
                data["sex_en"], data["sex_am"] = "Male", "ወንድ"
        elif key == "expiry":
            eth_exp, greg_exp = extract_dates_smart(val_str, is_dob=False)
        
        data["expiry_eth"] = eth_exp
        data["expiry_greg"] = greg_exp
            
            # 3. CALCULATION (Now 'eth_exp' and 'greg_exp' are safe to use)
        if greg_exp != "—":
                data["issue_greg"] = calculate_issue_from_expiry(greg_exp, 5)
        if eth_exp != "—":
                data["issue_eth"] = calculate_issue_from_expiry(eth_exp, 5)

        elif key == "issue":
            # Only extract if we haven't already calculated it from Expiry
            if data.get("issue_greg") == "—":
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

    # 6. QR OVERRIDES (Prioritize OCR, Skip Names per user request)
    if qr_data:
        # data["name_en"] = qr_data.get("name", data["name_en"])
        # data["name_am"] = qr_data.get("name_am", data["name_am"])
            
        if qr_data.get("dob"): data["dob_greg"] = qr_data["dob"]
        if qr_data.get("sex"):
            data["sex_en"] = qr_data["sex"]
            data["sex_am"] = "ወንድ" if qr_data["sex"] == "Male" else "ሴት"
        if qr_data.get("fin"): data["id_number"] = qr_data["fin"]

    # 7. FINAL NAME TRANSLITERATION (Smart Fallback)
    if data["name_en"] != "—":
        if not any(0x1200 <= ord(c) <= 0x137F for c in data["name_am"]) or len(data["name_am"]) < 2:
            data["name_am"] = transliterate_to_amharic(data["name_en"])
        else:
            # If we have both, verify word counts
            en_words = data["name_en"].split()
            am_words = data["name_am"].split()
            if len(en_words) != len(am_words) and len(en_words) > 1:
                # Transliteration is often safer than broken OCR
                data["name_am"] = transliterate_to_amharic(data["name_en"])
    
    # ── 8. VERTICAL ISSUE DATE EXTRACTION (Unified Dynamic Layout) ──
    h_img, w_img = image.shape[:2]
    is_portrait = h_img > w_img
    
    # 8.1 Define search areas based on aspect ratio
    # Screenshots are portrait, cards are landscape
    if is_portrait:
        areas = [
            {'x0': int(w_img * 0.82), 'x1': int(w_img * 0.96), 'side': 'right'},
            {'x0': int(w_img * 0.01), 'x1': int(w_img * 0.15), 'side': 'left'} # fallback
        ]
    else:
        areas = [
            {'x0': int(w_img * 0.01), 'x1': int(w_img * 0.15), 'side': 'left'},
            {'x0': int(w_img * 0.85), 'x1': int(w_img * 0.99), 'side': 'right'} # fallback
        ]
        
    # 8.2 Side detection (Primary only, no more debug saves)
    if is_portrait:
        area = {'x0': int(w_img * 0.82), 'x1': int(w_img * 0.98)}
    else:
        area = {'x0': int(w_img * 0.01), 'x1': int(w_img * 0.16)}
    
    best_strip = image[0:h_img, area['x0']:area['x1']]
    
    if best_strip is not None and best_strip.size > 0:
        h_s, w_s = best_strip.shape[:2]
        greg_y, eth_y = None, None
        
        # Multi-rotation OCR to find markers (with safety)
        for rot_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            try:
                rotated = cv2.rotate(best_strip, rot_code)
                s_data = pytesseract.image_to_data(rotated, output_type=pytesseract.Output.DICT)
                for i in range(len(s_data['text'])):
                    txt = s_data['text'][i].strip()
                    if len(txt) < 3: continue
                    if re.search(r'\b(19\d{2}|20\d{2})\b', txt):
                        # Map back to original Y
                        if rot_code == cv2.ROTATE_90_CLOCKWISE:
                            y_val = h_s - s_data['left'][i]
                        else:
                            y_val = s_data['top'][i] # Wait, rotated CCW is (x,y) -> (y, ws-1-x)
                            # Actually, cleaner to just use fallbacks if marker is tricky
                            y_val = h_s - s_data['left'][i] if rot_code == cv2.ROTATE_90_CLOCKWISE else s_data['left'][i]
                        
                        year = int(re.search(r'\b(19\d{2}|20\d{2})\b', txt).group(1))
                        # Identify label context
                        if any(m in txt.lower() for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
                            greg_y = y_val
                        elif year > 2024: greg_y = y_val
                        else: eth_y = y_val
                if greg_y is not None or eth_y is not None: break
            except: pass
        
        # Reliable fallbacks
        if greg_y is None: greg_y = int(h_s * (0.22 if is_portrait else 0.65))
        if eth_y is None: eth_y = int(h_s * (0.35 if is_portrait else 0.30))
            
        pad = int(h_s * 0.08) # 8% height padding
        crop_g = best_strip[max(0, greg_y - pad):min(h_s, greg_y + pad), :]
        crop_e = best_strip[max(0, eth_y - pad):min(h_s, eth_y + pad), :]

        def process_vertical_crop(c):
            if c is None or c.size == 0: return None
            c_res = cv2.resize(c, (20, 110), interpolation=cv2.INTER_LANCZOS4)
            pil = Image.fromarray(cv2.cvtColor(c_res, cv2.COLOR_BGR2RGB))
            pil = ImageEnhance.Contrast(pil).enhance(2.8)
            pil = ImageEnhance.Sharpness(pil).enhance(3.0)
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Gregorian Processing
        ig_proc = process_vertical_crop(crop_g)
        if ig_proc is not None:
            _, buffer = cv2.imencode('.png', ig_proc)
            data["issue_greg_encoded"] = buffer
            try:
                # OCR Fallback
                for rc in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rot = cv2.rotate(crop_g, rc)
                    ocr = pytesseract.image_to_string(rot, lang="eng", config='--psm 7').strip()
                    if re.search(r'\d', ocr): 
                        data["issue_greg"] = ocr
                        break
            except: pass

        # Ethiopic Processing
        ie_proc = process_vertical_crop(crop_e)
        if ie_proc is not None:
            _, buffer = cv2.imencode('.png', ie_proc)
            data["issue_eth_encoded"] = buffer
            try:
                for rc in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rot = cv2.rotate(crop_e, rc)
                    ocr = pytesseract.image_to_string(rot, lang="amh+eng", config='--psm 7').strip()
                    if re.search(r'\b\d{2}/\d{2}/\d{2,4}\b|\b\d{2}-\d{2}-\d{2,4}\b', ocr):
                        data["issue_eth"] = ocr
                        break
            except: pass

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
    skip_labels = {"Region", "Zone", "Woreda", "Address", "ክልል", "ዞን", "ወረዳ", "ቁጥር", "አድራሻ", "APD SAL", "አፕድ ሳል"}

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

    try:
        with open("result.html", "r", encoding="utf-8") as f:
            template_content = f.read()
    except:
        print("result.html not found.")
        return

    # Find ID
    actual_id = "—"
    for k in ["id_number", "fan", "barcode_data"]:
        val = data.get(k)
        if val and val != "—" and len(val) >= 10:
            actual_id = val
            break

    # -----------------------------
    # CALCULATE ISSUE DATE
    # -----------------------------
    from datetime import datetime

    expiry_greg = data.get("expiry_greg") or data.get("expiry") or "—"
    expiry_eth = data.get("expiry_eth", "—")

    issue_greg = "—"
    issue_eth = "—"

    try:
        if expiry_greg != "—":
            d = datetime.strptime(expiry_greg, "%Y-%m-%d")
            issue_greg = d.replace(year=d.year - 5).strftime("%Y-%m-%d")
    except:
        pass

    try:
        if expiry_eth != "—":
            parts = expiry_eth.split("-")
            parts[0] = str(int(parts[0]) - 5)
            issue_eth = "-".join(parts)
    except:
        pass

    # -----------------------------
    # CARD DATA
    # -----------------------------
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

            "expiry_greg": expiry_greg,
            "expiry_eth": expiry_eth,

            "issue_greg_date": issue_greg,
            "issue_eth_date": issue_eth,

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

            "fin_image": "",
            "expiry_image": ""
        }
    }]

    template = Template(template_content)

    def mock_url_for(endpoint, **values):
        return values.get('filename', '')

    html = template.render(cards=cards, url_for=mock_url_for)

    with open("final_id_result.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Exported to final_id_result.html")

# -----------------------------
# 12. MAIN ENTRY POINTS
# -----------------------------
def process_image_front(image_path):
    """
    Stand-alone function for processing only the front image.
    """
    img = load_image(image_path)
    if img is None: return {}
    return extract_front(img)

def process_image(paths):
    """
    General entry point for processing images.
    """
    if not paths: return {}
    img = load_image(paths[0])
    if img is None: return {}
    return extract_front(img)

# -----------------------------
# 13. SCREENSHOTS PROCESSOR
# -----------------------------
def process_screenshots(front_img, back_img, qr_img):
    """
    Specialized function for front_id, back_id, and qr_id images.
    Accepts path strings or OpenCV image objects.
    """
    # 1. Extract assets from qr_id
    portrait_encoded, qr_code_encoded = extract_assets(qr_img)
    
    # 2. Extract QR data
    qr_img_obj = load_image(qr_img)
    qr_raw = read_qr(qr_img_obj)
    qr_data = parse_qr_data(qr_raw)
    
    # 3. Extract Front
    front_img_obj = load_image(front_img)
    front_data = extract_front(front_img_obj, qr_data)
    
    # 4. Extract Back
    back_img_obj = load_image(back_img)
    back_data = extract_back(back_img_obj)
    
    # Merge results
    final_data = {**front_data, **back_data}
    final_data["qr_raw"] = qr_raw
    final_data["portrait_encoded"] = portrait_encoded
    final_data["qr_code_encoded"] = qr_code_encoded
    final_data["name_am_encoded"] = front_data.get("name_am_encoded") # Added
    
    return final_data


# -----------------------------
# 13. RUN PROGRAM
# -----------------------------
if __name__ == "__main__":

    cards = []

    # Process up to 5 cards
    for i in range(1, 6):

        front = f"front_{i}.jpg"
        back = f"back_{i}.jpg"
        qr_id = f"qr_{i}.jpg"

        if all(os.path.exists(f) for f in [front, back, qr_id]):

            print(f"Processing Card {i}...")

            try:
                data = process_screenshots(front, back, qr_id)

                # Avoid print encoding issues with Amharic
                safe_print = {
                    k: v if isinstance(v, str) and not any(ord(c) > 127 for c in v) else "..."
                    for k, v in data.items()
                }

                print(safe_print)

                cards.append(data)

            except Exception as e:
                print(f"Card {i} failed:", e)

    # Export results
    if cards:
        for idx, card in enumerate(cards, start=1):
            export_html(card)
    else:
        # Fallback single-image processing
        image_path = "id_image.jpg"
        if os.path.exists(image_path):
            data = process_image([image_path])
            print(data)
            export_html(data)