import pdfplumber
import pypdfium2 as pdfium
import os, re, base64, io, threading, pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from collections import deque
import cv2

# ── pypdfium2 is NOT thread-safe on Windows — serialize all render calls ───────
_PDFIUM_LOCK = threading.Lock()

# ── Tesseract Configuration ──────────────────────────────────────────────────
# Path calibrated for User's Windows environment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT v2  (portrait)  — original coordinates, unchanged
# ══════════════════════════════════════════════════════════════════════════════

# Photo crop ratios (left card)
PHOTO_X0, PHOTO_Y0 = 0.0935, 0.1206
PHOTO_X1, PHOTO_Y1 = 0.2322, 0.2480

# Right card crop ratios
RIGHT_X0, RIGHT_Y0 = 0.48, 0.00
RIGHT_X1, RIGHT_Y1 = 1.00, 0.75

# Exact pixel coordinates (calibrated for 2481×3508 @ 300 dpi)
EXP_X0, EXP_Y0 = 1734, 1155
EXP_X1, EXP_Y1 = 2017, 1182

FIN_X0, FIN_Y0 = 1978, 2046
FIN_X1, FIN_Y1 = 2251, 2087

ISSUE_GREG_X0, ISSUE_GREG_Y0 = 2220, 534
ISSUE_GREG_X1, ISSUE_GREG_Y1 = 2269, 708

ISSUE_ETH_X0, ISSUE_ETH_Y0 = 2220, 728
ISSUE_ETH_X1, ISSUE_ETH_Y1 = 2266, 879


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT v1  (landscape)  — ratio-based crops
# ══════════════════════════════════════════════════════════════════════════════

# Photo — upper-left of left card panel
PHOTO_V1_X0, PHOTO_V1_Y0 = 0.040, 0.160
PHOTO_V1_X1, PHOTO_V1_Y1 = 0.195, 0.600

# FCN / FIN barcode strip — middle of left card
FIN_V1_X0, FIN_V1_Y0 = 0.030, 0.545
FIN_V1_X1, FIN_V1_Y1 = 0.450, 0.650

# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT v1  (landscape) — ratio-based crops
# ══════════════════════════════════════════════════════════════════════════════

# Issue date row — Captures "Issue Date 2026/02/20" and "13/6/2018"
ISSUE_V1_X0, ISSUE_V1_Y0 = 0.740, 0.760
ISSUE_V1_X1, ISSUE_V1_Y1 = 0.980, 0.820

# Expiry date row — Captures "Expiry Date 2036/02/20" and "13/6/2028"
EXP_V1_X0, EXP_V1_Y0 = 0.740, 0.820
EXP_V1_X1, EXP_V1_Y1 = 0.980, 0.880

# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_is_v1(raw_text):
    """
    Returns True for Format v1 (landscape, pipe-separated bilingual fields).
    Detected by presence of 'FCN:DIGITS' label and pipe-separated sex field.
    """
    has_fcn_label  = bool(re.search(r'FCN:\s*\d{16}', raw_text))
    has_pipe_sex   = bool(re.search(r'(ወንድ|ሴት)\s*\|\s*(Male|Female)', raw_text))
    has_pipe_dob   = bool(re.search(r'\d{2}/\d{2}/\d{4}\s*\|\s*\d{4}/\d{2}/\d{2}', raw_text))
    no_fayda_copy  = "FAYDA DIGITAL COPY" not in raw_text
    return (has_fcn_label or has_pipe_sex or has_pipe_dob) and no_fayda_copy


# ══════════════════════════════════════════════════════════════════════════════
#  CORE HELPERS  (shared)
# ══════════════════════════════════════════════════════════════════════════════

def clean(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def img_to_datauri(pil_img, fmt="PNG", is_qr=False):
    buf = io.BytesIO()
    if is_qr:
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        pil_img.save(buf, format="PNG", compress_level=0)
    else:
        if fmt == "JPEG" and pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format=fmt)
    b64  = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def render_page(pdf_bytes, scale=300/72):
    """Render first page from bytes — thread-safe via lock."""
    with _PDFIUM_LOCK:
        doc    = pdfium.PdfDocument(pdf_bytes)
        page   = doc[0]
        bitmap = page.render(scale=scale, rotation=0)
        img    = bitmap.to_pil()
        doc.close()
    return img


def get_right_card_image(full_page):
    W, H = full_page.size
    return full_page.crop((
        int(W * RIGHT_X0), int(H * RIGHT_Y0),
        int(W * RIGHT_X1), int(H * RIGHT_Y1),
    ))


def crop_photo(full_page):
    """v2 photo crop."""
    W, H = full_page.size
    return full_page.crop((
        int(W * PHOTO_X0), int(H * PHOTO_Y0),
        int(W * PHOTO_X1), int(H * PHOTO_Y1),
    )).convert("RGB")


def crop_photo_v1(full_page):
    """v1 photo crop (landscape layout)."""
    W, H = full_page.size
    return full_page.crop((
        int(W * PHOTO_V1_X0), int(H * PHOTO_V1_Y0),
        int(W * PHOTO_V1_X1), int(H * PHOTO_V1_Y1),
    )).convert("RGB")


def remove_background(pil_img):
    img = pil_img.convert("RGBA")
    arr = np.array(img).copy().astype(np.float32)
    h, w = arr.shape[:2]
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (R + G + B) / 3.0
    bg_candidate = (
        (brightness > 160) &
        ((B > R + 8) | ((R > 220) & (G > 220) & (B > 220)))
    )
    visited = np.zeros((h, w), dtype=bool)
    is_bg   = np.zeros((h, w), dtype=bool)
    queue   = deque()
    for x in range(w):
        queue.append((0, x)); queue.append((h - 1, x))
    for y in range(h):
        queue.append((y, 0)); queue.append((y, w - 1))
    while queue:
        y, x = queue.popleft()
        if visited[y, x]: continue
        visited[y, x] = True
        if bg_candidate[y, x]:
            is_bg[y, x] = True
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    queue.append((ny, nx))
    mask_pil  = Image.fromarray((is_bg * 255).astype(np.uint8), mode='L')
    mask_soft = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
    mask_arr  = np.array(mask_soft) / 255.0
    result    = arr.astype(np.uint8).copy()
    result[:,:,3] = ((1.0 - mask_arr) * 255).astype(np.uint8)
    return Image.fromarray(result, mode='RGBA')


def remove_light_background(pil_img, darkness_threshold=140):
    img = pil_img.convert("RGBA")
    arr = np.array(img).copy()
    R, G, B = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    is_light = (R > darkness_threshold) & (G > darkness_threshold) & (B > darkness_threshold)
    arr[:,:,3] = np.where(is_light, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGBA')


def apply_warm_tone(rgba_img):
    rgb  = rgba_img.convert('RGB')
    gray = rgb.convert('L')
    gray = ImageEnhance.Contrast(gray).enhance(1.1)
    gray = ImageEnhance.Brightness(gray).enhance(1.15)
    arr  = np.array(gray.convert('RGB')).astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * 1.15 + 10, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * 0.98 + 2,  0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * 0.78 - 5,  0, 255)
    rgb_out = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    alpha   = np.array(rgba_img)[:,:,3]
    r, g, b = rgb_out.split()
    return Image.merge('RGBA', [r, g, b, Image.fromarray(alpha)])


from PIL import Image
from pyzbar.pyzbar import decode

def get_tight_crop(img_cv, decoded_obj):
    """
    Refines the bounding box from pyzbar to remove extra margins and patterns.
    """
    x, y, w, h = decoded_obj.rect
    # 1. Focus on the area pyzbar found
    roi = img_cv[max(0, y):y+h, max(0, x):x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Threshold to isolate black modules from background patterns
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Find the footprint of actual black modules
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        tx, ty, tw, th = cv2.boundingRect(coords)
        # Apply the refined coordinates to the original image
        # Adding a tiny 2px padding for better readability
        pad = 2
        final_x = max(0, x + tx - pad)
        final_y = max(0, y + ty - pad)
        final_w = tw + (pad * 2)
        final_h = th + (pad * 2)
        return img_cv[final_y:final_y+final_h, final_x:final_x+final_w]
    
    return roi # Fallback to original if refinement fails

def preprocess_for_qr(img_cv):
    """
    Removes green security patterns and handles uneven lighting on ID cards.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding turns background patterns white and text/QR black
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return processed

def detect_qr_robust(pil_img):
    """
    Detects QR and returns a perfectly tight 300x300 grayscale crop.
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Try 1: Detect on raw image
    decoded_objects = decode(img_cv)
    
    # Try 2: Detect on pre-processed image (better for card photos)
    if not decoded_objects:
        processed = preprocess_for_qr(img_cv)
        decoded_objects = decode(processed)
        
    if decoded_objects:
        # Get the first detected QR and refine the crop
        tight_qr = get_tight_crop(img_cv, decoded_objects[0])
        
        # Resize for better decoding
        qr_pil = Image.fromarray(cv2.cvtColor(tight_qr, cv2.COLOR_BGR2GRAY))
        return qr_pil.resize((600, 600), Image.Resampling.LANCZOS)

    return None

def crop_qr_with_fallback(full_page):
    """
    Standard entry point for both PDFs and full-page scans.
    """
    # 1. Full page detection (handles most cases)
    qr = detect_qr_robust(full_page)
    if qr: return qr

    # 2. Quadrant search (if QR is too small/dense for full-page detection)
    W, H = full_page.size
    quadrants = [
        (0, 0, W//2, H//2),      # Top Left
        (W//2, 0, W, H//2),      # Top Right
        (0, H//2, W//2, H),      # Bottom Left
        (W//2, H//2, W, H)       # Bottom Right
    ]
    
    for quad in quadrants:
        crop = full_page.crop(quad)
        qr = detect_qr_robust(crop)
        if qr: return qr

    return None
# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE EXTRACTION  — v2  (portrait, hardcoded pixels)
# ══════════════════════════════════════════════════════════════════════════════

def extract_fin_expiry_from_rendered(img_cv):
    """Accept pre-rendered BGR numpy array — no re-render. (Format v2 only)"""
    expiry_crop     = img_cv[EXP_Y0:EXP_Y1,               EXP_X0:EXP_X1]
    fin_crop        = img_cv[FIN_Y0:FIN_Y1,               FIN_X0:FIN_X1]
    issue_greg_crop = img_cv[ISSUE_GREG_Y0:ISSUE_GREG_Y1, ISSUE_GREG_X0:ISSUE_GREG_X1]
    issue_eth_crop  = img_cv[ISSUE_ETH_Y0:ISSUE_ETH_Y1,   ISSUE_ETH_X0:ISSUE_ETH_X1]

    def bgr2pil(crop):
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    expiry_pil = remove_light_background(bgr2pil(expiry_crop), darkness_threshold=140)
    fin_pil    = remove_light_background(bgr2pil(fin_crop),    darkness_threshold=200)

    if fin_pil.mode == "RGBA":
        white_bg = Image.new("RGB", fin_pil.size, (255, 255, 255))
        white_bg.paste(fin_pil, mask=fin_pil.split()[3])
        fin_pil = white_bg
    else:
        fin_pil = fin_pil.convert("RGB")

    def process_strip(crop_bgr, target_w=26, target_h=310):
        pil = bgr2pil(crop_bgr).resize((target_w, target_h), Image.Resampling.LANCZOS)
        pil = ImageEnhance.Contrast(pil).enhance(2.8)
        pil = ImageEnhance.Sharpness(pil).enhance(4.5)
        return remove_light_background(pil.convert("RGBA"), darkness_threshold=165)

    # issue_greg_pil = process_strip(issue_greg_crop, target_w=22, target_h=155)
    # issue_eth_pil  = process_strip(issue_eth_crop,  target_w=22, target_h=155)

    return {
        "fin_image":        img_to_datauri(fin_pil,        "PNG"),
        "expiry_image":     img_to_datauri(expiry_pil,     "PNG"),
        "issue_greg_image": "", # img_to_datauri(issue_greg_pil, "PNG"),
        "issue_eth_image":  "", # img_to_datauri(issue_eth_pil,  "PNG"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE EXTRACTION  — v1  (landscape, ratio-based crops)
# ══════════════════════════════════════════════════════════════════════════════

def extract_fin_expiry_v1(full_page):
    """Ratio-based image crops for Format v1 landscape layout."""
    W, H = full_page.size

    def ratio_crop(x0r, y0r, x1r, y1r):
        return full_page.crop((int(W*x0r), int(H*y0r), int(W*x1r), int(H*y1r)))

    # FIN / FCN barcode
    fin_crop = ratio_crop(FIN_V1_X0, FIN_V1_Y0, FIN_V1_X1, FIN_V1_Y1)
    fin_pil  = fin_crop.convert("RGB")

    # Issue date row (detects both date types on this line)
    # issue_crop = ratio_crop(ISSUE_V1_X0, ISSUE_V1_Y0, ISSUE_V1_X1, ISSUE_V1_Y1)
    # issue_pil  = remove_light_background(issue_crop.convert("RGBA"), darkness_threshold=150)

    # Expiry date row (detects both date types on this line)
    exp_crop = ratio_crop(EXP_V1_X0, EXP_V1_Y0, EXP_V1_X1, EXP_V1_Y1)
    exp_pil  = remove_light_background(exp_crop.convert("RGBA"), darkness_threshold=150)

    return {
        "fin_image":         img_to_datauri(fin_pil,   "PNG"),
        "expiry_image":      img_to_datauri(exp_pil,   "PNG"),
        "issue_greg_image":  "", # img_to_datauri(issue_pil, "PNG"),
        "issue_eth_image":   "", # img_to_datauri(issue_pil, "PNG"), 
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT FIELD PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_fields_v1(raw_text):
    """
    Anchor-regex extraction for Format v1.
    Handles the mixed two-column table that pdfplumber interleaves.
    All patterns verified against real v1 pdfplumber output.
    """
    # ── FCN ───────────────────────────────────────────────────────────────────
    m = re.search(r'FCN:\s*(\d{16})', raw_text)
    raw_fcn = m.group(1) if m else ""
    fcn = ' '.join([raw_fcn[i:i+4] for i in range(0, 16, 4)]) if len(raw_fcn) == 16 else raw_fcn

    # ── English name — standalone 3-word title-case line after name label ─────
    name_en = ""
    m = re.search(
        r'First,.*?Surname[^\n]*\n[^\n]+\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        raw_text, re.IGNORECASE
    )
    if m:
        name_en = clean(m.group(1))
    else:
        # fallback: first standalone 3-word title-case line
        m = re.search(r'^([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)$', raw_text, re.MULTILINE)
        if m: name_en = clean(m.group(1))

    # ── Amharic name — Ethiopic words right after name label ─────────────────
    name_am = ""
    m = re.search(
        r'First,.*?Surname[^\n]*\n([\u1200-\u137F]+(?:\s+[\u1200-\u137F]+)*)',
        raw_text, re.IGNORECASE
    )
    if m: name_am = clean(m.group(1))

    # ── Date of Birth (pipe-separated on one line) ────────────────────────────
    m = re.search(r'(\d{2}/\d{2}/\d{4})\s*\|\s*(\d{4}/\d{2}/\d{2})', raw_text)
    dob_greg = m.group(1) if m else ""
    dob_eth  = m.group(2) if m else ""

    # ── Region ───────────────────────────────────────────────────────────────
    # English: appears after the dob_eth value on the same line
    reg_en = ""
    m = re.search(r'\d{4}/\d{2}/\d{2}\s+([A-Za-z][A-Za-z\s]+?)(?:\n|$)', raw_text)
    if m: reg_en = clean(m.group(1))

    # Amharic: standalone Ethiopic line immediately after Region/City Admin label
    reg_am = ""
    m = re.search(r'Region/City Admin\n([\u1200-\u137F]+)', raw_text)
    if m: reg_am = clean(m.group(1))

    # ── Sex (pipe-separated) ──────────────────────────────────────────────────
    m = re.search(r'(ወንድ|ሴት)\s*\|\s*(Male|Female)', raw_text)
    sex_am = m.group(1) if m else ""
    sex_en = m.group(2) if m else ""

    # ── Zone ─────────────────────────────────────────────────────────────────
    # English: appears after Male/Female on the same line
    zone_en = ""
    m = re.search(r'(?:Male|Female)\s+([A-Z][A-Za-z\s]*?\sZone)\b', raw_text)
    if m: zone_en = clean(m.group(1))

    # Amharic: standalone Ethiopic line immediately after Zone/City Admin label
    zone_am = ""
    m = re.search(r'Zone/City Admin[^\n]*\n([\u1200-\u137F\s]+?)(?:\n|$)', raw_text)
    if m: zone_am = clean(m.group(1))

    # ── Nationality & Woreda ──────────────────────────────────────────────────
    nat_am = ""; woreda_am = ""
    # Digital ID style: "ኢትዮጵያዊ ምንጃርና ሸንኮራ"
    m_dig_am = re.search(r'Nationality.*?\n(ኢትዮጵያ(?:ዊ|ዊት)?)\s+([\u1200-\u137F\s]+)', raw_text, re.IGNORECASE)
    if m_dig_am:
        nat_am = clean(m_dig_am.group(1))
        woreda_am = clean(m_dig_am.group(2))
    else:
        m = re.search(r'^(Ethiopia|ኢትዮጵያ)\s+([\u1200-\u137F\s]+)$', raw_text, re.MULTILINE)
        if m:
            nat_am    = clean(m.group(1))
            woreda_am = clean(m.group(2))
        else:
            m2 = re.search(r'\b(ኢትዮጵያ(?:ዊ|ዊት)?)\b', raw_text)
            if m2: nat_am = clean(m2.group(1))

    nat_en = ""; woreda_en = ""
    m_dig_en = re.search(r'Nationality.*?\n.*?\n(Ethiopian|Eritrean)\s+([A-Z][A-Za-z\s]+?)(?:\n|$)', raw_text, re.IGNORECASE)
    if m_dig_en:
        nat_en = clean(m_dig_en.group(1))
        woreda_en = clean(m_dig_en.group(2))
    else:
        m = re.search(r'^(Ethiopian)\s+([A-Z][A-Za-z\s]+?)(?:\n|$)', raw_text, re.MULTILINE)
        if m:
            nat_en    = "Ethiopian"
            woreda_en = clean(m.group(2))
        else:
            m2 = re.search(r'\b(Ethiopian|Eritrean)\b', raw_text)
            if m2: nat_en = m2.group(1)

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone = ""
    m = re.search(r'^(0[79]\d{8})$', raw_text, re.MULTILINE)
    if m: phone = m.group(1)
    if not phone:
        m = re.search(r'\b(0[79]\d{8})\b', raw_text)
        if m: phone = m.group(1)

    # ── SN ────────────────────────────────────────────────────────────────────
    sn = ""
    m = re.search(r'\bSN\s+(\d{2,4})\b', raw_text)
    if m: sn = m.group(1)

    # ── Issue Dates ──────────────────────────────────────────────────────────
    # Format A: Issue Date 2026/02/20 የተሰጠበት ቀን 13/6/2018
    m_iss = re.search(r'Issue Date\s*(\d{4}/\d{2}/\d{2})\s*የተሰጠበት ቀን\s*(\d{1,2}/\d{1,2}/\d{4})', raw_text, re.IGNORECASE)
    issue_eth = ""; issue_greg = ""
    if m_iss:
        issue_greg, issue_eth = m_iss.group(1), m_iss.group(2)
    else:
        # Format B: Piped (DD/MM/YYYY | YYYY/MM/DD)
        m_iss = re.search(r'(\d{2}/\d{2}/\d{4})\s*\|\s*(\d{4}/\d{2}/\d{2}).*?(Issue Date|የተሰጠበት ቀን)', raw_text, re.IGNORECASE | re.DOTALL)
        if not m_iss:
            m_iss = re.search(r'(Issue Date|የተሰጠበት ቀን).*?(\d{2}/\d{2}/\d{4})\s*\|\s*(\d{4}/\d{2}/\d{2})', raw_text, re.IGNORECASE | re.DOTALL)
        
        if m_iss:
            if re.match(r'\d', m_iss.group(1)):
                issue_eth, issue_greg = m_iss.group(1), m_iss.group(2)
            else:
                issue_eth, issue_greg = m_iss.group(2), m_iss.group(3)

    # ── Expiry Dates ─────────────────────────────────────────────────────────
    # Format A: Expiry Date 2036/02/20 የሚያበቃበት ቀን 13/6/2028
    m_exp = re.search(r'Expiry Date\s*(\d{4}/\d{2}/\d{2})\s*የሚያበቃበት ቀን\s*(\d{1,2}/\d{1,2}/\d{4})', raw_text, re.IGNORECASE)
    expiry_eth = ""; expiry_greg = ""
    if m_exp:
        expiry_greg, expiry_eth = m_exp.group(1), m_exp.group(2)
    else:
        # Format B: Piped
        m_exp_piped = re.search(r'(\d{2}/\d{2}/\d{4})\s*\|\s*(\d{4}/\d{2}/\d{2}).*?(Expiry Date|የሚቆይበት ቀን)', raw_text, re.IGNORECASE | re.DOTALL)
        if not m_exp_piped:
            m_exp_piped = re.search(r'(Expiry Date|የሚቆይበት ቀን).*?(\d{2}/\d{2}/\d{4})\s*\|\s*(\d{4}/\d{2}/\d{2})', raw_text, re.IGNORECASE | re.DOTALL)
        
        if m_exp_piped:
            if re.match(r'\d', m_exp_piped.group(1)):
                expiry_eth, expiry_greg = m_exp_piped.group(1), m_exp_piped.group(2)
            else:
                expiry_eth, expiry_greg = m_exp_piped.group(2), m_exp_piped.group(3)
        else:
            # fallback: find last two dates in text if we haven't found issue yet
            all_dates = re.findall(r'\b(\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2})\b', raw_text)
            if len(all_dates) >= 4:
                expiry_eth, expiry_greg = all_dates[-2], all_dates[-1]

    return {
        "fcn": fcn, "name_am": name_am, "name_en": name_en,
        "dob_greg": dob_greg, "dob_eth": dob_eth,
        "expiry_greg": expiry_greg, "expiry_eth": expiry_eth,
        "issue_greg": issue_greg, "issue_eth": issue_eth,
        "sex_am": sex_am,  "sex_en": sex_en,
        "reg_am": reg_am,  "reg_en": reg_en,
        "zone_am": zone_am, "zone_en": zone_en,
        "nat_am": nat_am,  "nat_en": nat_en,
        "woreda_am": woreda_am, "woreda_en": woreda_en,
        "phone": phone, "sn": sn,
    }


def parse_fields_v2(lines, fcn_line_idx, raw_text):
    """
    Line-offset extraction for Format v2.
    This is the original parsing logic, unchanged.
    """
    def first_re(pattern, flags=re.MULTILINE):
        m = re.search(pattern, raw_text, flags)
        return clean(m.group(1)) if m else ""

    def gl(offset):
        i = fcn_line_idx + offset
        return lines[i] if 0 <= i < len(lines) else ""

    # Amharic name — line before FCN
    name_am = ""
    if fcn_line_idx > 0:
        c = lines[fcn_line_idx - 1]
        if re.match(r'^[\u1200-\u137F\s]+$', c):
            name_am = clean(c)

    m1 = re.match(r'(\d{2}/\d{2}/\d{4})\s+(.*)', gl(1))
    dob_greg = m1.group(1) if m1 else ""
    reg_am   = clean(m1.group(2)) if m1 else ""

    m2 = re.match(r'(\d{4}/\d{2}/\d{2})\s+(.*)', gl(2))
    dob_eth = m2.group(1) if m2 else ""
    reg_en  = clean(m2.group(2)) if m2 else ""

    _sm     = re.match(r'^(ወንድ|ሴት)\s+(.*)', gl(3))
    sex_am  = _sm.group(1) if _sm else ""
    zone_am = clean(_sm.group(2)) if _sm else ""

    m4      = re.match(r'(Male|Female)\s+(.*)', gl(4))
    sex_en  = m4.group(1) if m4 else ""
    zone_en = clean(m4.group(2)) if m4 else ""

    _nw       = gl(5).split()
    nat_am    = _nw[0] if _nw else ""
    woreda_am = ' '.join(_nw[1:]) if len(_nw) > 1 else ""

    m6        = re.match(r'([A-Z][a-z]+)\s+(.*)', gl(6))
    nat_en = ""; woreda_en = ""
    if m6:
        nat_en    = clean(m6.group(1))
        woreda_en = clean(m6.group(2))
    else:
        nat_en = clean(gl(6))

    # Sanity check: if Nationality looks like Sex, it's likely a shift
    if nat_am in ["ወንድ", "ሴት", "Male", "Female"]: 
        nat_am = ""; woreda_am = ""
    if nat_en in ["Male", "Female"]: 
        nat_en = ""; woreda_en = ""

    m7    = re.match(r'(\d{9,})', gl(7))
    phone = m7.group(1) if m7 else first_re(r'(\d{10,})')

    sn = ""
    sn_m = re.search(r'\bSN\s+(\d{2,4})\b', raw_text)
    if sn_m:
        sn = sn_m.group(1)
    else:
        sn_m2 = re.search(r'(?:^|\s)(0[1-9])\s*(?:\n|$)', raw_text, re.MULTILINE)
        if sn_m2:
            sn = sn_m2.group(1)

    return {
        "name_am": name_am, "name_en": "",  # name_en comes from FCN-line in caller
        "dob_greg": dob_greg, "dob_eth": dob_eth,
        "sex_am": sex_am,  "sex_en": sex_en,
        "reg_am": reg_am,  "reg_en": reg_en,
        "zone_am": zone_am, "zone_en": zone_en,
        "nat_am": nat_am,  "nat_en": nat_en,
        "woreda_am": woreda_am, "woreda_en": woreda_en,
        "phone": phone, "sn": sn,
    }


def parse_fields_flexible(raw_text):
    """
    Robust fallback parser that searches for keywords and patterns anywhere in the text.
    Handles irregular layouts and reversed text extraction.
    """
    data = {
        "fcn": "", "name_am": "", "name_en": "",
        "dob_greg": "", "dob_eth": "",
        "sex_am": "", "sex_en": "",
        "reg_am": "", "reg_en": "",
        "zone_am": "", "zone_en": "",
        "nat_am": "", "nat_en": "",
        "woreda_am": "", "woreda_en": "",
        "phone": "", "sn": "",
    }
    
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    
    # 1. Name Detection (Top-Down search for first valid Name Candidates)
    # We look for a line of 2-5 words with Title Case (English) 
    # and potentially the line immediately above it (Amharic)
    for i, l in enumerate(lines):
        # English Name Candidate
        en_words = l.split()
        if not data["name_en"] and len(en_words) >= 2 and len(en_words) <= 5:
            if all(len(w)>0 and w[0].isupper() for w in en_words if len(w) > 1):
                labels_en = ["ethiopia", "fayda", "digital", "copy", "region", "zone", "woreda", "fcn", "phone", "number", "sex", "gender", "date", "birth", "signature", "national", "id", "program", "debre", "birhan", "amhara", "office"]
                if not any(x.lower() in l.lower().replace(' ', '') for x in labels_en):
                    data["name_en"] = clean(l)
                    # If we found EN name, check the line ABOVE for Amharic name
                    if i > 0:
                        prev_l = lines[i-1]
                        am_chars = re.sub(r'[^\u1200-\u137F]', '', prev_l).strip()
                        if len(am_chars) > 5 and len(prev_l.split()) <= 5:
                            labels_am = ["ሙሉ", "ስም", "ትውልድ", "ፆታ", "ክልል", "ዞን", "ወረዳ", "ቁጥር", "ስልክ", "ጡረቁይ", "ብሔራዊ", "መታወቂያ", "ፕሮግራም", "ግልባጭ", "ፊርማ", "ይቁረጡ", "ኢትዮጵያ", "ብርሃን", "ደብረ", "አማራ", "መረጃ", "መግለጫ", "በተመለከተ", "መግለጫ", "ናቸው", "ተመዝጋቢው"]
                            if not any(x in prev_l.replace(' ', '') for x in labels_am):
                                data["name_am"] = clean(prev_l)
        
        # Fallback for Amharic name if not found above
        if not data["name_am"]:
            am_chars = re.sub(r'[^\u1200-\u137F]', '', l).strip()
            if len(am_chars) > 5 and len(l.split()) <= 5:
                 labels_am = ["ሙሉ", "ስም", "ትውልድ", "ፆታ", "ክልል", "ዞን", "ወረዳ", "ቁጥር", "ስልክ", "ጡረቁይ", "ብሔራዊ", "መታወቂያ", "ፕሮግራም", "ግልባጭ", "ፊርማ", "ይቁረጡ", "ኢትዮጵያ", "ብርሃን", "ደብረ", "አማራ", "መረጃ", "መግለጫ", "በተመለከተ", "ናቸው", "ተመዝጋቢው"]
                 if not any(x in l.replace(' ', '') for x in labels_am):
                     data["name_am"] = clean(l)

    # 2. Sex Detection
    m_sex = re.search(r'(ወንድ|ሴት)\b', raw_text)
    if m_sex: data["sex_am"] = m_sex.group(1)
    
    m_sex_en = re.search(r'\b(Male|Female)\b', raw_text, re.IGNORECASE)
    if m_sex_en: data["sex_en"] = m_sex_en.group(1).capitalize()

    # 2.5 Nationality Detection
    m_nat_en = re.search(r'\b(Ethiopian|Eritrean|Ethiopia|Djiboutian|Somali|Sudanese|Kenyan)\b', raw_text, re.IGNORECASE)
    if m_nat_en: 
        nat_cand = m_nat_en.group(1).capitalize()
        data["nat_en"] = "Ethiopian" if nat_cand == "Ethiopia" else nat_cand
    
    m_nat_am = re.search(r'(ኢትዮጵያ(?:ዊ|ዊት)?)\b', raw_text)
    if m_nat_am:
        nat_cand_am = m_nat_am.group(1)
        data["nat_am"] = "ኢትዮጵያዊ" if nat_cand_am == "ኢትዮጵያ" else nat_cand_am

    # 3. Date Detection (DOB/Expiry/Issue)
    # Search for all DD/MM/YYYY or YYYY/MM/DD matches
    dates = re.findall(r'(?<![a-zA-Z0-9/])(\d{1,2}/\d{1,2}/\d{4}|\d{4}/\d{2}/\d{2})(?![a-zA-Z0-9/])', raw_text)
    if dates:
        # Sort by year to categorize
        processed_dates = []
        for d in dates:
            y_match = re.search(r'\d{4}', d)
            if y_match: processed_dates.append((d, int(y_match.group(0))))
        
        # Unique and sorted by year
        seen = set()
        unique_dates = []
        for d, y in sorted(processed_dates, key=lambda x: x[1]):
            if d not in seen:
                unique_dates.append((d, y))
                seen.add(d)

        if len(unique_dates) >= 6:
            # Maybe DOB, Issue, Expiry
            data["dob_eth"] = unique_dates[0][0]
            data["dob_greg"] = unique_dates[1][0]
            data["issue_eth"] = unique_dates[2][0]
            data["issue_greg"] = unique_dates[3][0]
            data["expiry_eth"] = unique_dates[4][0]
            data["expiry_greg"] = unique_dates[5][0]
        elif len(unique_dates) >= 4:
            # DOB is smaller, Expiry is larger
            data["dob_eth"] = unique_dates[0][0]
            data["dob_greg"] = unique_dates[1][0]
            data["expiry_eth"] = unique_dates[2][0]
            data["expiry_greg"] = unique_dates[3][0]
        elif len(unique_dates) >= 2:
            # If only 2, check if they are DOB or Expiry
            # Usually if years are > 2025, it's Expiry
            if unique_dates[0][1] > 2025:
                 data["expiry_eth"] = unique_dates[0][0]
                 data["expiry_greg"] = unique_dates[1][0]
            else:
                 data["dob_eth"] = unique_dates[0][0]
                 data["dob_greg"] = unique_dates[1][0]

    # 4. Regional Info (Region, Zone, Woreda)
    # Often found on lines with DOB or Sex
    for l in lines:
        if "አማራ" in l or "Amhara" in l:
            data["reg_am"] = "አማራ"
            data["reg_en"] = "Amhara"
        elif "ኦሮሚያ" in l or "Oromia" in l:
            data["reg_am"] = "ኦሮሚያ"
            data["reg_en"] = "Oromia"
            
        if "ዞን" in l:
            data["zone_am"] = clean(re.sub(r'^[ወንድ|ሴት|Male|Female|\s]+', '', l))
        if "Zone" in l:
            m_z = re.search(r'(.*?\sZone)\b', l, re.IGNORECASE)
            if m_z:
                data["zone_en"] = clean(re.sub(r'^[ወንድ|ሴት|Male|Female|\s]+', '', m_z.group(1)))
            else:
                data["zone_en"] = clean(re.sub(r'^[ወንድ|ሴት|Male|Female|\s]+', '', l))
            
        if "ደብረ ብርሃን" in l or "Debre Birhan" in l:
            data["woreda_am"] = "ደብረ ብርሃን"
            data["woreda_en"] = "Debre Birhan"

    # 5. Phone & FCN
    m_phone = re.search(r'\b(0[79]\d{8})\b', raw_text)
    if m_phone: data["phone"] = m_phone.group(1)
    
    # Try more aggressive FCN patterns: look for 16 digits even if jumbled
    fcn_candidate = re.sub(r'[^0-9]', '', raw_text)
    m_fcn_long = re.search(r'\d{16}', fcn_candidate)
    if m_fcn_long:
        raw_fcn = m_fcn_long.group(0)
        data["fcn"] = ' '.join([raw_fcn[j:j+4] for j in range(0, 16, 4)])
    else:
        # Fallback to standard regex
        m_fcn = re.search(r'(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})|(\d{16})', raw_text)
        if m_fcn:
            raw_fcn = m_fcn.group(0).replace(' ', '').replace('\n', '')
            if len(raw_fcn) == 16:
                data["fcn"] = ' '.join([raw_fcn[j:j+4] for j in range(0, 16, 4)])

    return data


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_all(pdf_bytes):
    """
    ONE render pass for the whole page.
    Automatically detects Format v1 (landscape) or v2 (portrait).
    Returns (data_dict, photo_uri, photo_warm_uri, qr_uri, right_card_uri).
    """
    # Initialize defaults to avoid UnboundLocalError
    f = {"fcn": "", "name_am": "", "name_en": "", "sex_am": "", "sex_en": "", "dob_greg": "", "dob_eth": "", "expiry_greg": "", "expiry_eth": ""}
    fcn = ""
    name_en_from_fcn = ""
    fcn_line_idx = None
    # ── 1. Render once ────────────────────────────────────────────────────────
    full_page = render_page(pdf_bytes)
    img_cv    = cv2.cvtColor(np.array(full_page), cv2.COLOR_RGB2BGR)

    # ── 2. Text extraction ────────────────────────────────────────────────────
    raw_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                raw_text += t + "\n"

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # Deduplicate doubled characters if present (happens in some Digital ID PDFs)
    def deduplicate(t):
        if not t: return t
        # Only deduplicate if many characters are doubled (heuristic for problematic PDFs)
        doubles = len(re.findall(r'(.)\1', t))
        if doubles > 20: 
            res = []
            i = 0
            while i < len(t):
                if i + 1 < len(t) and t[i] == t[i+1]:
                    res.append(t[i])
                    i += 2
                else:
                    res.append(t[i])
                    i += 1
            return "".join(res)
        return t

    raw_text = deduplicate(raw_text)
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # ── 3. Detect format ──────────────────────────────────────────────────────
    is_v1 = detect_is_v1(raw_text)

    # ── 4. Parse text fields ──────────────────────────────────────────────────
    if is_v1:
        f   = parse_fields_v1(raw_text)
        fcn = f["fcn"]

    else:
        
        fcn = ""; name_en_from_fcn = ""; fcn_line_idx = None

        for i, line in enumerate(lines):
            
            m = re.match(r'^(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\s+(.*)', line)
            if m:
                raw_fcn = m.group(1).replace(' ', '')
                fcn = ' '.join([raw_fcn[j:j+4] for j in range(0, 16, 4)])
                name_en_from_fcn = clean(m.group(2))
                fcn_line_idx = i
                break
            # Case B: bare 16-digit line (fallback)
            if re.match(r'^\d{16}$', line.replace(' ', '')):
                raw_fcn = line.replace(' ', '')
                fcn = ' '.join([raw_fcn[j:j+4] for j in range(0, 16, 4)])
                fcn_line_idx = i
                break

        if fcn_line_idx is None:
            m_fcn = re.search(r'(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})', raw_text)
            if m_fcn:
                raw_fcn = m_fcn.group(1).replace(' ', '')
                fcn = ' '.join([raw_fcn[j:j+4] for j in range(0, 16, 4)])
            fcn_line_idx = len(lines) - 10

        f = parse_fields_v2(lines, fcn_line_idx, raw_text)
        # Inject name_en detected from FCN line (not available inside parse_fields_v2)
        if name_en_from_fcn:
            f["name_en"] = name_en_from_fcn

    # ── 5. Image crops ────────────────────────────────────────────────────────
    # ── 5. Image crops & QR supplement ────────────────────────────────────────
    qr_pil = crop_qr_with_fallback(full_page)
    qr_uri = img_to_datauri(qr_pil, "PNG", is_qr=True) if qr_pil else ""
    
    # NEW: Decode QR and supplement missing fields
    qr_parsed = {}
    if qr_pil:
        try:
            qr_cv = cv2.cvtColor(np.array(qr_pil), cv2.COLOR_RGB2BGR)
            decoded = decode(qr_cv)
            if decoded:
                qr_string = decoded[0].data.decode("utf-8")
                qr_parsed = parse_qr_data(qr_string)
        except Exception as e:
            print(f"⚠️ QR decode failure: {e}")

    # NEW: Flexible fallback if name/fcn is missing
    # If fcn was not found by standard v2 line-offsets, we trust flexible parser more.
    if not f.get("name_en") or not fcn or not name_en_from_fcn:
        f_flex = parse_fields_flexible(raw_text)
        # Merge flex data into f. 
        for k, v in f_flex.items():
            if k == "fcn":
                if not fcn: fcn = v
            else:
                # If we didn't have a real FCN anchor from text, flex values are more reliable
                # Also override if v looks much more specific (e.g. Debre Birhan vs Zone name)
                if not name_en_from_fcn or not f.get(k) or len(v) > len(f.get(k, "")) or f.get(k) in ["Male", "Female", "ወንድ", "ሴት"]: 
                    if v: f[k] = v

    # NEW: QR Data Overrides (Most reliable source)
    if qr_parsed:
        if qr_parsed.get("qr_fcn"): fcn = qr_parsed["qr_fcn"]
        if qr_parsed.get("qr_name"):
            # QR name is often English, but can be mixed
            q_name = qr_parsed["qr_name"]
            if any(0x1200 <= ord(c) <= 0x137F for c in q_name):
                f["name_am"] = q_name
            else:
                f["name_en"] = q_name
        if qr_parsed.get("qr_gender"):
            f["sex_en"] = qr_parsed["qr_gender"]
            f["sex_am"] = "ወንድ" if f["sex_en"] == "Male" else "ሴት"
        if qr_parsed.get("qr_dob_eth"):
            f["dob_eth"] = qr_parsed["qr_dob_eth"]
        if qr_parsed.get("qr_expiry_eth"):
            f["expiry_eth"] = qr_parsed["qr_expiry_eth"]

    def sub_years(date_str, years=8):
        if not date_str or '/' not in date_str: return ""
        m = re.search(r'(\d{4})', date_str)
        if m:
            old_y = int(m.group(1))
            new_y = old_y - years
            return date_str.replace(str(old_y), str(new_y))
        return ""

    def ocr_extract_date(pil_img):
        """Helper to OCR an image and find the first valid date."""
        if not pil_img: return ""
        try:
            # Upscale and enhance for better OCR on small date fields
            w, h = pil_img.size
            if w < 200:
                pil_img = pil_img.resize((w*3, h*3), Image.Resampling.LANCZOS)
            text = pytesseract.image_to_string(pil_img, config='--psm 7').strip()
            # Find YYYY/MM/DD or DD/MM/YYYY
            m = re.search(r'\b(\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2})\b', text)
            return m.group(1) if m else ""
        except:
            return ""


    if f.get("expiry_greg") and not f.get("issue_greg"):
        f["issue_greg"] = sub_years(f["expiry_greg"])
    if f.get("expiry_eth") and not f.get("issue_eth"):
        f["issue_eth"] = sub_years(f["expiry_eth"])

    if is_v1:
        ef    = extract_fin_expiry_v1(full_page)
        color = crop_photo_v1(full_page)
    else:
        ef    = extract_fin_expiry_from_rendered(img_cv)
        color = crop_photo(full_page)

    # Try OCR fallback for expiry if text extraction failed (NOW ef is populated)
    if not f.get("expiry_greg") and isinstance(ef, dict) and ef.get("expiry_image"):
        try:
            prefix = "data:image/png;base64,"
            if ef["expiry_image"].startswith(prefix):
                b64_data = ef["expiry_image"][len(prefix):]
                expiry_pil_ocr = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                txt_all = pytesseract.image_to_string(expiry_pil_ocr, config='--psm 6')
                found_dates = re.findall(r'\b(\d{2,4}[/-][\d\w]{2,}[/-]\d{2,4})\b', txt_all)
                dated_years = []
                for d in found_dates:
                    ym = re.search(r'\d{4}', d)
                    if ym: dated_years.append((d, int(ym.group(0))))
                if len(dated_years) >= 2:
                    sorted_d = sorted(dated_years, key=lambda x: x[1])
                    f["expiry_eth"] = sorted_d[0][0]
                    f["expiry_greg"] = sorted_d[1][0]
                elif len(dated_years) == 1:
                    d_val, y_val = dated_years[0]
                    if y_val > 2030: f["expiry_greg"] = d_val
                    else: f["expiry_eth"] = d_val
        except: pass

    # Recalculate Issue Date if Expiry was found via OCR
    if f.get("expiry_greg") and not f.get("issue_greg"):
        f["issue_greg"] = sub_years(f["expiry_greg"])
    if f.get("expiry_eth") and not f.get("issue_eth"):
        f["issue_eth"] = sub_years(f["expiry_eth"])

    nobg           = remove_background(color)
    photo_uri      = img_to_datauri(nobg,                  "PNG")
    photo_warm_uri = img_to_datauri(apply_warm_tone(nobg), "PNG")
    # qr_uri already generated above
    right_card_uri = img_to_datauri(get_right_card_image(full_page),  "PNG")

    data = {
        "fcn":        fcn,
        "name_am":    f.get("name_am", ""),
        "name_en":    f.get("name_en", ""),
        "dob_greg":   f.get("dob_greg", ""),
        "dob_eth":    f.get("dob_eth", ""),
        "sex_am":     f.get("sex_am", ""),
        "sex_en":     f.get("sex_en", ""),
        "nat_am":     f.get("nat_am", ""),
        "nat_en":     f.get("nat_en", ""),
        "phone":      f.get("phone", ""),
        "reg_am":     f.get("reg_am", ""),
        "reg_en":     f.get("reg_en", ""),
        "zone_am":    f.get("zone_am", ""),
        "zone_en":    f.get("zone_en", ""),
        "woreda_am":  f.get("woreda_am", ""),
        "woreda_en":  f.get("woreda_en", ""),
        "expiry_greg": f.get("expiry_greg", ""),
        "expiry_eth":  f.get("expiry_eth", ""),
        "issue_greg":  f.get("issue_greg", ""),
        "issue_eth":   f.get("issue_eth", ""),
        "sn":         f.get("sn", ""),
        "fin":        f.get("fcn", "") if is_v1 else "",
        "fin_image":        ef["fin_image"],
        "expiry_image":     ef["expiry_image"],
        "issue_greg_image": ef["issue_greg_image"],
        "issue_eth_image":  ef["issue_eth_image"],
        "raw_text":   raw_text,
    }
    return data, photo_uri, photo_warm_uri, qr_uri, right_card_uri


# ══════════════════════════════════════════════════════════════════════════════
#  QR PARSER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def parse_qr_data(qr_string):
    result = {
        "qr_name": "", "qr_gender": "", "qr_fcn": "",
        "qr_dob_eth": "", "qr_version": "",
        "qr_raw": qr_string[:80] + "..." if len(qr_string) > 80 else qr_string
    }
    if not qr_string or ":DLT:" not in qr_string:
        return result
    segments = qr_string.split(":DLT:", 1)[1].split(":")
    try:
        sign_idx = next((i for i, s in enumerate(segments) if s == "SIGN"), None)
        if sign_idx is None:
            return result
        if sign_idx >= 2 and segments[sign_idx - 2] == "D":
            result["qr_dob_eth"] = segments[sign_idx - 1]
            fcn_raw = segments[sign_idx - 3]
            result["qr_fcn"] = ' '.join([fcn_raw[i:i+4] for i in range(0, 16, 4)]) \
                if len(fcn_raw) == 16 and fcn_raw.isdigit() else fcn_raw
            v_idx = next((i for i, s in enumerate(segments) if s == "V"), None)
            if v_idx is not None:
                result["qr_name"]    = ":".join(segments[:v_idx])
                result["qr_version"] = segments[v_idx + 1] if v_idx + 1 < len(segments) else ""
                for i in range(v_idx, sign_idx):
                    if segments[i] == "G" and i + 1 < sign_idx:
                        g = segments[i + 1]
                        result["qr_gender"] = "Male" if g == "M" else "Female" if g == "F" else g
                        break
    except Exception as e:
        print(f"QR parse error: {e}")
    return result