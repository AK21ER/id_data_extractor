"""
pdf_to_card.py — Core logic module.
Supports both PDF formats:
  • Format v1 — landscape page (2469×1752), pipe-separated bilingual fields,
                disclaimer on right half. (e.g. AKLIL_KELILEW style)
  • Format v2 — portrait page (2481×3508), Fayda Digital Copy on right,
                FCN+name on same line.  (e.g. fayda_id_XXXXXX style)

Contains ONLY extraction functions.  No Flask app lives here.
Import extract_all() and parse_qr_data() from app.py.
"""

import pdfplumber
import pypdfium2 as pdfium
import os, re, base64, io, threading
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from collections import deque
import cv2

# ── pypdfium2 is NOT thread-safe on Windows — serialize all render calls ───────
_PDFIUM_LOCK = threading.Lock()

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

ISSUE_GREG_X0, ISSUE_GREG_Y0 = 2240, 534
ISSUE_GREG_X1, ISSUE_GREG_Y1 = 2269, 708

ISSUE_ETH_X0, ISSUE_ETH_Y0 = 2240, 728
ISSUE_ETH_X1, ISSUE_ETH_Y1 = 2266, 879


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT v1  (landscape)  — ratio-based crops
# ══════════════════════════════════════════════════════════════════════════════

# Photo — upper-left of left card panel
PHOTO_V1_X0, PHOTO_V1_Y0 = 0.060, 0.115
PHOTO_V1_X1, PHOTO_V1_Y1 = 0.195, 0.520

# FCN / FIN barcode strip — middle of left card
FIN_V1_X0, FIN_V1_Y0 = 0.030, 0.545
FIN_V1_X1, FIN_V1_Y1 = 0.450, 0.650

# Issue date row — bottom-right of disclaimer section
ISSUE_V1_X0, ISSUE_V1_Y0 = 0.745, 0.710
ISSUE_V1_X1, ISSUE_V1_Y1 = 0.965, 0.775

# Expiry date row — just below issue date
EXP_V1_X0, EXP_V1_Y0 = 0.745, 0.775
EXP_V1_X1, EXP_V1_Y1 = 0.965, 0.840


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
def preprocess_for_qr(img_cv):
    """
    Advanced preprocessing to remove background patterns (like green ID lines)
    and handle uneven lighting.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding (Crucial for IDs with patterns)
    # This calculates thresholds locally, ignoring global color shifts/backgrounds
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. Optional: Dilation to connect broken modules in blurry scans
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    
    return processed

def detect_qr_robust(pil_img):
    """
    Robustly detects QR from a PIL image using adaptive pre-processing.
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Try 1: Detect on raw image (works for clean PDFs)
    decoded_objects = decode(img_cv)
    
    # Try 2: Detect on pre-processed image (works for card scans/photos)
    if not decoded_objects:
        processed = preprocess_for_qr(img_cv)
        decoded_objects = decode(processed)
        
    if decoded_objects:
        obj = decoded_objects[0]
        x, y, w, h = obj.rect
        
        # Add 10% padding for a clean crop
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
        qr_crop = img_cv[max(0, y-pad_h):y+h+pad_h, max(0, x-pad_w):x+w+pad_w]
        
        # Final formatting: grayscale and resize
        qr_pil = Image.fromarray(cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY))
        return qr_pil.resize((300, 300), Image.Resampling.LANCZOS)

    return None

def crop_qr_with_fallback(full_page):
    """
    Attempts detection on the whole page first, then zoomed regions.
    """
    # 1. Try full page detection (Best for PDFs)
    qr = detect_qr_robust(full_page)
    if qr: return qr

    # 2. If failed, the QR might be too small to detect on a large canvas.
    # Split the image into 4 quadrants and try each (Handles high-res scans)
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

    # 3. Final Fallback: Manual static crop (as a last resort)
    # Adjusted to focus on typical ID placements
    return detect_qr_robust(full_page.crop((int(W*0.8), 0, W, int(H*0.10))))

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

    issue_greg_pil = process_strip(issue_greg_crop, target_w=22, target_h=155)
    issue_eth_pil  = process_strip(issue_eth_crop,  target_w=22, target_h=155)

    return {
        "fin_image":        img_to_datauri(fin_pil,        "PNG"),
        "expiry_image":     img_to_datauri(expiry_pil,     "PNG"),
        "issue_greg_image": img_to_datauri(issue_greg_pil, "PNG"),
        "issue_eth_image":  img_to_datauri(issue_eth_pil,  "PNG"),
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

    # Expiry date row
    exp_crop = ratio_crop(EXP_V1_X0, EXP_V1_Y0, EXP_V1_X1, EXP_V1_Y1)
    exp_pil  = remove_light_background(exp_crop.convert("RGBA"), darkness_threshold=150)

    # Issue date row
    issue_crop = ratio_crop(ISSUE_V1_X0, ISSUE_V1_Y0, ISSUE_V1_X1, ISSUE_V1_Y1)
    issue_pil  = remove_light_background(issue_crop.convert("RGBA"), darkness_threshold=150)

    return {
        "fin_image":        img_to_datauri(fin_pil,   "PNG"),
        "expiry_image":     img_to_datauri(exp_pil,   "PNG"),
        "issue_greg_image": img_to_datauri(issue_pil, "PNG"),
        "issue_eth_image":  img_to_datauri(issue_pil, "PNG"),
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
        r'First,\s*Middle,\s*Surname[^\n]*\n[^\n]+\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
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
        r'First,\s*Middle,\s*Surname[^\n]*\n([\u1200-\u137F]+(?:\s+[\u1200-\u137F]+)+)',
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
    m = re.search(r'(?:Male|Female)\s+([A-Z][A-Za-z\s]+?Zone\b)', raw_text)
    if m: zone_en = clean(m.group(1))

    # Amharic: standalone Ethiopic line immediately after Zone/City Admin label
    zone_am = ""
    m = re.search(r'Zone/City Admin[^\n]*\n([\u1200-\u137F\s]+?)(?:\n|$)', raw_text)
    if m: zone_am = clean(m.group(1))

    # ── Nationality ───────────────────────────────────────────────────────────
    # Line pattern: "Ethiopia <woreda_am>" and "Ethiopian <woreda_en>"
    nat_am = ""; woreda_am = ""
    m = re.search(r'^(Ethiopia|ኢትዮጵያ)\s+([\u1200-\u137F\s]+)$', raw_text, re.MULTILINE)
    if m:
        nat_am    = clean(m.group(1))
        woreda_am = clean(m.group(2))
    else:
        # nat alone
        m2 = re.search(r'^(ኢትዮጵያ(?:ዊ|ዊት)?)$', raw_text, re.MULTILINE)
        if m2: nat_am = clean(m2.group(1))

    nat_en = ""; woreda_en = ""
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

    return {
        "fcn": fcn, "name_am": name_am, "name_en": name_en,
        "dob_greg": dob_greg, "dob_eth": dob_eth,
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
    nat_en    = clean(m6.group(1)) if m6 else clean(gl(6))
    woreda_en = clean(m6.group(2)) if m6 else ""

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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_all(pdf_bytes):
    """
    ONE render pass for the whole page.
    Automatically detects Format v1 (landscape) or v2 (portrait).
    Returns (data_dict, photo_uri, photo_warm_uri, qr_uri, right_card_uri).
    """
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
    if is_v1:
        ef    = extract_fin_expiry_v1(full_page)
        color = crop_photo_v1(full_page)
    else:
        ef    = extract_fin_expiry_from_rendered(img_cv)
        color = crop_photo(full_page)

    nobg           = remove_background(color)
    photo_uri      = img_to_datauri(nobg,                  "PNG")
    photo_warm_uri = img_to_datauri(apply_warm_tone(nobg), "PNG")
    qr_uri         = img_to_datauri(crop_qr_with_fallback(full_page), "PNG", is_qr=True)
    right_card_uri = img_to_datauri(get_right_card_image(full_page),  "PNG")

    data = {
        "fcn":        fcn,
        "name_am":    f["name_am"],
        "name_en":    f["name_en"],
        "dob_greg":   f["dob_greg"],
        "dob_eth":    f["dob_eth"],
        "sex_am":     f["sex_am"],
        "sex_en":     f["sex_en"],
        "nat_am":     f["nat_am"],
        "nat_en":     f["nat_en"],
        "phone":      f["phone"],
        "reg_am":     f["reg_am"],
        "reg_en":     f["reg_en"],
        "zone_am":    f["zone_am"],
        "zone_en":    f["zone_en"],
        "woreda_am":  f["woreda_am"],
        "woreda_en":  f["woreda_en"],
        "sn":         f["sn"],
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