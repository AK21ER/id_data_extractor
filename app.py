import sys
import os
import webbrowser
import threading
import time
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import traceback
from main import process_screenshots
from pdf_to_card import extract_all

# ── Resource path (works for PyInstaller and dev) ───────────────
def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

app = Flask(
    __name__,
    static_folder=get_resource_path('static'),
    template_folder=get_resource_path('templates')
)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SECRET_KEY'] = 'ethiopian-id-processor-secret'

# ── Helpers ─────────────────────────────────────────────
def file_to_img(file_obj):
    file_obj.seek(0)
    nparr = np.frombuffer(file_obj.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    file_obj.seek(0)
    return img

def to_base64(encoded):
    if encoded is None:
        return None
    return base64.b64encode(encoded).decode('utf-8')

def get_orig_b64(file_obj):
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode('utf-8')

def safe_get(data, key, default="—"):
    if not isinstance(data, dict):
        return default
    val = data.get(key)
    return val if val and str(val).strip() else default

# ── Browser opener ─────────────────────────────────────
def open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open('http://127.0.0.1:5000')
    except:
        pass

# ── Routes ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdf-processor')
def pdf_processor():
    return render_template('pdf_processor.html')

@app.route('/multi', methods=['POST'])
def multi():
    try:
        if 'pdf' not in request.files:
            return render_template('pdf_processor.html', error="No PDF files uploaded.")

        files = request.files.getlist('pdf')
        valid_files = [f for f in files if f.filename.lower().endswith('.pdf')]

        if not valid_files:
            return render_template('pdf_processor.html', error="Please select valid PDF files.")

        cards = []
        for pdf_file in valid_files[:100]:
            try:
                pdf_bytes = pdf_file.read()
                result = extract_all(pdf_bytes)
                if not result or len(result) < 5:
                    continue
                data, photo, photo_warm, qr, _ = result
                cards.append({
                    "photo": photo or "",
                    "photo_warm": photo_warm or "",
                    "qr": qr or "",
                    "data": data or {}
                })
            except Exception as e:
                print(f"⚠️ PDF processing error for {pdf_file.filename}: {e}")
                continue

        if not cards:
            return render_template('pdf_processor.html', error="No valid ID cards could be extracted from the PDFs.")

        return render_template('id_cards_multi.html', cards=cards, count=len(cards))

    except Exception as e:
        traceback.print_exc()
        return render_template('pdf_processor.html', error=f"Server error: {str(e)}")

@app.route('/process', methods=['POST'])
def process():
    try:
        cards = []
        layout = request.form.get('layout', 'document')

        for c in range(1, 6):
            front_key = f'card{c}-front'
            back_key = f'card{c}-back'
            qr_key = f'card{c}-qr'

            if front_key not in request.files or back_key not in request.files or qr_key not in request.files:
                continue

            front_file = request.files[front_key]
            back_file = request.files[back_key]
            qr_file = request.files[qr_key]

            if not front_file.filename or not back_file.filename or not qr_file.filename:
                continue

            front_img = file_to_img(front_file)
            back_img = file_to_img(back_file)
            qr_img = file_to_img(qr_file)

            if front_img is None or back_img is None or qr_img is None:
                continue

            try:
                data = process_screenshots(front_img, back_img, qr_img)
            except Exception as e:
                print(f"Card {c} processing failed: {e}")
                continue

            actual_id = "—"
            for k in ["id_number", "fan", "barcode_data"]:
                val = data.get(k)
                if val and val != "—" and len(str(val)) >= 10:
                    actual_id = val
                    break

            portrait_b64 = to_base64(data.get("portrait_encoded"))
            qr_b64 = to_base64(data.get("qr_code_encoded"))
            name_am_b64 = to_base64(data.get("name_am_encoded"))
            issue_greg_b64 = to_base64(data.get("issue_greg_encoded"))
            issue_eth_b64 = to_base64(data.get("issue_eth_encoded"))
            front_orig_b64 = get_orig_b64(front_file)
            back_orig_b64 = get_orig_b64(back_file)

            cards.append({
                "photo": f"data:image/png;base64,{portrait_b64}" if portrait_b64 else "",
                "qr": f"data:image/png;base64,{qr_b64}" if qr_b64 else "",
                "front_orig": f"data:image/jpeg;base64,{front_orig_b64}",
                "back_orig": f"data:image/jpeg;base64,{back_orig_b64}",
                "data": {
                    "name_en": safe_get(data, "name_en", ""),
                    "name_am": safe_get(data, "name_am"),
                    "name_am_image": f"data:image/png;base64,{name_am_b64}" if name_am_b64 else "",
                    "dob_greg": safe_get(data, "dob_greg", ""),
                    "dob_eth": safe_get(data, "dob_eth"),
                    "sex_en": safe_get(data, "sex_en", ""),
                    "sex_am": safe_get(data, "sex_am"),
                    "expiry_greg": safe_get(data, "expiry_greg"),
                    "expiry_eth": safe_get(data, "expiry_eth"),
                    "issue_greg": safe_get(data, "issue_greg"),
                    "issue_eth": safe_get(data, "issue_eth"),
                    "fcn": actual_id,
                    "fin": safe_get(data, "fin"),
                    "barcode_data": actual_id,
                    "phone": safe_get(data, "phone"),
                    "nat_en": "Ethiopian",
                    "nat_am": "ኢትዮጵያዊ",
                    "reg_en": safe_get(data, "reg_en"),
                    "reg_am": safe_get(data, "reg_am"),
                    "zone_en": safe_get(data, "zone_en"),
                    "zone_am": safe_get(data, "zone_am"),
                    "woreda_en": safe_get(data, "woreda_en"),
                    "woreda_am": safe_get(data, "woreda_am"),
                    "fin_image": "",
                    "expiry_image": "",
                    "issue_greg_image": f"data:image/png;base64,{issue_greg_b64}" if issue_greg_b64 else "",
                    "issue_eth_image": f"data:image/png;base64,{issue_eth_b64}" if issue_eth_b64 else ""
                }
            })

        if not cards:
            return jsonify({"error": "No valid cards uploaded."}), 400

        template_name = 'official_doc.html' if layout == 'document' else 'result.html'
        return render_template(template_name, cards=cards)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok", "version": "1.0"})

# ── Error handlers ─────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error="Internal server error."), 500


# ── Main ─────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(get_resource_path('static'), exist_ok=True)
    os.makedirs(get_resource_path('templates'), exist_ok=True)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)