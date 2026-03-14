from flask import Flask, request, jsonify, render_template, send_from_directory, render_template_string
import cv2
import numpy as np
import base64
import os
from main import process_screenshots
from jinja2 import Template

app = Flask(__name__, static_folder='static', template_folder='.')

@app.route('/')
def index():
    return render_template('static/index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'front' not in request.files or 'back' not in request.files or 'qr' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        front_file = request.files['front']
        back_file = request.files['back']
        qr_file = request.files['qr']

        # Convert files to OpenCV images
        def file_to_img(file):
            nparr = np.frombuffer(file.read(), np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        front_data = file_to_img(front_file)
        back_data = file_to_img(back_file)
        qr_data = file_to_img(qr_file)

        if front_data is None or back_data is None or qr_data is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Process images
        data = process_screenshots(front_data, back_data, qr_data)

        # Helper to convert encoded image to base64
        def to_base64(encoded):
            if encoded is None: return None
            # encoded is already a numpy array from cv2.imencode
            return base64.b64encode(encoded).decode('utf-8')

        actual_id = "—"
        for k in ["id_number", "fan", "barcode_data"]:
            val = data.get(k)
            if val and val != "—" and len(val) >= 10:
                actual_id = val
                break

        # Prepare cards for rendering
        portrait_b64 = to_base64(data.get("portrait_encoded"))
        qr_b64 = to_base64(data.get("qr_code_encoded"))
        
        cards = [{
            "photo": f"data:image/png;base64,{portrait_b64}" if portrait_b64 else "",
            "photo_warm": f"data:image/png;base64,{portrait_b64}" if portrait_b64 else "",
            "qr": f"data:image/png;base64,{qr_b64}" if qr_b64 else "",
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

        layout = request.form.get('layout', 'document') # Default to document as requested
        template_name = 'official_doc.html' if layout == 'document' else 'result.html'
        
        # Render the chosen template
        return render_template(template_name, cards=cards)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5000)
