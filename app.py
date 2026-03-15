from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
from main import process_screenshots  # Your existing processing function
from pdf_to_card import extract_all # PDF conversion logic

app = Flask(__name__, static_folder='static', template_folder='.')

def file_to_img(file):
    """Convert uploaded file to OpenCV image."""
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    file.seek(0)
    return img

def to_base64(encoded):
    """Convert OpenCV encoded image (numpy array) to base64 string."""
    if encoded is None:
        return None
    return base64.b64encode(encoded).decode('utf-8')

def get_orig_b64(file_obj):
    """Return base64 of original uploaded file."""
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('static/index.html')

@app.route('/pdf-processor')
def pdf_processor():
    """Serve the PDF upload page (index.html in root)."""
    return render_template('index.html')

@app.route('/multi', methods=['POST'])
def multi():
    """Process multiple PDF files and return id_cards_multi.html."""
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('pdf')
        cards = []

        for pdf_file in files:
            if pdf_file.filename == '':
                continue
            
            pdf_bytes = pdf_file.read()
            # extract_all returns (data, photo_uri, photo_warm_uri, qr_uri, right_card_uri)
            data, photo, photo_warm, qr, _ = extract_all(pdf_bytes)
            
            cards.append({
                "photo": photo,
                "photo_warm": photo_warm,
                "qr": qr,
                "data": data
            })

        if not cards:
            return render_template('index.html', error="No valid PDF ID cards were processed.")

        return render_template('id_cards_multi.html', cards=cards)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Conversion failed: {str(e)}")


@app.route('/process', methods=['POST'])
def process():
    try:
        cards = []

        for c in range(1, 6):  # Card 1 to Card 5
            front_key = f'card{c}-front'
            back_key = f'card{c}-back'
            qr_key = f'card{c}-qr'

            # Check if files exist
            if front_key not in request.files or back_key not in request.files or qr_key not in request.files:
                continue

            front_file = request.files[front_key]
            back_file = request.files[back_key]
            qr_file = request.files[qr_key]

            # Skip empty uploads
            if front_file.filename == "" or back_file.filename == "" or qr_file.filename == "":
                continue

            # Convert uploaded files to OpenCV images
            front_data = file_to_img(front_file)
            back_data = file_to_img(back_file)
            qr_data = file_to_img(qr_file)

            # Skip invalid images
            if front_data is None or back_data is None or qr_data is None:
                continue

            try:
                # Process images
                data = process_screenshots(front_data, back_data, qr_data)
            except Exception as e:
                print(f"Card {c} processing failed: {e}")
                continue

            # Determine actual ID
            actual_id = "—"
            for k in ["id_number", "fan", "barcode_data"]:
                val = data.get(k)
                if val and val != "—" and len(val) >= 10:
                    actual_id = val
                    break

            # Encode images
            portrait_b64 = to_base64(data.get("portrait_encoded"))
            qr_b64 = to_base64(data.get("qr_code_encoded"))
            issue_greg_b64 = to_base64(data.get("issue_greg_encoded"))
            issue_eth_b64 = to_base64(data.get("issue_eth_encoded"))
            front_orig_b64 = get_orig_b64(front_file)
            back_orig_b64 = get_orig_b64(back_file)

            card_dict = {
                "photo": f"data:image/png;base64,{portrait_b64}" if portrait_b64 else "",
                "qr": f"data:image/png;base64,{qr_b64}" if qr_b64 else "",
                "front_orig": f"data:image/jpeg;base64,{front_orig_b64}",
                "back_orig": f"data:image/jpeg;base64,{back_orig_b64}",
                "data": {
                    "name_en": data.get("name_en", ""),
                    "name_am": data.get("name_am", "—"),
                    "dob_greg": data.get("dob_greg", ""),
                    "dob_eth": data.get("dob_eth", "—"),
                    "sex_en": data.get("sex_en", ""),
                    "sex_am": data.get("sex_am", "—"),
                     "expiry_greg": data.get("expiry_greg") or data.get("expiry") or data.get("expiry_eth") or "—",
                    "expiry_eth": data.get("expiry_eth", "—"),
                    "issue_greg": data.get("issue_greg", "—"),
                    "issue_eth": data.get("issue_eth", "—"),
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
                    "expiry_image": "",
                    "issue_greg_image": f"data:image/png;base64,{issue_greg_b64}" if issue_greg_b64 else "",
                    "issue_eth_image": f"data:image/png;base64,{issue_eth_b64}" if issue_eth_b64 else ""
                }
            }

            cards.append(card_dict)

        if not cards:
            return jsonify({"error": "No valid cards uploaded"}), 400

        layout = request.form.get('layout', 'document')
        template_name = 'official_doc.html' if layout == 'document' else 'result.html'

        return render_template(template_name, cards=cards)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5000)
