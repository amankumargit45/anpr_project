from flask import Flask, request, render_template, redirect
from ultralytics import YOLO
import cv2
import easyocr
import os
import pandas as pd

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# Load OCR
reader = easyocr.Reader(['en'], gpu=False)

# -------------------------------
# Video Processing
# -------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    detected_plates = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        

        results = model(frame)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = int(box.cls[0])

                    # Class 0 assumed as license plate
                    if label == 0:
                        plate_img = frame[y1:y2, x1:x2]

                        if plate_img.size == 0:
                            continue

                        ocr_results = reader.readtext(plate_img)

                        for (_, text, prob) in ocr_results:
                            if prob > 0.5:
                                clean_text = text.replace(" ", "").upper()
                                detected_plates.add(clean_text)

                except Exception as e:
                    print("⚠️ Error:", e)
                    continue

    cap.release()

    # Save detected plates
    csv_path = os.path.join(OUTPUT_FOLDER, 'License_plate.csv')
    df = pd.DataFrame({'license_plate_numbers': list(detected_plates)})
    df.to_csv(csv_path, index=False)

    return detected_plates

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']

    if file.filename == '':
        return redirect('/')

    # Save file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process video
    plates = process_video(file_path)

    # Debug prints (IMPORTANT)
    print("Detected Plates:", plates)

    # -------------------------------
    # Load stolen vehicles CSV
    # -------------------------------
    try:
        stolen_record = pd.read_csv('stolen_vehicles.csv')

        if 'lplate' not in stolen_record.columns:
            return "❌ Column 'lplate' not found in stolen_vehicles.csv"

        stolen_list = [
            str(x).replace(" ", "").upper()
            for x in stolen_record['lplate']
        ]

    except Exception as e:
        return f"❌ CSV Error: {e}"

    print("Stolen Plates:", stolen_list)

    # -------------------------------
    # Match plates
    # -------------------------------
    matched = set(plates).intersection(set(stolen_list))

    print("Matched Plates:", matched)

    # -------------------------------
    # Send result
    # -------------------------------
   


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)