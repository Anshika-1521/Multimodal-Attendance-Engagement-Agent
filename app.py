from flask import Flask, render_template, render_template_string, request, jsonify
from datetime import datetime
import os
import csv
import cv2
import numpy as np
import base64
app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face detection model could not be loaded.")
else:
    print("SUCCESS: Face detection model loaded.")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Attendance & Engagement Agent</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f0f4f8;
            text-align: center;
            padding: 50px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 4px 15px #ccc;
        }
        h1 { color: #2346a0; }
        button {
            background: #2346a0;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .status {
            background: #e0f7e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multimodal Attendance & Engagement Agent</h1>
        <p>AI-powered classroom attendance and engagement monitoring.</p>
        <div class="status">
            <h3>System Status: Running</h3>
            <p>Flask application is working successfully!</p>
        </div>
        <br>
            Test System
        </button>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)
@app.route('/camera')
def camera():
    return render_template('camera.html')
@app.route('/attendance')
def attendance():
    records = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", newline="") as file:
            records = list(csv.reader(file))
    
        return render_template_string("""
        <html>
        <head>
        <title>Attendance Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                padding: 30px;
            }
            h1 { color: #1e293b; }
            .card {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                box-shadow: 0 2px 8px #ddd;
            }
            table {
                width: 100%;
                background: white;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th { background: #2563eb; color: white; }
            a { color: #2563eb; }
        </style>
    </head>
    <body>
        <h1>Multimodal Attendance Dashboard</h1>

        <div class="card">
            <h2>Total Attendance Records: {{ records|length }}</h2>
            <p>System Status: Running</p>
            <p>Face Detection: Enabled</p>
            <p>Engagement Score: Demo score available on camera page</p>
        </div>

        <h2>Attendance History</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Time</th>
                <th>Faces Detected</th>
                <th>Status</th>
            </tr>
            {% for row in records %}
            <tr>
                <td>{{ row[0] }}</td>
                <td>{{ row[1] }}</td>
                <td>{{ row[2] }}</td>
                <td>{{ row[3] }}</td>
            </tr>
            {% endfor %}
        </table>

        <br>
        <a href="/camera">Back to Camera</a>
    </body>
    </html>
    """, records=records)

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "message": "No image received"
            }), 400

        image_data = data['image'].split(',', 1)[-1]
        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Invalid image"
            }), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        engagement_score = 85 if len(faces) > 0 else 0
        if len(faces) > 0:
            with open("attendance.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                     datetime.now().strftime("%Y-%m-%d"),
                       datetime.now().strftime("%H:%M:%S"),
                       len(faces),
                       "Present"
                       ])
                now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        file_exists = os.path.exists("attendance.csv")

        with open("attendance.csv", "a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Date", "Time", "Faces Detected", "Status"])

            writer.writerow([date, time, len(faces), "Present" if len(faces) > 0 else "Absent"])

        return jsonify({'engagement_score': engagement_score,
            "success": True,
            "faces_detected": len(faces),
            "message": f"Detected {len(faces)} face(s). Attendance saved!"
        })
        return jsonify({
            "success": True,
            "faces_detected": len(faces),
            "message": f"Detected {len(faces)} face(s)"
        })
    except Exception as e:
        app.logger.exception("Face detection failed")
        return jsonify({
             "success": False,
             "message": str(e)
               }), 500
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)