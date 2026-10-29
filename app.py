from flask import Flask, render_template, render_template_string, request, jsonify
from datetime import datetime
import os
import csv
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Face detection model could not be loaded.")
else:
    print("SUCCESS: Face detection model loaded.")

# Attendance CSV file
CSV_FILE = "attendance.csv"


# =========================
# HOMEPAGE
# =========================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Attendance & Engagement Agent</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f0f4f8;
            text-align: center;
            padding: 30px;
            margin: 0;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 700px;
            margin: 40px auto;
            box-shadow: 0 4px 15px #ccc;
        }

        h1 {
            color: #2346a0;
        }

        button, .link-button {
            background: #2346a0;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin: 8px;
            font-size: 15px;
        }

        button:hover, .link-button:hover {
            background: #183477;
        }

        .status {
            background: #e0f7e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }

        #test-result {
            color: #166534;
            font-weight: bold;
            margin-top: 15px;
        }
    </style>
</head>

<body>
    <div class="container">

        <h1>Multimodal Attendance & Engagement Agent</h1>

        <p>
            AI-powered classroom attendance and engagement monitoring.
        </p>

        <div class="status">
            <h3>System Status: Running</h3>
            <p>Flask application is running.</p>
        </div>

        <br>

        <button
            type="button"
            onclick="document.getElementById('test-result').textContent =
            'System is running successfully!'">
            Test System
        </button>

        <p id="test-result" aria-live="polite"></p>

        <br>

        <a class="link-button" href="/camera">Open Camera</a>

        <a class="link-button" href="/attendance">
            View Attendance
        </a>

    </div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


# =========================
# CAMERA PAGE
# =========================

@app.route("/camera")
def camera():
    return render_template("camera.html")


# =========================
# ATTENDANCE DASHBOARD
# =========================

@app.route("/attendance")
def attendance():
    records = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            records = list(csv.reader(file))

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>Attendance Dashboard</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                padding: 30px;
            }

            h1, h2 {
                color: #1e293b;
            }

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

            th {
                background: #2563eb;
                color: white;
            }

            a {
                color: #2563eb;
            }

            .table-container {
                overflow-x: auto;
            }
        </style>
    </head>

    <body>
        <h1>Multimodal Attendance Dashboard</h1>

        <div class="card">
            <h2>Total Attendance Records: {{ count }}</h2>
            <p>System Status: Running</p>
            <p>Face Detection: Enabled</p>
            <p>
                Engagement Score: Demo score available on the camera page.
            </p>
        </div>

        <h2>Attendance History</h2>

        <div class="table-container">
            <table>
                <tr>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Faces Detected</th>
                    <th>Status</th>
                </tr>

                {% for row in records %}
                    {% if not loop.first %}
                    <tr>
                        <td>{{ row[0] if row|length > 0 else '' }}</td>
                        <td>{{ row[1] if row|length > 1 else '' }}</td>
                        <td>{{ row[2] if row|length > 2 else '' }}</td>
                        <td>{{ row[3] if row|length > 3 else '' }}</td>
                    </tr>
                    {% endif %}
                {% endfor %}
            </table>
        </div>

        <br>

        <a href="/">Back to Home</a>
        <br><br>
        <a href="/camera">Back to Camera</a>
    </body>
    </html>
    """, records=records, count=max(len(records) - 1, 0))


# =========================
# FACE DETECTION
# =========================

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "message": "No image received."
            }), 400

        image_data = data["image"].split(",", 1)[-1]
        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Invalid image."
            }), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces_count = len(faces)
        now = datetime.now()

        # Demo score only; this is not real engagement analysis.
        engagement_score = 85 if faces_count > 0 else 0

        file_exists = (
            os.path.exists(CSV_FILE)
            and os.path.getsize(CSV_FILE) > 0
        )

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Date",
                    "Time",
                    "Faces Detected",
                    "Status"
                ])

            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                faces_count,
                "Present" if faces_count > 0 else "Absent"
            ])

        return jsonify({
            "success": True,
            "faces_detected": faces_count,
            "engagement_score": engagement_score,
            "message": (
                f"Detected {faces_count} face(s). Attendance saved!"
            )
        })

    except Exception as e:
        app.logger.exception("Face detection failed")

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# =========================
# START APPLICATION
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )