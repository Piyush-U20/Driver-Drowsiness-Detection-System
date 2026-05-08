from flask import Flask, jsonify, Response, send_file, request
from flask_cors import CORS
import cv2
import time
import threading
import numpy as np
from scipy.spatial import distance as dist
import dlib
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_START,  LEFT_END  = 42, 48
RIGHT_START, RIGHT_END = 36, 42

EAR_THRESHOLD = 0.25
WARN_FRAMES   = 8
DANGER_FRAMES = 15
OPEN_CONFIRM  = 4

camera_running   = False
state            = "safe"
latest_frame     = None
frame_lock       = threading.Lock()
cap              = None
cap_lock         = threading.Lock()
session_log_file = None


def start_session_log():
    global session_log_file
    os.makedirs("logs", exist_ok=True)
    timestamp        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_log_file = os.path.join("logs", f"session_{timestamp}.txt")
    with open(session_log_file, "w", encoding="utf-8") as f:
        f.write("Driver Drowsiness Detection - Session Log\n")
        f.write(f"Session started: {datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}\n")
        f.write("=" * 60 + "\n\n")


def write_to_log(message, level, location=""):
    if not session_log_file:
        return
    timestamp = datetime.now().strftime("%I:%M:%S %p")
    line      = f"[{timestamp}]  [{level:<6}]  {message}"
    if location:
        line += f"  |  Location: {location}"
    with open(session_log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def end_session_log():
    if not session_log_file:
        return
    with open(session_log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Session ended: {datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}\n")


def eye_aspect_ratio(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def draw_eye(frame, pts, color):
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1)


def capture_loop():
    global state, latest_frame, camera_running, cap

    closed_frames = 0
    open_frames   = 0
    last_state    = "safe"

    while True:
        if not camera_running:
            time.sleep(0.1)
            continue

        with cap_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.03)
            continue

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = detector(gray, 0)
        eyes_open = True

        for face in faces:
            shape  = predictor(gray, face)
            coords = shape_to_np(shape)

            left_eye  = coords[LEFT_START:LEFT_END]
            right_eye = coords[RIGHT_START:RIGHT_END]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear   = (left_ear + right_ear) / 2.0
            eyes_open = avg_ear >= EAR_THRESHOLD

            color = (0, 255, 0) if eyes_open else (0, 0, 255)
            draw_eye(frame, left_eye,  color)
            draw_eye(frame, right_eye, color)

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            hud = {"safe": (0,220,0), "warn": (0,200,255), "danger": (0,0,255)}.get(state, (255,255,255))
            cv2.putText(frame, f"EAR: {avg_ear:.3f}  threshold: {EAR_THRESHOLD}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 2)
            cv2.putText(frame, f"Eyes: {'OPEN' if eyes_open else 'CLOSED'}  State: {state.upper()}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud, 2)
            cv2.putText(frame, f"Closed frames: {closed_frames}/{DANGER_FRAMES}",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud, 1)
            break

        if eyes_open:
            open_frames += 1
            if open_frames >= OPEN_CONFIRM:
                if last_state != "safe":
                    write_to_log("Driver alert — eyes open, normal state", "OK")
                closed_frames = 0
                open_frames   = 0
                state         = "safe"
        else:
            closed_frames += 1
            open_frames    = 0
            if closed_frames >= DANGER_FRAMES:
                if last_state != "danger":
                    write_to_log("CRITICAL — eyes closed too long, pull over immediately", "DANGER")
                state = "danger"
            elif closed_frames >= WARN_FRAMES:
                if last_state != "warn":
                    write_to_log("WARNING — driver getting drowsy, take a break", "WARN")
                state = "warn"

        last_state = state

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)

threading.Thread(target=capture_loop, daemon=True).start()

def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/status")
def status():
    if not camera_running:
        return jsonify({"state": "stopped"})
    return jsonify({"state": state})
@app.route("/start_camera", methods=["GET", "POST"])
def start_camera():
    global camera_running, cap, state
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Could not open camera"}), 500
    state          = "safe"
    camera_running = True
    start_session_log()
    write_to_log("Camera started — monitoring session began", "OK")
    return jsonify({"status": "started", "log_file": session_log_file})

@app.route("/stop_camera", methods=["GET", "POST"])
def stop_camera():
    global camera_running, cap
    write_to_log("Camera stopped by user", "OK")
    end_session_log()
    camera_running = False
    time.sleep(0.15)
    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None
    return jsonify({"status": "stopped", "log_file": session_log_file})
@app.route("/save_log", methods=["POST"])
def save_log():
    try:
        data    = request.get_json(force=True)
        entries = data.get("entries", [])
        if not entries:
            return jsonify({"ok": False, "message": "No entries"}), 400

        if session_log_file:
            with open(session_log_file, "a", encoding="utf-8") as f:
                f.write("\n--- Frontend log snapshot ---\n")
                for e in entries:
                    ts  = e.get("time", "")[:19].replace("T", " ")
                    lvl = e.get("type", "")
                    msg = e.get("message", "")
                    loc = e.get("location", "")
                    line = f"[{ts}]  [{lvl:<6}]  {msg}"
                    if loc:
                        line += f"  |  Location: {loc}"
                    f.write(line + "\n")
            return jsonify({"ok": True, "path": session_log_file})

        return jsonify({"ok": False, "message": "No active session"}), 400

    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
@app.route("/get_log")
def get_log():
    try:
        if not session_log_file or not os.path.exists(session_log_file):
            return jsonify({"ok": True, "content": ""})
        with open(session_log_file, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"ok": True, "content": content, "path": session_log_file})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
@app.route("/")
def index():
    return send_file("driver_drowsiness_app.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)