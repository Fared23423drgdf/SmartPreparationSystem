import os
import time
from datetime import datetime, time as dt_time
from flask import Flask, render_template, Response, request, redirect, url_for, session, send_from_directory, jsonify
import cv2
from cv2_enumerate_cameras import enumerate_cameras

from face_recognition import face_utils
from utils import attendance

# Hardcoded admin password
ADMIN_PASSWORD = "admin123"

app = Flask(__name__, static_folder='static')
app.secret_key = 'change_this'

# Global face data
known_encodings, known_names = face_utils.load_known_faces()
last_unknown_alert = 0
camera_index = 0
camera = None

ALLOWED_HOURS = (dt_time(6, 30), dt_time(15, 0))


def allowed_time():
    now = datetime.now().time()
    return ALLOWED_HOURS[0] <= now <= ALLOWED_HOURS[1]


def get_camera_list():
    cams = list(enumerate_cameras())
    return cams


@app.before_request
def require_login():
    if request.endpoint not in ('login', 'do_login', 'processed_file') and not session.get('logged_in'):
        return redirect(url_for('login'))


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def do_login():
    if request.form.get('password') == ADMIN_PASSWORD:
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return render_template('login.html', error='Invalid password')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def dashboard():
    cams = get_camera_list()
    stats = attendance.get_stats()
    return render_template('dashboard.html', cameras=cams, stats=stats)


def generate_frames():
    global last_unknown_alert, camera
    camera = cv2.VideoCapture(camera_index)
    while True:
        success, frame = camera.read()
        if not success:
            break
        locations, names = face_utils.recognize_faces(frame, known_encodings, known_names)
        frame = face_utils.draw_overlays(frame, locations, names)
        if 'Unknown' in names:
            last_unknown_alert = time.time()
        for (top, right, bottom, left), name in zip(locations, names):
            snap = frame[top:bottom, left:right]
            if name == 'Unknown':
                folder = attendance.UNKNOWN_DIR
                status = 'Unknown - Logged'
            else:
                folder = attendance.SNAPSHOT_DIR
                status = 'Present'
            if allowed_time():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(folder, f"{name}_{timestamp}.jpg")
                cv2.imwrite(path, snap)
                attendance.log_entry(name, path, status)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_camera', methods=['POST'])
def set_camera():
    global camera_index, camera
    camera_index = int(request.form.get('index', 0))
    if camera:
        camera.release()
        camera = None
    return redirect(url_for('dashboard'))


@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    global known_encodings, known_names
    if request.method == 'POST':
        name = request.form.get('name')
        f = request.files.get('image')
        if name and f:
            os.makedirs(attendance.KNOWN_DIR, exist_ok=True)
            ext = os.path.splitext(f.filename)[1]
            filename = f"{name}_{int(time.time())}{ext}"
            path = os.path.join(attendance.KNOWN_DIR, filename)
            f.save(path)
            known_encodings, known_names = face_utils.load_known_faces(attendance.KNOWN_DIR)
            return redirect(url_for('dashboard'))
    return render_template('add_student.html')


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory('data/processed', filename)


@app.route('/unknown_alert')
def unknown_alert():
    alert = time.time() - last_unknown_alert < 5
    return jsonify({'alert': alert})


if __name__ == '__main__':
    attendance.ensure_dirs()
    app.run(host='0.0.0.0', port=8000, debug=False)
