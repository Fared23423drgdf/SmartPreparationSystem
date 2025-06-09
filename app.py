import os
from flask import Flask, render_template, send_from_directory, Response
from flask_socketio import SocketIO, emit

from face_recognition.camera import Camera
from face_recognition import face_utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

KNOWN_ENCODINGS, KNOWN_NAMES = face_utils.load_known_faces()
camera = Camera()


@app.route('/')
def index():
    return render_template('Face_recognition_demo.html')


@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(os.path.join('data', 'processed'), filename)


@socketio.on('upload_image')
def handle_image_upload(data):
    file_data = data.get('file')
    filename = data.get('filename')
    out_name, summary = face_utils.process_uploaded_image(
        file_data, filename, KNOWN_ENCODINGS, KNOWN_NAMES
    )
    emit('image_processed', {
        'url': f'/processed/{out_name}',
        'summary': summary,
        'filename': filename,
    })


@app.route('/video_feed')
def video_feed():
    return Response(
        camera.generate_frames(KNOWN_ENCODINGS, KNOWN_NAMES),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

