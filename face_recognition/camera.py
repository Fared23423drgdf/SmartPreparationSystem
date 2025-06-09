import cv2
from . import face_utils

class Camera:
    """Simple camera capture class."""
    def __init__(self, index=0):
        self.index = index
        self.cap = None

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.index)

    def read_frame(self):
        if self.cap is None:
            self.start()
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def generate_frames(self, known_face_encodings, known_face_names):
        """Generator that yields processed frames for streaming."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            locations, names = face_utils.recognize_faces(
                frame, known_face_encodings, known_face_names
            )
            frame = face_utils.draw_overlays(frame, locations, names)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

