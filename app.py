import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from datetime import datetime

from face_recognition.camera import Camera
from face_recognition import face_utils
from utils import attendance


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance")

        attendance.ensure_dirs()
        self.known_encodings, self.known_names = face_utils.load_known_faces()
        self.camera = Camera()
        self.running = False

        self.video_label = tk.Label(root)
        self.video_label.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

    def start(self):
        if not self.running:
            self.camera.start()
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.update_frame()

    def stop(self):
        self.running = False
        self.camera.release()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def update_frame(self):
        if not self.running:
            return
        frame = self.camera.read_frame()
        if frame is not None:
            locations, names = face_utils.recognize_faces(frame, self.known_encodings, self.known_names)
            face_utils.draw_overlays(frame, locations, names)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for (top, right, bottom, left), name in zip(locations, names):
                face_img = frame[top:bottom, left:right]
                if name == "Unknown":
                    save_dir = attendance.UNKNOWN_DIR
                    status = "Unknown - Logged"
                else:
                    save_dir = attendance.CAPTURED_DIR
                    status = "Present"
                os.makedirs(save_dir, exist_ok=True)
                snap_path = os.path.join(save_dir, f"{name}_{timestamp}.jpg")
                cv2.imwrite(snap_path, face_img)
                attendance.log_entry(name, snap_path, status)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)


def main():
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
