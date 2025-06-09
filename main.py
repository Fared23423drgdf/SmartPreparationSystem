import os
import threading
import time
import webbrowser
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from cv2_enumerate_cameras import enumerate_cameras
import face_recognition

from face_recognition import face_utils
from utils import attendance


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نظام الحضور والتعرف على الوجه")
        attendance.ensure_dirs()

        self.cameras = list(enumerate_cameras())
        names = [cam.name for cam in self.cameras]
        self.camera_index = self.cameras[0].index if self.cameras else 0

        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="اختر الكاميرا:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value=names[0] if names else '')
        self.camera_combo = ttk.Combobox(top_frame, values=names, textvariable=self.camera_var, state='readonly')
        self.camera_combo.pack(side=tk.LEFT)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_selected)

        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=10)
        self.start_btn = tk.Button(ctrl_frame, text="بدء النظام", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(ctrl_frame, text="إيقاف النظام", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.dash_btn = tk.Button(ctrl_frame, text="فتح لوحة التحكم", command=self.open_dashboard)
        self.dash_btn.pack(side=tk.LEFT, padx=5)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        manage_frame = tk.Frame(root)
        manage_frame.pack(pady=10)
        tk.Label(manage_frame, text="الوجوه المعروفة:").pack()
        self.tree = ttk.Treeview(manage_frame, columns=('name',), show='headings')
        self.tree.heading('name', text='الاسم')
        self.tree.pack()

        btns = tk.Frame(manage_frame)
        btns.pack()
        tk.Button(btns, text="إضافة وجه جديد", command=self.add_face).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="تعديل الاسم", command=self.rename_face).pack(side=tk.LEFT, padx=3)
        tk.Button(btns, text="حذف الوجه", command=self.delete_face).pack(side=tk.LEFT, padx=3)

        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var).pack(fill=tk.X, side=tk.BOTTOM)

        self.cap = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.process_next = True
        self.last_unknown = 0
        self.dashboard_opened = False

        self.load_known_faces()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    # ------------------ Camera and capture ------------------
    def on_camera_selected(self, event=None):
        idx = self.camera_combo.current()
        if idx >= 0:
            self.camera_index = self.cameras[idx].index
            self.status_var.set(f"تم اختيار الكاميرا: {self.cameras[idx].name}")

    def start_system(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("خطأ", "تعذر فتح الكاميرا")
            self.cap = None
            return
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_system(self):
        if self.cap is None:
            return
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join()
        self.cap.release()
        self.cap = None
        cv2.destroyAllWindows()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("تم إيقاف النظام")

    def capture_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            display = frame.copy()
            if self.process_next:
                self.detect_faces(frame, display)
            self.process_next = not self.process_next
            cv2.imshow("Live Feed", display)
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop_event.set()
                break
        self.stop_system()

    # ------------------ Face recognition and logging ------------------
    def detect_faces(self, frame, display):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        names = []
        for enc in encodings:
            name = "Unknown"
            if self.known_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, enc)
                dist = face_recognition.face_distance(self.known_encodings, enc)
                if len(dist) > 0:
                    best = int(np.argmin(dist))
                    if matches[best]:
                        name = self.known_names[best]
            names.append(name)

        for (top, right, bottom, left), name in zip(locations, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(display, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            snap = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if name == "Unknown":
                if time.time() - self.last_unknown < 5:
                    continue
                self.last_unknown = time.time()
                folder = attendance.UNKNOWN_DIR
                status = "Unknown - Logged"
            else:
                if attendance.was_logged_today(name):
                    continue
                folder = attendance.SNAPSHOT_DIR
                status = "Present"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{name}_{timestamp}.jpg")
            cv2.imwrite(path, snap)
            attendance.log_entry(name, path, status)
            if not self.dashboard_opened:
                self.open_dashboard()
                self.dashboard_opened = True

    # ------------------ Known faces management ------------------
    def load_known_faces(self):
        self.known_encodings, self.known_names = face_utils.load_known_faces(attendance.KNOWN_DIR)
        self.tree.delete(*self.tree.get_children())
        for name in sorted(set(self.known_names)):
            self.tree.insert('', 'end', values=(name,))

    def add_face(self):
        files = filedialog.askopenfilenames(filetypes=[('Images', '*.jpg *.jpeg *.png')])
        if not files:
            return
        name = simpledialog.askstring('الاسم', 'أدخل اسم الشخص:')
        if not name:
            return
        for f in files:
            ext = os.path.splitext(f)[1]
            dest = os.path.join(attendance.KNOWN_DIR, f"{name}_{int(time.time())}{ext}")
            try:
                os.makedirs(attendance.KNOWN_DIR, exist_ok=True)
                with open(f, 'rb') as src, open(dest, 'wb') as dst:
                    dst.write(src.read())
            except Exception:
                pass
        self.load_known_faces()

    def rename_face(self):
        sel = self.tree.focus()
        if not sel:
            return
        old = self.tree.item(sel)['values'][0]
        new = simpledialog.askstring('تعديل الاسم', 'الاسم الجديد:', initialvalue=old)
        if not new or new == old:
            return
        for fname in os.listdir(attendance.KNOWN_DIR):
            if fname.startswith(old):
                ext = os.path.splitext(fname)[1]
                os.rename(
                    os.path.join(attendance.KNOWN_DIR, fname),
                    os.path.join(attendance.KNOWN_DIR, f"{new}_{int(time.time())}{ext}")
                )
        self.load_known_faces()

    def delete_face(self):
        sel = self.tree.focus()
        if not sel:
            return
        name = self.tree.item(sel)['values'][0]
        if not messagebox.askyesno('حذف', f'حذف جميع صور {name}?'):
            return
        for fname in os.listdir(attendance.KNOWN_DIR):
            if fname.startswith(name):
                os.remove(os.path.join(attendance.KNOWN_DIR, fname))
        self.load_known_faces()

    # ------------------ Dashboard ------------------
    def open_dashboard(self):
        path = attendance.generate_dashboard_html()
        webbrowser.open('file://' + os.path.abspath(path))

    # ------------------ Closing ------------------
    def on_close(self):
        if messagebox.askokcancel('خروج', 'هل تريد إغلاق النظام؟'):
            self.stop_system()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
