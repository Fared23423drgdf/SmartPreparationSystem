import os
import json
from datetime import datetime

DATA_DIR = "data"
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.json")
CAPTURED_DIR = os.path.join(DATA_DIR, "captured_faces")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown_faces")


def ensure_dirs():
    os.makedirs(CAPTURED_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)


def log_entry(name, snapshot_path, status):
    ensure_dirs()
    record = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "time_arrival": datetime.now().strftime("%H:%M:%S"),
        "status": status,
        "snapshot_path": snapshot_path,
    }
    data = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(record)
    with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
