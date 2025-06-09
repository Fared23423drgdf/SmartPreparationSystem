import os
import json
from datetime import datetime, date

DATA_DIR = os.path.join('data')
KNOWN_DIR = os.path.join(DATA_DIR, 'known_faces')
UNKNOWN_DIR = os.path.join(DATA_DIR, 'unknown_faces_detected')
SNAPSHOT_DIR = os.path.join(DATA_DIR, 'snapshots')
LOG_DIR = os.path.join(DATA_DIR, 'attendance_logs')
ATTENDANCE_FILE = os.path.join(LOG_DIR, 'attendance_log.json')


def ensure_dirs():
    """Create required data directories if they don't exist."""
    os.makedirs(KNOWN_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def load_log():
    """Return the list of attendance records from the log file."""
    ensure_dirs()
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    try:
        with open(ATTENDANCE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def was_logged_today(name: str) -> bool:
    """Check if the given person was already logged today."""
    today = date.today().isoformat()
    for rec in load_log():
        if rec['name'] == name and rec['timestamp'].startswith(today):
            return True
    return False


def log_entry(name: str, snapshot_path: str, status: str) -> None:
    """Append a new attendance entry to the JSON log."""
    ensure_dirs()
    record = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'time_arrival': datetime.now().strftime('%H:%M:%S'),
        'status': status,
        'snapshot_path': snapshot_path,
    }
    data = load_log()
    data.append(record)
    with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_dashboard_html() -> str:
    """Generate the dashboard HTML file for today's records and return its path."""
    ensure_dirs()
    records = [r for r in load_log() if r['timestamp'].startswith(date.today().isoformat())]
    rows = []
    for i, rec in enumerate(records, 1):
        snap_rel = os.path.relpath(rec['snapshot_path'], 'frontend/public')
        row = (
            f'<tr>'
            f'<td>{i}</td>'
            f'<td><a href="{snap_rel}" class="snapshot-link">مشاهدة</a></td>'
            f'<td>{rec["name"]}</td>'
            f'<td>{rec["time_arrival"]}</td>'
            f'<td>{rec["status"]}</td>'
            f'<td data-timestamp="{rec["timestamp"]}">{rec["timestamp"]}</td>'
            f'</tr>'
        )
        rows.append(row)
    if not rows:
        rows.append('<tr><td colspan="6" class="no-records">لا توجد سجلات حضور لهذا اليوم بعد.</td></tr>')

    with open(os.path.join('templates', 'attendance_dashboard.html'), 'r', encoding='utf-8') as f:
        template = f.read()

    html = template
    html = html.replace('2025-06-09', date.today().isoformat())
    html = html.replace('00:47:05', datetime.now().strftime('%H:%M:%S'))
    html = html.replace('<tbody>', '<tbody>' + '\n'.join(rows))

    out_dir = os.path.join('frontend', 'public')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'attendance_dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path
