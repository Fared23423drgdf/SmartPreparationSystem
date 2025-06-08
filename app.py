import cv2
import face_recognition
import numpy as np
import os
import json
import shutil
from datetime import datetime, date, timedelta
import time
import sys
import webbrowser
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Import the new library for camera enumeration
try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    print("Error: 'cv2-enumerate-cameras' library not found.")
    print("Please install it using: pip install cv2-enumerate-cameras")
    sys.exit(1)


# --- Configuration ---
CONFIG = {
    "CAMERA_INDEX": 0, # This will be set dynamically now
    "RESIZE_FACTOR": 0.25,
    "FACE_RECOGNITION_TOLERANCE": 0.55,
    # Updated paths based on the new structure
    "KNOWN_FACES_DIR": "data/known_faces",
    "UNKNOWN_FACES_DETECTED_DIR": "data/unknown_faces_detected",
    "SNAPSHOTS_DIR": "data/snapshots",
    "ATTENDANCE_LOGS_DIR": "data/attendance_logs",
    "ATTENDANCE_LOG_FILE": "data/attendance_logs/attendance_log.json",
    "DASHBOARD_HTML_FILE": "frontend/public/attendance_dashboard.html",
    "DASHBOARD_REFRESH_INTERVAL_SECONDS": 10,
    "DASHBOARD_AUTO_CLOSE_SECONDS": 3600,
    "CONSOLE_TYPEWRITER_DELAY": 0.01,
    "DEBOUNCE_SECONDS_FOR_UNKNOWN": 5,
}

# --- Global Variables ---
known_face_encodings = []
known_face_names = []
last_unknown_log_time = {}

# Camera and threading control
video_capture = None
camera_thread = None
stop_camera_event = threading.Event()
system_running = False

# --- Console Output Utilities (less used with GUI, but kept for core logic) ---
def typewriter_print(text, delay=CONFIG["CONSOLE_TYPEWRITER_DELAY"]):
    """
    Prints text character by character to simulate a typewriter effect.
    For GUI, this might update a text widget or just print to console for debugging.
    """
    print(text, end='')
    sys.stdout.flush()
    # time.sleep(delay) # Removed for speed, as GUI should be responsive

def animated_loading_bar(duration, message_prefix="Processing"):
    """
    Displays an animated loading bar in the console.
    This will be replaced by GUI feedback in the application.
    """
    print(f"{message_prefix}...")
    time.sleep(duration)

# --- Directory and File Setup ---
def setup_directories():
    """
    Ensures that necessary directories for storing known faces, unknown face detections,
    snapshots, and attendance logs exist. Creates them if they don't.
    """
    typewriter_print("Setting up directories...\n")
    dirs_to_create = [
        CONFIG["KNOWN_FACES_DIR"],
        CONFIG["UNKNOWN_FACES_DETECTED_DIR"],
        CONFIG["SNAPSHOTS_DIR"],
        CONFIG["ATTENDANCE_LOGS_DIR"]
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        typewriter_print(f"  -> Ensured directory exists: {dir_path}\n", 0.005)
    typewriter_print("Directories setup complete.\n\n")

# --- Face Loading and Encoding ---
def load_known_faces():
    """
    Loads images from the 'known_faces' directory, encodes the faces found in them,
    and stores the encodings and corresponding names globally.
    """
    global known_face_encodings, known_face_names
    typewriter_print("Loading known faces database...\n")
    start_time = time.time()

    # Clear existing known faces before reloading
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(CONFIG["KNOWN_FACES_DIR"]) or not os.listdir(CONFIG["KNOWN_FACES_DIR"]):
        typewriter_print(f"Warning: '{CONFIG['KNOWN_FACES_DIR']}' is empty or does not exist. No known faces to load.\n", 0.005)
        typewriter_print("Please add images of known individuals to this folder (e.g., 'Your_Name.jpg').\n", 0.005)
        return

    image_files = [f for f in os.listdir(CONFIG["KNOWN_FACES_DIR"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        typewriter_print(f"Warning: No image files found in '{CONFIG['KNOWN_FACES_DIR']}'.\n", 0.005)
        return

    total_faces = len(image_files)
    loaded_count = 0
    for i, filename in enumerate(image_files):
        name = os.path.splitext(filename)[0].replace("_", " ")
        image_path = os.path.join(CONFIG["KNOWN_FACES_DIR"], filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                loaded_count += 1
            else:
                typewriter_print(f"Warning: No face found in {filename}. Skipping.\n", 0.005)
        except Exception as e:
            typewriter_print(f"Error loading {filename}: {e}\n", 0.005)

    end_time = time.time()
    duration = end_time - start_time
    if loaded_count > 0:
        typewriter_print(f"Successfully loaded {loaded_count}/{total_faces} known faces in {duration:.2f} seconds.\n\n")
    else:
        typewriter_print(f"No known faces were successfully loaded from {total_faces} files.\n\n")


# --- Attendance Logging ---
def load_attendance_log():
    """
    Loads the attendance log from a JSON file.
    Returns an empty list if the file doesn't exist or is corrupted.
    """
    try:
        # Ensure the attendance logs directory exists before trying to load
        os.makedirs(CONFIG["ATTENDANCE_LOGS_DIR"], exist_ok=True)
        if os.path.exists(CONFIG["ATTENDANCE_LOG_FILE"]):
            with open(CONFIG["ATTENDANCE_LOG_FILE"], 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        # Corrected typo here
        typewriter_print(f"Warning: Error decoding {CONFIG['ATTENDANCE_LOG_FILE']}. Starting with an empty log.\n", 0.005)
        return []
    except Exception as e:
        typewriter_print(f"Error loading attendance log: {e}\n", 0.005)
        return []

def save_attendance_log(data):
    """
    Saves the attendance log data to a JSON file.
    """
    try:
        # Ensure the attendance logs directory exists before saving
        os.makedirs(CONFIG["ATTENDANCE_LOGS_DIR"], exist_ok=True)
        with open(CONFIG["ATTENDANCE_LOG_FILE"], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        typewriter_print(f"Error saving attendance log: {e}\n", 0.005)

def log_attendance(name, frame, face_location_scaled_up, face_encoding_tuple=None):
    """
    Logs attendance for a recognized individual or an unknown individual.
    Takes a snapshot for new entries. Includes debouncing for unknown individuals.
    Returns True if a new log entry was added, False otherwise.
    """
    global last_unknown_log_time
    current_time = datetime.now()
    today_str = current_time.strftime("%Y-%m-%d")
    timestamp_str = current_time.isoformat()
    time_str = current_time.strftime("%H:%M:%S")

    all_attendance_data = load_attendance_log()
    todays_log_entries = [
        entry for entry in all_attendance_data
        if entry.get('timestamp', '').startswith(today_str)
    ]

    if name == "Unknown Individual":
        if face_encoding_tuple:
            current_timestamp_numeric = time.time()
            if face_encoding_tuple in last_unknown_log_time and \
               (current_timestamp_numeric - last_unknown_log_time[face_encoding_tuple]) < CONFIG["DEBOUNCE_SECONDS_FOR_UNKNOWN"]:
                print(f"DEBUG: Unknown Individual debounced at {time_str}.") # Debug print
                return False # Debounced, don't log again
            last_unknown_log_time[face_encoding_tuple] = current_timestamp_numeric
        
        status = "Unknown - Logged"
        snapshot_filename = f"Unknown_{current_time.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        # Corrected path for snapshot_path_relative
        snapshot_path_relative = os.path.join("data", os.path.basename(CONFIG["UNKNOWN_FACES_DETECTED_DIR"]), snapshot_filename).replace("\\", "/")
        full_snapshot_path = os.path.join(CONFIG["UNKNOWN_FACES_DETECTED_DIR"], snapshot_filename)

        top, right, bottom, left = face_location_scaled_up
        face_image = frame[top:bottom, left:right]
        if face_image.size > 0:
            cv2.imwrite(full_snapshot_path, face_image)
            typewriter_print(f"   -> Saved unknown snapshot: {snapshot_path_relative}\n", 0.005)
        else:
            typewriter_print(f"   -> Warning: Snapshot for unknown was empty, not saved.\n", 0.005)
            snapshot_path_relative = None

        new_entry = {
            "name": name, "timestamp": timestamp_str, "time_arrival": time_str,
            "status": status, "snapshot_path": snapshot_path_relative
        }
        all_attendance_data.append(new_entry)
        save_attendance_log(all_attendance_data)
        typewriter_print(f"   -> Logged 'Unknown Individual' at {time_str}.\n", 0.005)
        print(f"DEBUG: New Unknown Individual logged: {time_str}.") # Debug print
        return True # New entry added
    else: # Known individual
        already_logged_today = any(
            entry['name'] == name and entry['status'].startswith("Present")
            for entry in todays_log_entries
        )

        if not already_logged_today:
            status = "Present"
            typewriter_print(f"   -> Logging {name} as {status}.\n", 0.005)
            snapshot_filename = f"{name.replace(' ', '_')}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            # Corrected path for snapshot_path_relative
            snapshot_path_relative = os.path.join("data", os.path.basename(CONFIG["SNAPSHOTS_DIR"]), snapshot_filename).replace("\\", "/")
            full_snapshot_path = os.path.join(CONFIG["SNAPSHOTS_DIR"], snapshot_filename)

            top, right, bottom, left = face_location_scaled_up
            face_image = frame[top:bottom, left:right]

            if face_image.size > 0:
                cv2.imwrite(full_snapshot_path, face_image)
                typewriter_print(f"   -> Saved snapshot for {name}: {snapshot_path_relative}\n", 0.005)
            else:
                typewriter_print(f"   -> Warning: Snapshot for {name} was empty, not saved.\n", 0.005)
                snapshot_path_relative = None

            new_entry = {
                "name": name, "timestamp": timestamp_str, "time_arrival": time_str,
                "status": status, "snapshot_path": snapshot_path_relative
            }
            all_attendance_data.append(new_entry)
            save_attendance_log(all_attendance_data)
            typewriter_print(f"   -> {name} marked {status} at {time_str}.\n", 0.005)
            print(f"DEBUG: New Known Individual logged: {name} at {time_str}.") # Debug print
            return True # New entry added
        else:
            print(f"DEBUG: Known Individual {name} already logged today at {time_str}. No new entry.") # Debug print
    return False # No new entry added


# --- Dashboard Generation ---
def generate_dashboard_html():
    """
    Generates an HTML dashboard file displaying today's attendance records.
    Includes functionality to download data as CSV and PDF.
    """
    typewriter_print("Generating dashboard...\n", 0.005)
    print("DEBUG: generate_dashboard_html called.") # Debug print
    start_time = time.time()
    attendance_data = load_attendance_log()
    today_str = date.today().isoformat()
    todays_entries = [
        entry for entry in attendance_data
        if entry.get('timestamp', '').startswith(today_str)
    ]
    todays_entries.sort(key=lambda x: x.get('timestamp', ''))
    print(f"DEBUG: Found {len(todays_entries)} entries for today in dashboard generation.") # Debug print
    print(f"DEBUG: Todays entries for dashboard HTML generation: {todays_entries}") # New debug line

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="{CONFIG['DASHBOARD_REFRESH_INTERVAL_SECONDS']}">
        <title>Live Attendance Dashboard - {today_str}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.23/jspdf.plugin.autotable.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f0f2f5; color: #333; line-height: 1.6; }}
            .container {{ background-color: #fff; padding: 25px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 1200px; margin: 20px auto; }}
            h1 {{ color: #0056b3; text-align: center; margin-bottom: 20px; font-size: 2em; }}
            .info-bar {{ text-align: center; margin-bottom: 25px; font-size: 0.95em; color: #555; }}
            .info-bar span {{ font-weight: bold; color: #007bff; }}
            .countdown-message {{ text-align: center; margin: 20px 0; padding: 12px; background-color: #e9f7fd; border: 1px solid #b3e0f2; color: #0c5460; border-radius: 5px; }}

            .filter-section {{
                background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;
                display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; align-items: center;
                margin-bottom: 25px;
            }}
            .filter-group {{ display: flex; flex-direction: column; }}
            .filter-group label {{ font-size: 0.85em; margin-bottom: 5px; color: #555; }}
            .filter-section input[type="date"],
            .filter-section input[type="text"],
            .filter-section select {{
                padding: 8px 10px; border: 1px solid #ccc; border-radius: 4px;
                font-size: 0.9em; width: 150px;
            }}
            .filter-section button {{
                background-color: #28a745; color: white; border: none; padding: 9px 18px;
                border-radius: 5px; cursor: pointer; font-size: 0.9em; transition: background-color 0.2s;
            }}
            .filter-section button:hover {{ background-color: #218838; }}
            .filter-section button.clear {{ background-color: #6c757d; }}
            .filter-section button.clear:hover {{ background-color: #5a6268; }}


            .action-buttons {{ margin-bottom: 20px; text-align: right; }}
            .action-buttons button {{
                background-color: #007bff; color: white; border: none; padding: 10px 15px;
                border-radius: 5px; cursor: pointer; font-size: 0.9em; margin-left: 10px;
                transition: background-color 0.2s;
            }}
            .action-buttons button:hover {{ background-color: #0056b3; }}
            .action-buttons button.pdf {{ background-color: #dc3545; }}
            .action-buttons button.pdf:hover {{ background-color: #c82333; }}

            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.95em; }}
            th, td {{ padding: 12px 15px; border: 1px solid #dee2e6; text-align: left; vertical-align: middle; }}
            th {{ background-color: #007bff; color: white; font-weight: 600; text-transform: uppercase; }}
            tbody tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tbody tr:hover {{ background-color: #e9ecef; }}
            .status-present {{ color: #28a745; font-weight: bold; }}
            .status-unknown {{ color: #fd7e14; font-weight: bold; }}
            .status-present::before {{ content: '✔ '; }}
            .status-unknown::before {{ content: '⚠ '; }}
            .snapshot-link {{ color: #007bff; text-decoration: underline; cursor: pointer; }}
            .snapshot-link:hover {{ color: #0056b3; }}

            .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #6c757d; }}
            .no-records {{ text-align: center; padding: 20px; color: #888; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Live Attendance Dashboard</h1>
            <div class="info-bar">
                <span>Date: {today_str}</span> | Last Updated: <span>{datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            <div class="countdown-message" id="countdown-message">
                This page will show a session end message in <span id="countdown-timer">{CONFIG['DASHBOARD_AUTO_CLOSE_SECONDS']}</span> seconds. It will continue to auto-refresh.
            </div>

            <div class="filter-section">
                <div class="filter-group">
                    <label for="startDate">تاريخ البدء:</label>
                    <input type="date" id="startDate">
                </div>
                <div class="filter-group">
                    <label for="endDate">تاريخ الانتهاء:</label>
                    <input type="date" id="endDate">
                </div>
                <div class="filter-group">
                    <label for="filterName">الاسم:</label>
                    <input type="text" id="filterName" placeholder="بحث بالاسم">
                </div>
                <div class="filter-group">
                    <label for="filterStatus">الحالة:</label>
                    <select id="filterStatus">
                        <option value="">الكل</option>
                        <option value="Present">حاضر</option>
                        <option value="Unknown - Logged">فرد غير معروف</option>
                    </select>
                </div>
                <button onclick="applyFilters()">تطبيق الفلاتر</button>
                <button class="clear" onclick="clearFilters()">مسح الفلاتر</button>
            </div>

            <div class="action-buttons">
                <button onclick="downloadCSV('attendance_log_{today_str}.csv')">تنزيل CSV مفلتر</button>
                <button onclick="downloadPDF('attendance_log_{today_str}.pdf')" class="pdf">تنزيل PDF مفلتر</button>
            </div>

            <table id="attendanceTable">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>اللقطة</th>
                        <th>الاسم</th>
                        <th>وقت الوصول</th>
                        <th>الحالة</th>
                        <th>الطابع الزمني (كامل)</th>
                    </tr>
                </thead>
                <tbody>
    """
    if not todays_entries:
        html_content += '<tr><td colspan="6" class="no-records">لا توجد سجلات حضور لهذا اليوم بعد.</td></tr>'
    else:
        for i, entry in enumerate(todays_entries):
            status_class = ""
            if entry['status'] == "Present":
                status_class = "status-present"
            elif entry['status'] == "Unknown - Logged":
                status_class = "status-unknown"

            snapshot_display = "N/A"
            # Construct full path for snapshot to check existence
            # Adjust path for display in HTML, assuming HTML is in frontend/public
            # and snapshots are in data/snapshots or data/unknown_faces_detected
            base_snapshot_dir = ""
            if entry.get("snapshot_path"):
                if "unknown_faces_detected" in entry["snapshot_path"]:
                    base_snapshot_dir = "../../data/unknown_faces_detected"
                elif "snapshots" in entry["snapshot_path"]:
                    base_snapshot_dir = "../../data/snapshots"
                
                img_path = os.path.join(base_snapshot_dir, os.path.basename(entry["snapshot_path"])).replace("\\", "/")
                
                # Check if the file actually exists on the server
                # This check is done on the Python side before generating HTML
                full_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", base_snapshot_dir, os.path.basename(entry["snapshot_path"]))
                if os.path.exists(full_server_path):
                    snapshot_display = f'<a href="{img_path}" target="_blank" class="snapshot-link">عرض الصورة</a>'
                else:
                    snapshot_display = "Image Not Found"


            html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{snapshot_display}</td>
                        <td>{entry.get('name', 'N/A')}</td>
                        <td>{entry.get('time_arrival', 'N/A')}</td>
                        <td class="{status_class}">{entry.get('status', 'N/A')}</td>
                        <td data-timestamp="{entry.get('timestamp', '')}">{entry.get('timestamp', 'N/A')}</td>
                    </tr>"""
    html_content += f"""
                </tbody>
            </table>
            <div class="footer">
                نظام الحضور بالتعرف على الوجه بالذكاء الاصطناعي. يتم التحديث تلقائياً كل {CONFIG['DASHBOARD_REFRESH_INTERVAL_SECONDS']} ثانية.
            </div>
        </div>

        <script>
            let countdown = {CONFIG['DASHBOARD_AUTO_CLOSE_SECONDS']};
            const timerElement = document.getElementById('countdown-timer');
            const messageElement = document.getElementById('countdown-message');

            // --- Countdown Timer ---
            function updateCountdown() {{
                if (countdown > 0) {{
                    timerElement.textContent = countdown;
                    countdown--;
                }} else {{
                    messageElement.innerHTML = 'انتهت فترة عرض جلسة لوحة التحكم. ستستمر الصفحة في التحديث ببيانات جديدة إذا كان نص Python البرمجي قيد التشغيل.';
                    clearInterval(intervalId);
                }}
            }}
            const intervalId = setInterval(updateCountdown, 1000);
            updateCountdown(); // Initial call

            // --- Filtering Logic ---
            function applyFilters() {{
                const startDate = document.getElementById('startDate').value;
                const endDate = document.getElementById('endDate').value;
                const filterName = document.getElementById('filterName').value.toLowerCase();
                const filterStatus = document.getElementById('filterStatus').value;

                const table = document.getElementById('attendanceTable');
                const tr = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

                for (let i = 0; i < tr.length; i++) {{
                    let displayRow = true;

                    // Skip the 'No records' row if it exists
                    if (tr[i].classList.contains('no-records') || tr[i].querySelector('.no-records')) {{
                        tr[i].style.display = 'none';
                        continue;
                    }}

                    const nameCell = tr[i].getElementsByTagName('td')[2];
                    const statusCell = tr[i].getElementsByTagName('td')[4];
                    const timestampCell = tr[i].getElementsByTagName('td')[5];

                    if (nameCell) {{
                        const nameText = nameCell.textContent.toLowerCase();
                        if (filterName && !nameText.includes(filterName)) {{
                            displayRow = false;
                        }}
                    }}

                    if (statusCell) {{
                        const statusText = statusCell.textContent.trim();
                        if (filterStatus && (filterStatus === "Present" && !statusText.startsWith("Present"))) {{
                            displayRow = false;
                        }} else if (filterStatus && (filterStatus === "Unknown - Logged" && statusText !== "Unknown - Logged")) {{
                            displayRow = false;
                        }}
                    }}

                    if (timestampCell && (startDate || endDate)) {{
                        const fullTimestamp = timestampCell.getAttribute('data-timestamp');
                        if (fullTimestamp) {{
                            const recordDate = fullTimestamp.substring(0, 10);
                            if (startDate && recordDate < startDate) {{
                                displayRow = false;
                            }}
                            if (endDate && recordDate > endDate) {{
                                displayRow = false;
                            }}
                        }} else {{
                             displayRow = false;
                        }}
                    }}

                    tr[i].style.display = displayRow ? '' : 'none';
                }}
            }}

            function clearFilters() {{
                document.getElementById('startDate').value = '';
                document.getElementById('endDate').value = '';
                document.getElementById('filterName').value = '';
                document.getElementById('filterStatus').value = '';
                applyFilters();
            }}

            // --- CSV Export Logic ---
            function escapeCsvCell(cellData) {{
                if (cellData == null) return '';
                let stringData = String(cellData);
                if (stringData.includes('<a href')) {{
                    const hrefMatch = stringData.match(/href="([^"]*)"/);
                    stringData = (hrefMatch && hrefMatch[1]) ? hrefMatch[1] : "Snapshot Link";
                }} else {{
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = stringData;
                    stringData = tempDiv.innerText;
                }}

                if (stringData.includes(',') || stringData.includes('"') || stringData.includes('\\n')) {{
                    return '"' + stringData.replace(/"/g, '""') + '"';
                }}
                return stringData;
            }}

            function downloadCSV(filename) {{
                let csv = [];
                const table = document.getElementById('attendanceTable');
                const headerRow = table.getElementsByTagName('thead')[0].getElementsByTagName('tr')[0];
                const visibleBodyRows = Array.from(table.getElementsByTagName('tbody')[0].getElementsByTagName('tr'))
                                        .filter(row => row.style.display !== 'none');

                const headerCols = headerRow.querySelectorAll("th");
                let headerData = [];
                for (let i = 0; i < headerCols.length; i++) {{
                    headerData.push(escapeCsvCell(headerCols[i].innerText));
                }}
                csv.push(headerData.join(","));

                for (let i = 0; i < visibleBodyRows.length; i++) {{
                    const row = [], cols = visibleBodyRows[i].querySelectorAll("td");
                    for (let j = 0; j < cols.length; j++) {{
                        let cellContent = cols[j].innerHTML;
                        if (j === 1 && cols[j].querySelector('a')) {{
                             cellContent = cols[j].querySelector('a').href;
                        }} else if (j === 5 && cols[j].hasAttribute('data-timestamp')) {{
                            cellContent = cols[j].getAttribute('data-timestamp');
                        }} else {{
                             cellContent = cols[j].innerText;
                        }}
                        row.push(escapeCsvCell(cellContent));
                    }}
                    csv.push(row.join(","));
                }}

                const csvFile = new Blob([csv.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
                const downloadLink = document.createElement("a");
                downloadLink.download = filename;
                downloadLink.href = window.URL.createObjectURL(csvFile);
                downloadLink.style.display = "none";
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }}

            // --- PDF Export Logic ---
            function downloadPDF(filename) {{
                const {{jsPDF}} = window.jspdf;
                const doc = new jsPDF({{ orientation: 'landscape' }});
                doc.text("Attendance Log - {today_str}", 14, 16);
                doc.setFontSize(10);
                doc.text("Generated: " + new Date().toLocaleString(), 14, 22);

                const table = document.getElementById('attendanceTable');
                const header = Array.from(table.getElementsByTagName('thead')[0].getElementsByTagName('th')).map(th => th.innerText);
                const body = [];
                const visibleBodyRows = Array.from(table.getElementsByTagName('tbody')[0].getElementsByTagName('tr'))
                                        .filter(row => row.style.display !== 'none');

                visibleBodyRows.forEach(row => {{
                    const rowData = [];
                    const cells = row.querySelectorAll('td');
                    cells.forEach((cell, index) => {{
                        if (index === 1) {{
                            const linkElement = cell.querySelector('a');
                            rowData.push(linkElement ? "Snapshot Available (Link)" : "N/A");
                        }} else if (index === 5 && cell.hasAttribute('data-timestamp')) {{
                            rowData.push(cell.getAttribute('data-timestamp'));
                        }} else {{
                            rowData.push(cell.innerText.trim());
                        }}
                    }});
                    body.push(rowData);
                }});


                doc.autoTable({{
                    head: [header],
                    body: body,
                    startY: 28,
                    theme: 'grid',
                    headStyles: {{ fillColor: [0, 123, 255] }},
                    columnStyles: {{
                        0: {{ cellWidth: 15 }},
                        1: {{ cellWidth: 35 }},
                        2: {{ cellWidth: 40 }},
                        3: {{ cellWidth: 25 }},
                        4: {{ cellWidth: 35 }},
                        5: {{ cellWidth: 'auto' }}
                    }}
                }});
                doc.save(filename);
            }}

            // Initial filter application when page loads (useful if you pre-set dates in the future)
            document.addEventListener('DOMContentLoaded', () => {{
                const today = new Date().toISOString().split('T')[0];
                document.getElementById('startDate').value = today;
                document.getElementById('endDate').value = today;
                applyFilters(); // Apply filters on load

                // --- Debugging for empty table ---
                const tableBody = document.getElementById('attendanceTable').getElementsByTagName('tbody')[0];
                const rowCount = tableBody.getElementsByTagName('tr').length;
                console.log('DEBUG (Browser): Table body row count on DOMContentLoaded:', rowCount);
                if (rowCount === 0 || (rowCount === 1 && tableBody.querySelector('.no-records'))) {{
                    console.log('DEBUG (Browser): Table appears empty or only has "no records" message.');
                }} else {{
                    console.log('DEBUG (Browser): Table appears to have data. First row innerHTML:', tableBody.getElementsByTagName('tr')[0].innerHTML);
                }}
                // --- End Debugging ---
            }});
        </script>
    </body>
    </html>
    """
    try:
        # Ensure the directory exists before writing the file
        dashboard_dir = os.path.dirname(CONFIG["DASHBOARD_HTML_FILE"])
        os.makedirs(dashboard_dir, exist_ok=True)
        
        with open(CONFIG["DASHBOARD_HTML_FILE"], 'w', encoding='utf-8') as f:
            f.write(html_content)
        end_time = time.time()
        typewriter_print(f"Dashboard generated in {end_time - start_time:.2f}s: {CONFIG['DASHBOARD_HTML_FILE']}\n", 0.005)
        return True
    except Exception as e:
        typewriter_print(f"Error generating dashboard: {e}\n", 0.005)
        return False

# --- Web Browser Interaction ---
dashboard_opened_today = False
def open_dashboard_in_browser_if_needed(force_open=False):
    """
    Opens the generated attendance dashboard HTML file in the default web browser.
    """
    global dashboard_opened_today
    if force_open or not dashboard_opened_today:
        try:
            # Construct the absolute path to the dashboard HTML file
            # Assuming app.py is in 'backend/' and dashboard is in 'frontend/public/'
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            dashboard_path_abs = os.path.join(current_script_dir, "..", CONFIG["DASHBOARD_HTML_FILE"])
            dashboard_path_abs = os.path.normpath(dashboard_path_abs) # Normalize path

            webbrowser.open_new_tab(f'file://{dashboard_path_abs}')
            typewriter_print(f"Opening dashboard: file://{dashboard_path_abs}\n", 0.005)
            dashboard_opened_today = True
        except Exception as e:
            typewriter_print(f"Error opening dashboard in browser: {e}\n", 0.005)


# --- Camera Selection Functions ---
def list_available_cameras_with_names():
    """
    Detects and lists available cameras using cv2-enumerate-cameras to get actual names.
    """
    available_cameras_info = []
    typewriter_print(f"Searching for available cameras with actual names...\n")
    try:
        # Use the enumerate_cameras function from the imported library
        camera_list = enumerate_cameras()
        if not camera_list:
            typewriter_print("  -> No cameras found using cv2-enumerate-cameras.\n", 0.005)
            # Fallback to basic OpenCV check if the library finds nothing
            return list_available_cameras_fallback()

        for cam_info in camera_list:
            # cam_info is a CameraInfo object with attributes like index, name, backend
            available_cameras_info.append({
                "index": cam_info.index,
                "name": cam_info.name,
                "api_preference": cam_info.backend # Store the backend if available
            })
            typewriter_print(f"  -> Found camera: Index {cam_info.index}, Name: '{cam_info.name}', Backend: {cam_info.backend}\n", 0.005)
    except Exception as e:
        typewriter_print(f"  -> Error enumerating cameras with cv2-enumerate-cameras: {e}\n", 0.005)
        typewriter_print("  -> Falling back to basic OpenCV camera detection.\n", 0.005)
        # Fallback to basic OpenCV check if the library fails
        return list_available_cameras_fallback()
        
    return available_cameras_info

def list_available_cameras_fallback(max_cameras_to_check=5):
    """
    Fallback function to detect and list available cameras by trying different indices
    and OpenCV backends, if cv2-enumerate-cameras fails or is not installed.
    """
    available_cameras = []
    typewriter_print(f"Searching for available cameras (fallback, checking first {max_cameras_to_check} indices)...\n")
    backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", None: "Default"}

    for i in range(max_cameras_to_check):
        found_for_index_i = False
        display_name_prefix = f"الكاميرا {i}"
        if i == 0:
            display_name_prefix = "الكاميرا المدمجة (افتراضي)"
        elif i > 0:
            display_name_prefix = f"كاميرا خارجية {i}"

        for backend in backends_to_try:
            cap = None
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(i, backend)
                else:
                    cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_name = display_name_prefix
                        if backend is not None:
                            camera_name = f"{display_name_prefix} (الخلفية: {backend_names[backend]})"

                        if not any(cam['index'] == i for cam in available_cameras):
                            available_cameras.append({"index": i, "name": camera_name, "api_preference": backend})
                            typewriter_print(f"  -> Found working camera: {camera_name}\n", 0.005)
                        found_for_index_i = True
                        break
                    else:
                         typewriter_print(f"  -> Index {i} (Backend: {backend_names[backend]}) opened but no frame.\n", 0.001)

            except Exception as e:
                typewriter_print(f"  -> Error testing Index {i} (Backend: {backend_names[backend]}): {e}\n", 0.001)
            finally:
                if cap and cap.isOpened():
                    cap.release()
            time.sleep(0.02)
        if not found_for_index_i:
             typewriter_print(f"  -> No working backend found for camera index {i}.\n", 0.005)
        time.sleep(0.05)
    return available_cameras


# --- GUI Application ---
class AttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("نظام الحضور والتعرف على الوجه")
        master.geometry("600x650")
        master.resizable(False, False)
        master.configure(bg='#f0f2f5')

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f2f5')
        self.style.configure('TLabel', background='#f0f2f5', foreground='#333', font=('Segoe UI', 10))
        self.style.configure('TButton', background='#007bff', foreground='white', font=('Segoe UI', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#0056b3')])
        self.style.configure('TCombobox', font=('Segoe UI', 10))
        self.style.configure('TEntry', font=('Segoe UI', 10))
        self.style.configure('Treeview', font=('Segoe UI', 9))
        self.style.configure('Treeview.Heading', font=('Segoe UI', 9, 'bold'))


        self.available_cameras = []
        self.selected_camera_index = None
        self.selected_api_preference = None
        self.system_running_flag = False
        self.camera_thread = None
        self.stop_camera_event = threading.Event()

        self._create_widgets()
        setup_directories()
        self._load_initial_data()

    def _create_widgets(self):
        # --- Camera Selection Frame ---
        camera_frame = ttk.LabelFrame(self.master, text="إعدادات الكاميرا", padding="15 10")
        camera_frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(camera_frame, text="اختر الكاميرا:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.camera_combobox = ttk.Combobox(camera_frame, state="readonly", width=40)
        self.camera_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.camera_combobox.bind("<<ComboboxSelected>>", self._on_camera_selected)

        # --- Control Buttons Frame ---
        control_frame = ttk.Frame(self.master, padding="15 10")
        control_frame.pack(pady=10, padx=20, fill="x")

        self.start_button = ttk.Button(control_frame, text="بدء النظام", command=self.start_system_gui)
        self.start_button.pack(side="left", padx=5, expand=True)

        self.stop_button = ttk.Button(control_frame, text="إيقاف النظام", command=self.stop_system_gui, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5, expand=True)

        self.dashboard_button = ttk.Button(control_frame, text="فتح لوحة التحكم", command=self.open_dashboard_gui)
        self.dashboard_button.pack(side="left", padx=5, expand=True)

        # --- Known Faces Management Frame ---
        faces_management_frame = ttk.LabelFrame(self.master, text="إدارة الوجوه المعروفة", padding="15 10")
        faces_management_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Treeview to display known faces
        self.known_faces_tree = ttk.Treeview(faces_management_frame, columns=("Name",), show="headings")
        self.known_faces_tree.heading("Name", text="الاسم")
        self.known_faces_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Scrollbar for the Treeview
        tree_scrollbar = ttk.Scrollbar(faces_management_frame, orient="vertical", command=self.known_faces_tree.yview)
        tree_scrollbar.pack(side="right", fill="y")
        self.known_faces_tree.configure(yscrollcommand=tree_scrollbar.set)

        # Buttons for managing known faces
        face_buttons_frame = ttk.Frame(faces_management_frame)
        face_buttons_frame.pack(side="top", fill="x", padx=5, pady=5)

        ttk.Button(face_buttons_frame, text="إضافة وجه جديد", command=self.add_known_face_gui).pack(side="left", padx=2, pady=2, expand=True)
        ttk.Button(face_buttons_frame, text="تعديل الاسم", command=self.edit_known_face_name_gui).pack(side="left", padx=2, pady=2, expand=True)
        ttk.Button(face_buttons_frame, text="حذف الوجه", command=self.delete_known_face_gui).pack(side="left", padx=2, pady=2, expand=True)
        
        # --- Status Bar ---
        self.status_label = ttk.Label(self.master, text="الحالة: جاهز", anchor="w", font=('Segoe UI', 10, 'italic'))
        self.status_label.pack(side="bottom", fill="x", padx=20, pady=10)

    def _load_initial_data(self):
        self.update_status("جاري البحث عن الكاميرات...")
        # Use the new function to list cameras with actual names
        cameras = list_available_cameras_with_names()
        if not cameras:
            messagebox.showerror("خطأ في الكاميرا", "لم يتم العثور على أي كاميرات عاملة. يرجى التحقق من الاتصال.")
            self.update_status("الحالة: خطأ في الكاميرا")
            self.start_button.config(state=tk.DISABLED)
            return

        self.available_cameras = cameras
        camera_names = [cam['name'] for cam in cameras]
        self.camera_combobox['values'] = camera_names
        if camera_names:
            self.camera_combobox.set(camera_names[0])
            self.selected_camera_index = cameras[0]['index']
            self.selected_api_preference = cameras[0]['api_preference']
            self.update_status(f"الحالة: تم تحديد الكاميرا: {camera_names[0]}")
        else:
            self.update_status("الحالة: لا توجد كاميرات متاحة")
            self.start_button.config(state=tk.DISABLED)

        self.update_status("جاري تحميل الوجوه المعروفة...")
        self.refresh_known_faces_display()
        if not known_face_encodings:
            self.update_status("الحالة: لا توجد وجوه معروفة (سيتعرف على 'غير معروف' فقط)")
        else:
            self.update_status(f"الحالة: جاهز. تم تحميل {len(known_face_encodings)} وجه معروف.")

    def _on_camera_selected(self, event):
        selected_name = self.camera_combobox.get()
        for cam in self.available_cameras:
            if cam['name'] == selected_name:
                self.selected_camera_index = cam['index']
                self.selected_api_preference = cam['api_preference']
                self.update_status(f"الحالة: تم تحديد الكاميرا: {selected_name}")
                break

    def update_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()
            
    def _run_camera_loop(self):
        global video_capture
        if self.selected_camera_index is None:
            self.update_status("خطأ: لم يتم تحديد الكاميرا.")
            return

        backend_name = "Default"
        if self.selected_api_preference == cv2.CAP_DSHOW: backend_name = "DSHOW"
        elif self.selected_api_preference == cv2.CAP_MSMF: backend_name = "MSMF"
        elif self.selected_api_preference is not None:
             backend_name = str(self.selected_api_preference) # In case cv2-enumerate-cameras returns a different backend constant

        self.update_status(f"جاري تهيئة الكاميرا (المؤشر: {self.selected_camera_index}, API: {backend_name})...")

        if self.selected_api_preference is not None:
            video_capture = cv2.VideoCapture(self.selected_camera_index, self.selected_api_preference)
        else:
            video_capture = cv2.VideoCapture(self.selected_camera_index)

        if not video_capture.isOpened():
            self.update_status(f"خطأ: تعذر فتح مصدر الفيديو (الكاميرا {self.selected_camera_index}).")
            messagebox.showerror("خطأ في الكاميرا", f"تعذر فتح الكاميرا {self.selected_camera_index}. يرجى التحقق من الاتصال والأذونات.")
            self.stop_system_gui()
            return

        self.update_status("النظام يعمل. (اضغط 'q' في نافذة الفيديو للإيقاف اليدوي)")
        
        face_locations = []
        recognized_names_in_frame = []
        process_this_frame_toggle = True

        while not self.stop_camera_event.is_set():
            ret, frame = video_capture.read()
            if not ret or frame is None:
                self.update_status("\nخطأ: تعذر قراءة الإطار من الكاميرا. قد يكون البث قد انتهى أو تم فصل الكاميرا. إيقاف.")
                break

            if process_this_frame_toggle:
                small_frame = cv2.resize(frame, (0, 0), fx=CONFIG["RESIZE_FACTOR"], fy=CONFIG["RESIZE_FACTOR"])
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                temp_recognized_names = []
                
                for i, face_encoding in enumerate(current_face_encodings):
                    name = "Unknown Individual"
                    if known_face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=CONFIG["FACE_RECOGNITION_TOLERANCE"])
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                    
                    temp_recognized_names.append(name)
                    
                    top_s, right_s, bottom_s, left_s = face_locations[i]
                    top = int(top_s / CONFIG["RESIZE_FACTOR"])
                    right = int(right_s / CONFIG["RESIZE_FACTOR"])
                    bottom = int(bottom_s / CONFIG["RESIZE_FACTOR"])
                    left = int(left_s / CONFIG["RESIZE_FACTOR"])
                    face_location_original_frame = (top, right, bottom, left)

                    encoding_tuple_key = None
                    if name == "Unknown Individual":
                        encoding_tuple_key = tuple(np.round(face_encoding[:8], decimals=3))

                    # Log attendance if it's a new entry (unknown or first time known today)
                    log_attendance(name, frame.copy(), face_location_original_frame, encoding_tuple_key)
                
                recognized_names_in_frame = temp_recognized_names

                # Always regenerate dashboard and open/focus if faces are detected in the current frame
                if face_locations: # If any face was detected
                    self.master.after(0, generate_dashboard_html)
                    print("DEBUG: Dashboard HTML regeneration triggered.") # Debug print
                    # Open dashboard automatically if not already opened for the day
                    # This will open it once. Subsequent refreshes are handled by meta refresh in HTML.
                    self.master.after(100, lambda: open_dashboard_in_browser_if_needed(force_open=False))


            process_this_frame_toggle = not process_this_frame_toggle

            display_frame = frame.copy()

            names_for_display_this_iteration = ["Processing..."] * len(face_locations)
            if len(recognized_names_in_frame) == len(face_locations):
                names_for_display_this_iteration = recognized_names_in_frame
            elif not known_face_encodings and face_locations:
                names_for_display_this_iteration = ["Unknown Individual"] * len(face_locations)

            for i_loc, (top_small, right_small, bottom_small, left_small) in enumerate(face_locations):
                name_display = "Processing..."
                if i_loc < len(names_for_display_this_iteration):
                     name_display = names_for_display_this_iteration[i_loc]

                top = int(top_small / CONFIG["RESIZE_FACTOR"])
                right = int(right_small / CONFIG["RESIZE_FACTOR"])
                bottom = int(bottom_small / CONFIG["RESIZE_FACTOR"])
                left = int(left_small / CONFIG["RESIZE_FACTOR"])

                box_color = (0, 0, 255)
                if name_display.startswith("Present"):
                    box_color = (0, 255, 0)
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
                # Corrected the typo here from box2_color to box_color
                cv2.rectangle(display_frame, (left, bottom - 25), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, name_display, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

            cv2.imshow('Attendance System - Live Feed (Press \'q\' to quit)', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera_event.set()
                break

        if video_capture:
            video_capture.release()
        cv2.destroyAllWindows()
        self.update_status("الحالة: تم إيقاف الكاميرا.")
        self.system_running_flag = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def start_system_gui(self):
        if self.system_running_flag:
            self.update_status("النظام يعمل بالفعل.")
            return
        
        if self.selected_camera_index is None:
            messagebox.showwarning("تحذير", "يرجى تحديد كاميرا أولاً.")
            return

        self.stop_camera_event.clear()
        self.camera_thread = threading.Thread(target=self._run_camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        self.system_running_flag = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("الحالة: بدء النظام يدوياً.")

    def stop_system_gui(self):
        if not self.system_running_flag:
            self.update_status("النظام متوقف بالفعل.")
            return

        self.stop_camera_event.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.update_status("الحالة: جاري إيقاف النظام...")
        
        self.update_status("الحالة: إيقاف النظام يدوياً.")

    def open_dashboard_gui(self):
        if generate_dashboard_html():
            open_dashboard_in_browser_if_needed(force_open=True)
            self.update_status("الحالة: تم فتح لوحة التحكم.")
        else:
            self.update_status("الحالة: فشل فتح لوحة التحكم.")

    def refresh_known_faces_display(self):
        """
        Clears and repopulates the Treeview with current known faces.
        """
        for item in self.known_faces_tree.get_children():
            self.known_faces_tree.delete(item)
        
        load_known_faces()
        
        for name in known_face_names:
            self.known_faces_tree.insert("", "end", values=(name,))
        self.update_status(f"الحالة: تم تحديث قائمة الوجوه المعروفة. ({len(known_face_encodings)} وجه)")

    def add_known_face_gui(self):
        """
        Opens a file dialog to select images and prompts for a name to add new known faces.
        """
        file_paths = filedialog.askopenfilenames(
            title="اختر صور الوجوه المعروفة",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_paths:
            self.update_status("الحالة: لم يتم اختيار أي صور لإضافة وجه معروف.")
            return

        name_dialog = tk.Toplevel(self.master)
        name_dialog.title("إدخال الاسم")
        name_dialog.geometry("300x150")
        name_dialog.transient(self.master)
        name_dialog.grab_set()

        ttk.Label(name_dialog, text="أدخل الاسم لهذا الوجه (مثل: John Doe):").pack(pady=10)
        name_entry = ttk.Entry(name_dialog, width=30)
        name_entry.pack(pady=5)
        name_entry.focus_set()

        def on_submit():
            person_name = name_entry.get().strip()
            if not person_name:
                messagebox.showwarning("تحذير", "الاسم لا يمكن أن يكون فارغاً.", parent=name_dialog)
                return
            
            clean_name = person_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

            try:
                for idx, src_path in enumerate(file_paths):
                    base_name, ext = os.path.splitext(os.path.basename(src_path))
                    dest_filename = f"{clean_name}{ext}" if len(file_paths) == 1 else f"{clean_name}_{idx}{ext}"
                    dest_path = os.path.join(CONFIG["KNOWN_FACES_DIR"], dest_filename)
                    
                    shutil.copy(src_path, dest_path)
                    self.update_status(f"تم نسخ '{os.path.basename(src_path)}' إلى '{CONFIG['KNOWN_FACES_DIR']}'.")
                
                name_dialog.destroy()
                self.refresh_known_faces_display()
                self.update_status(f"الحالة: تم إضافة الوجوه وتحديثها. تم تحميل {len(known_face_encodings)} وجه معروف.")

            except Exception as e:
                messagebox.showerror("خطأ", f"حدث خطأ أثناء إضافة الوجه: {e}", parent=name_dialog)
                name_dialog.destroy()
                self.update_status("الحالة: فشل إضافة وجه معروف.")

        submit_button = ttk.Button(name_dialog, text="إضافة", command=on_submit)
        submit_button.pack(pady=10)

        self.master.wait_window(name_dialog)

    def edit_known_face_name_gui(self):
        """
        Allows editing the name of a selected known face.
        """
        selected_items = self.known_faces_tree.selection()
        if not selected_items:
            messagebox.showwarning("تحذير", "الرجاء تحديد وجه لتعديل اسمه.")
            return

        selected_item = selected_items[0]
        old_name = self.known_faces_tree.item(selected_item, 'values')[0]

        edit_dialog = tk.Toplevel(self.master)
        edit_dialog.title("تعديل الاسم")
        edit_dialog.geometry("300x150")
        edit_dialog.transient(self.master)
        edit_dialog.grab_set()

        ttk.Label(edit_dialog, text=f"تعديل اسم: {old_name}\nأدخل الاسم الجديد:").pack(pady=10)
        new_name_entry = ttk.Entry(edit_dialog, width=30)
        new_name_entry.pack(pady=5)
        new_name_entry.insert(0, old_name)
        new_name_entry.focus_set()

        def on_edit_submit():
            new_name = new_name_entry.get().strip()
            if not new_name:
                messagebox.showwarning("تحذير", "الاسم الجديد لا يمكن أن يكون فارغاً.", parent=edit_dialog)
                return
            
            if new_name == old_name:
                messagebox.showinfo("معلومة", "الاسم الجديد هو نفسه الاسم القديم.", parent=edit_dialog)
                edit_dialog.destroy()
                return

            try:
                old_clean_name = old_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                new_clean_name = new_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

                files_to_rename = [f for f in os.listdir(CONFIG["KNOWN_FACES_DIR"]) if f.startswith(old_clean_name)]
                
                if not files_to_rename:
                    messagebox.showwarning("خطأ", f"لم يتم العثور على ملفات صور لـ '{old_name}'.", parent=edit_dialog)
                    edit_dialog.destroy()
                    return

                for filename in files_to_rename:
                    base_name, ext = os.path.splitext(filename)
                    if "_" in base_name and base_name.startswith(old_clean_name):
                        suffix = base_name[len(old_clean_name):]
                        new_filename = f"{new_clean_name}{suffix}{ext}"
                    else:
                        new_filename = f"{new_clean_name}{ext}"

                    old_path = os.path.join(CONFIG["KNOWN_FACES_DIR"], filename)
                    new_path = os.path.join(CONFIG["KNOWN_FACES_DIR"], new_filename)
                    
                    if os.path.exists(new_path) and old_path != new_path:
                        messagebox.showwarning("تحذير", f"الملف '{new_filename}' موجود بالفعل. يرجى اختيار اسم مختلف.", parent=edit_dialog)
                        return

                    os.rename(old_path, new_path)
                    self.update_status(f"تمت إعادة تسمية '{filename}' إلى '{new_filename}'.")

                edit_dialog.destroy()
                self.refresh_known_faces_display()
                self.update_status(f"الحالة: تم تحديث اسم الوجه من '{old_name}' إلى '{new_name}'.")

            except Exception as e:
                messagebox.showerror("خطأ", f"حدث خطأ أثناء تعديل الاسم: {e}", parent=edit_dialog)
                edit_dialog.destroy()
                self.update_status("الحالة: فشل تعديل اسم الوجه.")

        submit_button = ttk.Button(edit_dialog, text="تعديل", command=on_edit_submit)
        submit_button.pack(pady=10)
        self.master.wait_window(edit_dialog)

    def delete_known_face_gui(self):
        """
        Deletes the selected known face(s) from the system.
        """
        selected_items = self.known_faces_tree.selection()
        if not selected_items:
            messagebox.showwarning("تحذير", "الرجاء تحديد وجه واحد على الأقل لحذفه.")
            return

        names_to_delete = [self.known_faces_tree.item(item, 'values')[0] for item in selected_items]

        if messagebox.askokcancel("تأكيد الحذف", f"هل أنت متأكد أنك تريد حذف الوجوه التالية؟\n{', '.join(names_to_delete)}\nسيتم حذف جميع الصور المرتبطة بها من مجلد الوجوه المعروفة.", icon='warning'):
            try:
                deleted_count = 0
                for name_to_delete in names_to_delete:
                    clean_name_to_delete = name_to_delete.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    
                    files_to_delete = [f for f in os.listdir(CONFIG["KNOWN_FACES_DIR"]) if f.startswith(clean_name_to_delete)]
                    
                    if not files_to_delete:
                        self.update_status(f"لم يتم العثور على ملفات صور لـ '{name_to_delete}'.")
                        continue

                    for filename in files_to_delete:
                        file_path = os.path.join(CONFIG["KNOWN_FACES_DIR"], filename)
                        os.remove(file_path)
                        self.update_status(f"تم حذف الملف: {filename}")
                    deleted_count += 1

                self.refresh_known_faces_display()
                self.update_status(f"الحالة: تم حذف {deleted_count} وجه معروف بنجاح.")

            except Exception as e:
                messagebox.showerror("خطأ", f"حدث خطأ أثناء حذف الوجه: {e}")
                self.update_status("الحالة: فشل حذف وجه معروف.")


    def on_closing(self):
        if messagebox.askokcancel("خروج", "هل أنت متأكد أنك تريد إغلاق التطبيق؟ سيتم إيقاف نظام الكاميرا."):
            self.stop_system_gui()
            self.master.destroy()

if __name__ == "__main__":
    root = None # Initialize root to None
    app = None # Initialize app to None
    try:
        import cv2
        import face_recognition
        import numpy
        import webbrowser
        import tkinter
        # Only create Tkinter root and app if all imports succeed
        root = tk.Tk()
        app = AttendanceApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except ImportError as e:
        # If an ImportError occurs, display an error message and exit
        print(f"Error: Missing one or more required libraries: {e}")
        error_message = (
            f"حدث خطأ: مكتبة أو أكثر مطلوبة غير موجودة: {e}\n"
            "يرجى التأكد من تثبيت 'opencv-python', 'face_recognition', 'numpy', و 'tkinter' (مدمج).\n"
            "يمكنك تثبيتها عادةً باستخدام: pip install opencv-python face_recognition numpy cv2-enumerate-cameras"
        )
        # Use a messagebox for the GUI user
        messagebox.showerror("خطأ في المكتبات", error_message)
        # Ensure Tkinter window is destroyed if it was created before the error
        if root:
            root.destroy()
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during startup
        print(f"An unexpected error occurred during application startup: {e}")
        messagebox.showerror("خطأ في التطبيق", f"حدث خطأ غير متوقع أثناء تشغيل التطبيق: {e}")
        if root:
            root.destroy()
        sys.exit(1)
