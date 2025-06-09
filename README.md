# Smart Attendance System

This repository contains a face recognition based attendance system built with Python. The application captures faces from a webcam, matches them against a database of known images and records attendance to a JSON log. A live HTML dashboard displays daily attendance records with filtering and export options.

## Features

- Real-time face recognition using OpenCV and `face_recognition`.
- Tkinter GUI to control the camera and show status messages.
- Automatic logging of recognized and unknown faces to `data/attendance_logs/attendance_log.json`.
- Snapshots of detected faces saved under `data/snapshots` and `data/unknown_faces_detected`.
- Auto-generated dashboard (`attendance_dashboard.html`) with search, date filtering and CSV/PDF export.
- `organize_project.py` script for arranging the project directories.

## Project Structure

```
backend/              # Python source files (app.py, Face_recognition_demo.py)
frontend/public/      # HTML pages including the attendance dashboard
data/
  known_faces/        # Images of people to recognize
  unknown_faces_detected/  # Snapshots of unrecognized faces
  snapshots/          # Snapshots of recognized faces
  attendance_logs/    # JSON attendance log
```

## Installation

1. Install Python 3.8 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place images of known individuals inside `data/known_faces` before running the app.

## Quick Start (Beginner)
1. Clone or download this repository.
2. Verify Python is installed with `python --version`.
3. Install the requirements as shown above.
4. Add a few images to `data/known_faces` so the system can recognize them.
5. Start the server with:
```bash
python app.py
```
6. Open `http://localhost:5000` in your browser.

## Usage

Run the main application:

```bash
python app.py
```

A Tkinter window will appear allowing you to start/stop the camera. An HTML dashboard is generated automatically and can be opened in a browser to view attendance records.

## License

This project is provided for educational purposes. Use at your own discretion.

