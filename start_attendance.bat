@echo off
echo 🔄 Checking Python version...
python --version

echo 🧪 Installing requirements...
python -m pip install -r requirements.txt

echo 🚀 Starting the Smart Attendance System...
python app.py

pause