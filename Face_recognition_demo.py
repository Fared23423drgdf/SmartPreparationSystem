import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import time
import logging
from werkzeug.utils import secure_filename

# Configure logging to console and file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_debug.log"),
                        logging.StreamHandler()
                    ])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Directories - Updated paths based on the new structure
KNOWN_FACES_DIR = '../data/known_faces'
UPLOADS_DIR = '../uploads' # Still useful for temporary storage if needed, but SocketIO handler saves directly
PROCESSED_DIR = '../processed' # This directory might need to be reconsidered if processed images are served from frontend/public

# Ensure directories exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """
    Loads known faces and their encodings from the known_faces directory.
    Each image filename should be the person's name (e.g., 'John_Doe.jpg').
    Handles potential errors during image loading and encoding.
    """
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    logging.info("Loading known faces...")
    if not os.path.exists(KNOWN_FACES_DIR):
        logging.warning(f"Known faces directory not found: {KNOWN_FACES_DIR}")
        return

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                logging.info(f"Attempting to load image: {image_path}")
                image = face_recognition.load_image_file(image_path)
                logging.info(f"Image loaded successfully: {image_path}")
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    logging.info(f"Loaded face encoding for: {name}")
                else:
                    logging.warning(f"No face found in {filename}. Skipping.")
            except Exception as e:
                logging.error(f"Error loading face from {filename}: {e}", exc_info=True)
    logging.info(f"Finished loading {len(known_face_names)} known faces.")
    if not known_face_names:
        logging.warning("No known faces were loaded. All detected faces will be marked as 'Unknown'.")

# Load known faces on startup
load_known_faces()

@socketio.on('connect')
def handle_connect():
    """Handles client connection."""
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnection."""
    logging.info('Client disconnected')

@socketio.on('upload_image')
def handle_image_upload(data):
    """
    Handles single image upload and processing via SocketIO.
    This function will now be called for each image in the batch from the frontend.
    """
    logging.info("Received single image upload request via SocketIO")
    file_data = data.get('file')
    filename = data.get('filename')

    if not file_data or not filename:
        logging.error("Invalid image upload data received via SocketIO.")
        emit('error', {'message': 'Invalid file data received.'})
        return

    safe_filename = secure_filename(filename)
    temp_filepath = None

    try:
        logging.info(f"Processing image: {safe_filename}")

        image_np = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
             logging.error(f"Could not decode image data for {safe_filename}. Invalid image format?")
             raise ValueError("Could not decode image data.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        logging.info(f"Image loaded for processing: {safe_filename}")

        logging.info(f"Finding face locations and encodings in {safe_filename}...")
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        logging.info(f"Found {len(face_locations)} face(s) in {safe_filename}.")

        recognized_faces_summary = []

        if face_locations:
            for i in range(len(face_locations)):
                top, right, bottom, left = face_locations[i]
                face_encoding = face_encodings[i]

                name = "Unknown"
                color = (0, 0, 255)

                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        color = (0, 255, 0)
                        recognized_faces_summary.append({"name": name, "timestamp": "N/A (Image)"})


                cv2.rectangle(process_image, (left, top), (right, bottom), color, 2)

                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)[0]
                label_bottom = min(bottom, process_image.shape[0])
                label_top = max(bottom - 35, 0)
                cv2.rectangle(process_image, (left, label_top), (right, label_bottom), color, cv2.FILLED)
                cv2.putText(process_image, name, (left + 6, label_bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        else:
            logging.info(f"No faces detected in image: {safe_filename}")


        processed_filename = f"processed_{safe_filename}"
        processed_filepath = os.path.join(PROCESSED_DIR, processed_filename)
        logging.info(f"Saving processed image to: {processed_filepath}")
        cv2.imwrite(processed_filepath, process_image)
        logging.info(f"Saved processed image: {processed_filepath}")

        # Emit the processed image URL and summary (including the original filename)
        # The URL needs to be relative to the frontend/public directory
        # Assuming the Flask app serves from the root, and processed images are directly accessible
        # from the root or a specific route. For the new structure, we'll serve from /processed/
        processed_url = f"/processed/{processed_filename}"
        logging.info(f"Emitting image_processed event for {safe_filename} with URL: {processed_url}")
        emit('image_processed', {'url': processed_url, 'summary': recognized_faces_summary, 'filename': filename})
        logging.info(f"Emitted image_processed event for {safe_filename}")

    except Exception as e:
        logging.error(f"Error processing image {safe_filename}: {e}", exc_info=True)
        emit('error', {'message': f'Error processing image {filename}: {e}'})
    finally:
         if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logging.info(f"Cleaned up temporary uploaded file: {temp_filepath}")
            except OSError as cleanup_error:
                logging.error(f"Error cleaning up temporary uploaded file {temp_filepath}: {cleanup_error}", exc_info=True)


@app.route('/processed/<filename>')
def serve_processed_file(filename):
    """Serves processed files from the processed directory."""
    logging.info(f"Serving processed file: {filename}")
    try:
        # Ensure PROCESSED_DIR is absolute or relative to the app.py location
        # If app.py is in backend/, then PROCESSED_DIR needs to be relative to the root
        # or an absolute path. Let's make it absolute for clarity.
        abs_processed_dir = os.path.abspath(PROCESSED_DIR)
        return send_from_directory(abs_processed_dir, filename)
    except FileNotFoundError:
        logging.error(f"Processed file not found: {filename}", exc_info=True)
        return "File not found", 404
    except Exception as e:
        logging.error(f"Error serving processed file {filename}: {e}", exc_info=True)
        return "Error serving file", 500


@app.route('/')
def index():
    """Serves the main index.html file."""
    # Assuming Face_recognition_demo.html is now in frontend/public/
    # The Flask app needs to know where to find this.
    # We can either serve it from a specific route or change the root path.
    # For simplicity, let's serve it from the 'frontend/public' directory.
    return send_from_directory('../frontend/public', 'Face_recognition_demo.html')

if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server on 127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
