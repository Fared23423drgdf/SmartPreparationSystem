import os
import cv2
import numpy as np
import face_recognition
import logging

KNOWN_FACES_DIR = os.path.join('data', 'known_faces')
PROCESSED_DIR = os.path.join('data', 'processed')


def load_known_faces(directory=KNOWN_FACES_DIR):
    """Load and encode faces from a directory."""
    encodings = []
    names = []
    if not os.path.exists(directory):
        logging.warning("Known faces directory not found: %s", directory)
        return encodings, names
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            name = os.path.splitext(filename)[0]
            try:
                image = face_recognition.load_image_file(path)
                faces = face_recognition.face_encodings(image)
                if faces:
                    encodings.append(faces[0])
                    names.append(name)
            except Exception as e:
                logging.error("Error loading %s: %s", filename, e)
    return encodings, names


def recognize_faces(image, known_encodings, known_names):
    """Detect and recognize faces in a BGR image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    names = []
    for encoding in encodings:
        name = "Unknown"
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances) > 0:
                best = int(np.argmin(distances))
                if matches[best]:
                    name = known_names[best]
        names.append(name)
    return locations, names


def draw_overlays(image, face_locations, face_names):
    """Draw bounding boxes and labels on the image."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label_bottom = min(bottom, image.shape[0])
        label_top = max(bottom - 35, 0)
        cv2.rectangle(image, (left, label_top), (right, label_bottom), color, cv2.FILLED)
        cv2.putText(
            image,
            name,
            (left + 6, label_bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
    return image


def process_uploaded_image(file_data, filename, known_encodings, known_names):
    """Process uploaded image bytes and save processed result."""
    np_data = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")

    locations, names = recognize_faces(img, known_encodings, known_names)
    processed = draw_overlays(img.copy(), locations, names)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_name = f"processed_{filename}"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, processed)

    summary = [{"name": n} for n in names]
    return out_name, summary

