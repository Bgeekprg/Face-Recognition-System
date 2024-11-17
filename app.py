import re
import os
import cv2
import time
import numpy as np
from PIL import Image
import pickle
import streamlit as st
from deepface import DeepFace
from mtcnn import MTCNN  # Import MTCNN for face detection

# --- Constants ---
FACES_FOLDER = 'Faces/'  # Folder for pre-stored face images
EMBEDDINGS_FILE = 'face_embeddings.pkl'  # Pickle file to store embeddings
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for face matching

# Ensure Faces folder exists
if not os.path.exists(FACES_FOLDER):
    os.makedirs(FACES_FOLDER)

# --- Helper Functions ---

def load_face_embeddings(regenerate=False):
    """Load face embeddings from pickle file or initialize an empty dictionary."""
    embeddings = {}
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
    if regenerate:
        embeddings = regenerate_embeddings(embeddings)
        save_face_embeddings(embeddings)
    return embeddings

def regenerate_embeddings(existing_embeddings):
    """Regenerate embeddings for the existing faces in the folder."""
    for image_file in os.listdir(FACES_FOLDER):
        if image_file.endswith('.jpg'):
            name_prefix = image_file.split('_')[0]  # Use the first part of the filename as name
            image_path = os.path.join(FACES_FOLDER, image_file)
            generate_and_save_embeddings(image_path, name_prefix, existing_embeddings)
    return existing_embeddings

def save_face_embeddings(embeddings):
    """Save the embeddings to a pickle file."""
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

def generate_and_save_embeddings(image_path, name, existing_embeddings=None):
    """Generate embeddings from an image and save them in the embeddings dictionary."""
    try:
        embedding = DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)
        embedding_value = embedding[0]['embedding']
        
        embeddings = existing_embeddings if existing_embeddings else load_face_embeddings()
        if name not in embeddings:
            embeddings[name] = []
        
        if embedding_value not in embeddings[name]:
            embeddings[name].append(embedding_value)
        
        save_face_embeddings(embeddings)
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")

def get_next_filename(name_prefix):
    """Get the next available filename with incremental numbering."""
    existing_files = [f for f in os.listdir(FACES_FOLDER) if f.startswith(name_prefix)]
    existing_numbers = [
        int(re.match(rf'{re.escape(name_prefix)}_(\d+)\.jpg', file).group(1)) 
        for file in existing_files if re.match(rf'{re.escape(name_prefix)}_(\d+)\.jpg', file)
    ]
    next_number = max(existing_numbers, default=0) + 1
    return f"{name_prefix}_{next_number}.jpg"

def capture_image(name_prefix):
    """Capture an image from the webcam and save it with the incremental name."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to capture image.")
        return None

    new_filename = get_next_filename(name_prefix)
    image_path = os.path.join(FACES_FOLDER, new_filename)
    img = Image.fromarray(frame)
    img.save(image_path)
    return image_path

def extract_faces_from_frame(frame):
    """Extract faces from a frame using MTCNN."""
    detector = MTCNN()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = detector.detect_faces(rgb_frame)
        return faces
    except Exception:
        return []

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings using DeepFace."""
    try:
        result = DeepFace.verify(embedding1, embedding2, model_name='VGG-Face')
        return result['distance']
    except Exception:
        return 1.0  # High distance for error

# --- Webcam Stream and Face Recognition Logic ---
def start_face_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    video_placeholder = st.empty()

    while st.session_state.run_stream:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        faces = extract_faces_from_frame(frame)

        if not faces:
            st.warning("No faces detected. Please adjust the camera and try again.")

        for detected_face in faces:
            x, y, w, h = detected_face['box']
            detected_embedding = DeepFace.represent(frame[y:y+h, x:x+w], model_name='VGG-Face', enforce_detection=False)
            matched_name = None
            min_similarity = float('inf')
            embeddings = load_face_embeddings()
            for name, stored_embeddings in embeddings.items():
                for stored_embedding in stored_embeddings:
                    similarity = compare_embeddings(detected_embedding[0]['embedding'], stored_embedding)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        matched_name = name

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{matched_name}" if matched_name and min_similarity < SIMILARITY_THRESHOLD else "No Match"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_bgr)
        video_placeholder.image(img, channels="RGB", use_container_width=True)
        time.sleep(0.05)

    cap.release()

# --- Streamlit UI and App Logic ---
if 'run_stream' not in st.session_state:
    st.session_state.run_stream = False

# Initialize captured_faces if not already in session_state
if 'captured_faces' not in st.session_state:
    st.session_state.captured_faces = []

st.session_state.face_embeddings = load_face_embeddings()
st.title("Live Face Recognition with Pre-stored Faces")

# UI Buttons
start_button = st.button("Start Webcam", disabled=st.session_state.run_stream)
stop_button = st.button("Stop Webcam", disabled=not st.session_state.run_stream)
train_button = st.button("Train New Face")
capture_button = st.button("Capture Image", disabled=not st.session_state.run_stream)
regenerate_embeddings_button = st.button("Regenerate Face Embeddings")

with st.form(key='train_form'):
    name = st.text_input("Enter your name for image capture:")
    submit_button = st.form_submit_button("Set Name")

if start_button:
    st.session_state.run_stream = True
    st.text("Press 'Stop Webcam' to quit.")
    start_face_recognition()

if stop_button:
    st.session_state.run_stream = False
    st.text("Webcam is stopped. Press 'Start Webcam' to begin.")

# Handle capturing of new images
if capture_button and name:
    image_path = capture_image(name)
    if image_path:
        # Append the captured image path to the list in session_state
        st.session_state.captured_faces.append(image_path)
        
        st.success(f"Image captured and saved as {os.path.basename(image_path)}")
        # Generate and save embeddings for the new face
        generate_and_save_embeddings(image_path, name)
        st.image(image_path, caption=f"Captured Image for {name}", use_container_width=True)
    else:
        st.error("Please enter your name before capturing an image.")

# Handle training of new faces
if train_button:
    if st.session_state.captured_faces:
        for captured_image in st.session_state.captured_faces:
            generate_and_save_embeddings(captured_image, name)
        st.success("Training completed! New faces have been added to embeddings.")
    else:
        st.error("No images captured for training.")

# Handle regenerating embeddings
if regenerate_embeddings_button:
    st.session_state.face_embeddings = load_face_embeddings(regenerate=True)
    st.success("Embeddings have been regenerated and saved!")
