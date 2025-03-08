import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import gdown
from ultralytics import YOLO
from datetime import datetime
from itertools import combinations

# ---------------------------- #
# ✅ YOLO Model Download from Google Drive
# ---------------------------- #
MODEL_PATH = "bestt.pt"
FILE_ID = "1-GINT3-FjNbBz3INmtudoDgDqKzQRBJt"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def download_model():
    """Download the YOLO model from Google Drive if not present."""
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading YOLO model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        st.write("Model downloaded successfully.")
    else:
        st.write("Model already exists.")

# ✅ Ensure the model is available
download_model()

# ✅ Load YOLO model
model = YOLO(MODEL_PATH)

# ---------------------------- #
# ✅ Streamlit UI Configuration
# ---------------------------- #
st.set_page_config(page_title="ADNOC Aerial Insight", layout="wide", initial_sidebar_state="expanded")

# ---------------------------- #
# ✅ Custom Styling
# ---------------------------- #
st.markdown("""
    <style>
        .reportview-container {
            background-color: #0e1117;
        }
        .css-18e3th9 {
            background-color: #0e1117 !important;
        }
        .stTextInput>div>div>input {
            background-color: #262730 !important;
            color: white !important;
        }
        .stFileUploader>div>div>div>button {
            background-color: #0055A4 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------- #
# ✅ ADNOC Branding (Header)
# ---------------------------- #
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/ADNOC-Logo.jpg", width=150)

with col2:
    st.title("ADNOC Aerial Insight: Collision Detection Platform")
    st.write("Upload a video, and the system will analyze object movement, detect potential collisions, and provide a downloadable processed output.")

st.markdown("---")

# ---------------------------- #
# ✅ Video Upload
# ---------------------------- #
st.subheader("Upload a Video")
uploaded_video = st.file_uploader("Drag and drop a file here", type=["mp4", "avi", "mov", "mpg", "mpeg"])

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully!")

    # ---------------------------- #
    # ✅ Collision Detection Logic
    # ---------------------------- #
    st.subheader("Processing Video...")

    # ✅ Define collision detection parameters
    collision_threshold = 50  # Pixel distance for close collisions
    min_velocity_threshold = 5  # Minimum velocity for potential collision

    def calculate_velocity(positions, fps):
        if len(positions) < 2:
            return 0
        (x1, y1), (x2, y2) = positions[-2], positions[-1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance * fps  # pixels per second
        return velocity

    # ✅ Process video with YOLO
    video_source = temp_file.name
    output_video = video_source.replace(".mp4", "_processed.mp4")

    results = model.track(source=video_source, persist=True, save=False, conf=0.3, iou=0.5, tracker="bytetrack.yaml")

    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    object_paths = {}
    object_velocities = {}

    for i, result in enumerate(results):
        frame = result.orig_img
        detections = result.boxes

        for box in detections:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if obj_id not in object_paths:
                object_paths[obj_id] = []
            object_paths[obj_id].append((center_x, center_y))
            object_paths[obj_id] = object_paths[obj_id][-30:]  # Keep last 30 positions

            velocity = calculate_velocity(object_paths[obj_id], fps)
            object_velocities[obj_id] = velocity

            color = (0, 255, 0) if velocity < min_velocity_threshold else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    # ✅ Display Processed Video
    st.subheader("Processed Video")
    st.video(output_video)

    # ✅ Provide Download Link
    with open(output_video, "rb") as file:
        st.download_button(label="Download Processed Video", data=file, file_name="processed_video.mp4", mime="video/mp4")

    st.success("✅ Processing Complete! Download your processed video above.")

