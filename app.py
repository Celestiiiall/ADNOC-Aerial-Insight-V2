import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Set Streamlit page configuration
st.set_page_config(
    page_title="ADNOC Aerial Insight",
    page_icon="🛰️",
    layout="wide"
)

# Define paths for assets
LOGO_SIDEBAR = "assets/ADNOC-Logo.jpg"
LOGO_TOP = "assets/ADNOC-Logo.png"

# Sidebar with ADNOC branding
with st.sidebar:
    st.image(LOGO_SIDEBAR, width=200)
    st.markdown(
        '<p style="text-align:center; font-size:14px;">Developed as part of ADNOC’s Analytics & Data Science Initiative.</p>',
        unsafe_allow_html=True
    )

# Main Title & Header
st.image(LOGO_TOP, use_container_width=True)  # Top Logo Banner
st.markdown("## ADNOC Aerial Insight: Collision Detection Platform")
st.write(
    "Upload a video, and the system will analyze object movement, detect potential collisions, "
    "and provide a downloadable processed output."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a Video", type=["mp4", "avi", "mov", "mpg", "mpeg"]
)

# Load YOLO model (Ensure the model is in the same directory or accessible)
MODEL_PATH = "bestt.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Error: YOLO model file not found. Ensure `bestt.pt` is uploaded.")
    st.stop()

model = YOLO(MODEL_PATH)

# Collision Detection Parameters
collision_threshold = 50  # Pixel distance for close collisions
min_velocity_threshold = 5  # Minimum velocity for potential collision
frame_buffer = 10  # Number of frames to consider for trajectory

# Process Uploaded Video
if uploaded_file is not None:
    st.video(uploaded_file)

    # Save file temporarily
    video_ext = uploaded_file.name.split('.')[-1]
    temp_video_path = f"temp_video.{video_ext}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded successfully! Processing...")

    # Open video for processing
    cap = cv2.VideoCapture(temp_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Output video path
    output_video_path = f"processed_{uploaded_file.name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    object_paths = {}
    object_velocities = {}

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.track(source=frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")

        detections = results[0].boxes  # Get detected objects

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            obj_id = int(box.id[0]) if box.id is not None else None

            if obj_id is not None:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Store trajectory
                if obj_id not in object_paths:
                    object_paths[obj_id] = []
                object_paths[obj_id].append((center_x, center_y))

                # Limit trajectory history
                object_paths[obj_id] = object_paths[obj_id][-frame_buffer:]

                # Draw trajectory
                for j in range(1, len(object_paths[obj_id])):
                    alpha = j / len(object_paths[obj_id])
                    color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                    cv2.line(frame, object_paths[obj_id][j - 1], object_paths[obj_id][j], color, 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()

    # Provide download link
    with open(output_video_path, "rb") as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name=output_video_path,
            mime="video/mp4"
        )

    st.success("Processing complete! Download your video above.")

