import streamlit as st
import cv2
import numpy as np
import os
import gdown
from ultralytics import YOLO
from pathlib import Path

# ✅ Set up model download
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with actual Google Drive file ID
MODEL_PATH = "bestt.pt"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... This may take a moment.")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Load YOLO model
model = YOLO(MODEL_PATH)

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="ADNOC Aerial Insight", layout="wide", page_icon="🛡️")

# ✅ Sidebar with ADNOC Logo
st.sidebar.image("assets/adnoc-logo.jpg", use_column_width=True)
st.sidebar.markdown("**Developed as part of ADNOC's Analytics & Data Science Initiative.**")

# ✅ Header with ADNOC Banner
st.image("assets/adnoc-banner.jpg", use_column_width=True)
st.title("ADNOC Aerial Insight: Collision Detection Platform")
st.write("Upload a video, and the system will analyze object movement, detect collisions, and allow downloading the processed output.")

# ✅ File Upload
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mpeg", "mpg"])

# ✅ Process Uploaded Video
if uploaded_video is not None:
    # Save uploaded video
    video_path = Path("uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)  # Show uploaded video

    # Process video with YOLO + ByteTrack
    st.info("Processing video... This may take some time.")
    
    results = model.track(source=str(video_path), persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")

    # ✅ Save processed video
    processed_video_path = "processed_output.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    for result in results:
        frame = result.orig_img
        out.write(frame)
    
    cap.release()
    out.release()

    # ✅ Show Processed Video
    st.video(processed_video_path)

    # ✅ Download Link
    st.download_button(label="Download Processed Video", data=open(processed_video_path, "rb").read(), file_name="collision_output.mp4")

st.success("Upload a video to get started.")
