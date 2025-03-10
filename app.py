import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request
from ultralytics import YOLO

# Set Streamlit page config
st.set_page_config(page_title="ADNOC Aerial Insight", layout="wide")

# Ensure YOLO model is available
MODEL_PATH = "bestt.pt"
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1-GINT3-FjNbBz3INmtudoDgDqKzQRBJt"

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading missing YOLO model (bestt.pt)... This may take a moment.")
    urllib.request.urlretrieve(GDRIVE_MODEL_URL, MODEL_PATH)

# Sidebar with ADNOC branding
LOGO_PATH = "adnoc_logo.jpg"
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_column_width=True)
else:
    st.sidebar.warning("ADNOC logo missing!")

st.sidebar.markdown(
    "### Developed as part of ADNOC's Analytics & Data Science Initiative."
)

# Main title
