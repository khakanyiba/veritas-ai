import streamlit as st
import torch
import numpy as np
import io
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from torchvision import transforms
from model import VeritasCNN

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Veritas AI | Advanced Forensics",
    page_icon="👁️",
    layout="wide" # Changed to wide for better dashboard feel
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 5px; width: 100%; }
    .metric-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VeritasCNN()
    # Load weights
    try:
        model.load_state_dict(torch.load("veritas_model.pth", map_location=device))
    except FileNotFoundError:
        st.error("Model not found. Please run train.py first.")
        st.stop()
    model.eval()
    return model, device

def convert_to_ela_image(path, quality):
    """
    Generates an Error Level Analysis (ELA) image.
    Saves a temp copy at 'quality' % and differences it with original.
    """
    original_image = path.convert('RGB')
    
    # Save to buffer at reduced quality
    buffer = io.BytesIO()
    original_image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved_image = Image.open(buffer)
    
    # Calculate difference
    ela_image = ImageChops.difference(original_image, resaved_image)
    
    # Amplify the difference so humans can see it
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def get_exif_data(image):
    """Extracts metadata from the image (Safely)."""
    # 1. Check if the image format even supports EXIF (PNGs do not)
    if not hasattr(image, '_getexif') or image._getexif() is None:
        return None
    
    exif_data = image._getexif()
    
    readable_exif = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        readable_exif[tag_name] = value
    return readable_exif

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# --- MAIN APP ---

model, device = load_model()

st.title("👁️ Veritas AI | Forensics Lab")
st.caption("Deep Learning based Synthetic Image Detection System")

# Layout: 2 Columns (Left: Upload/Controls, Right: Analysis)
col_control, col_display = st.columns([1, 2])

with col_control:
    st.write("### 1. Evidence Input")
    uploaded_file = st.file_uploader("Upload Suspect Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Evidence #001", use_container_width=True)
        
        st.divider()
        st.write("### 2. Forensic Tools")
        run_scan = st.button("🚀 RUN FULL DIAGNOSTIC")

with col_display:
    if uploaded_file and run_scan:
        with st.spinner('Running Neural Scan & ELA Analysis...'):
            
            # --- AI PREDICTION ---
            img_tensor = process_image(image).to(device)
            with torch.no_grad():
                confidence = model(img_tensor).item()
            
            is_real = confidence > 0.5
            score = confidence * 100 if is_real else (1 - confidence) * 100
            
            # --- DASHBOARD RESULTS ---
            
            # 1. Top Level Verdict
            st.write("### 🔍 Analysis Verdict")
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                if is_real:
                    st.success("✅ AUTHENTIC")
                else:
                    st.error("🚨 ARTIFICIAL")
            
            with result_col2:
                st.metric("Confidence Score", f"{score:.2f}%")
            
            with result_col3:
                st.metric("Inference Device", device.upper())

            st.progress(int(score))
            
            st.divider()

            # 2. ELA Analysis (The New Upgrade)
            st.write("### 🔬 Error Level Analysis (ELA)")
            st.info("Highlights compression anomalies. White glowing edges often indicate tampering or AI generation artifacts.")
            
            ela_img = convert_to_ela_image(image, 90)
            st.image(ela_img, caption="ELA Heatmap (Compression Ghosts)", use_container_width=True)
            
            # 3. Metadata Extraction (The Source Check)
            st.write("### 📂 Metadata Extraction")
            exif = get_exif_data(image)
            
            if exif:
                with st.expander("View EXIF Data"):
                    st.json(exif)
            else:
                st.warning("⚠️ No Metadata found. (Common in AI Generated Images)")

    elif not uploaded_file:
        st.info("👈 Upload an image to begin the investigation.")