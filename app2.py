import math
from io import BytesIO

import numpy as np
from PIL import Image

import torch
import streamlit as st

# We use the Python wrapper for Real-ESRGAN
from realesrgan import RealESRGAN


# ==========================
# 1) Load pretrained model
# ==========================

@st.cache_resource
def load_realesrgan(weights_path: str = "RealESRGAN_x4plus.pth", scale: int = 4):
    """
    Load a pretrained Real-ESRGAN model.
    No training, just inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=scale)
    # This expects the weights file to be present in the same folder as app.py
    model.load_weights(weights_path, download=False)
    return model, device


# ==========================
# 2) Small utility helpers
# ==========================

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ==========================
# 3) Streamlit UI
# ==========================

st.set_page_config(page_title="Real-ESRGAN Super-Resolution Demo", layout="wide")

st.title("🧠 Real-ESRGAN Super-Resolution – No Training Needed")
st.write(
    "Upload a low-resolution image. A pretrained Real-ESRGAN model will upscale it "
    "by a factor of 4× and enhance details. No training or dataset from your side "
    "is required – we only run inference with a pretrained model."
)

# Try to load model
try:
    model, device = load_realesrgan("RealESRGAN_x4plus.pth", scale=4)
    st.success(f"Real-ESRGAN model loaded on **{device.type.upper()}**")
except Exception as e:
    st.error(
        "Could not load `RealESRGAN_x4plus.pth`.\n\n"
        "- Make sure the weight file is in the same folder as `app.py`, or\n"
        "- Update the path in `load_realesrgan()`.\n\n"
        "Once the weights are in place, rerun the app."
    )
    st.exception(e)
    st.stop()

st.markdown("---")
st.subheader("1️⃣ Upload an image")

uploaded = st.file_uploader(
    "Choose an image file (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

scale_factor = st.slider("Upscale factor", min_value=2, max_value=4, value=4, step=1)

if uploaded is not None:
    # Read image
    pil_img = Image.open(uploaded).convert("RGB")

    # Optionally downscale very large images before SR (to save time)
    max_side = 512
    w, h = pil_img.size
    if max(w, h) > max_side:
        ratio = max_side / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption=f"Original (resized if too large) – {pil_img.size}", use_column_width=True)

    # Run Real-ESRGAN
    with st.spinner(f"Running Real-ESRGAN ×{scale_factor}..."):
        # The Python wrapper always uses its internal scale (e.g. 4),
        # but we can instantiate a new model if the scale slider changes.
        if scale_factor != 4:
            # Reload model with different scale if user changed it
            model_dyn, _ = load_realesrgan("RealESRGAN_x4plus.pth", scale=scale_factor)
        else:
            model_dyn = model

        sr_img = model_dyn.predict(pil_img)   # PIL.Image output

    with col2:
        st.image(sr_img, caption=f"Super-resolved ×{scale_factor} – {sr_img.size}", use_column_width=True)

    st.markdown("---")
    st.subheader("2️⃣ Download enhanced image")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download PNG (enhanced)",
            data=pil_to_bytes(sr_img, "PNG"),
            file_name="enhanced_x{}_realesrgan.png".format(scale_factor),
            mime="image/png"
        )
    with dl_col2:
        st.download_button(
            label="Download JPEG (enhanced)",
            data=pil_to_bytes(sr_img, "JPEG"),
            file_name="enhanced_x{}_realesrgan.jpg".format(scale_factor),
            mime="image/jpeg"
        )

else:
    st.info("Upload an image to start the super-resolution demo.")
