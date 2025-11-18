import math
from io import BytesIO

import numpy as np
from PIL import Image

import cv2
import torch
import streamlit as st

from gfpgan import GFPGANer


# ==========================
# 1) Load pretrained GFPGAN
# ==========================

@st.cache_resource
def load_gfpgan(upscale: int = 2):
    """
    Load a pretrained GFPGAN model.
    - No training, just inference.
    - On first run, GFPGAN will download GFPGANv1.4 weights automatically.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    restorer = GFPGANer(
        model_path='GFPGANv1.4.pth',   # auto-downloaded if not present
        upscale=upscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,             # no RealESRGAN (simpler for Streamlit)
        device=device
    )
    return restorer, device


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ==========================
# 2) Streamlit UI
# ==========================

st.set_page_config(page_title="GFPGAN Face Restoration Demo", layout="wide")

st.title("🧠 GFPGAN Face Restoration – Pretrained, No Training Needed")
st.write(
    "Upload a face image. The pretrained GFPGAN model will restore and enhance the face "
    "and optionally upscale the whole image. No dataset or training from your side is "
    "required – this uses fully pretrained weights."
)

upscale_factor = st.slider(
    "Upscale factor (GFPGAN built-in)",
    min_value=1,
    max_value=4,
    value=2,
    step=1
)

# Try to load model
try:
    restorer, device = load_gfpgan(upscale=upscale_factor)
    st.success(f"GFPGAN model loaded on **{device.type.upper()}**")
except Exception as e:
    st.error(
        "Could not load GFPGAN model. Make sure `gfpgan` is installed and internet "
        "access is available for downloading the pretrained weights (GFPGANv1.4)."
    )
    st.exception(e)
    st.stop()

st.markdown("---")
st.subheader("1️⃣ Upload an image")

uploaded = st.file_uploader(
    "Choose an image file (preferably containing one or more faces) – JPG / JPEG / PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # Read image as PIL
    pil_img = Image.open(uploaded).convert("RGB")
    w, h = pil_img.size

    # Convert PIL → OpenCV BGR
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Running GFPGAN face restoration…"):
        # has_aligned=False → detect face(s) in the image
        # only_center_face=False → restore all faces
        # paste_back=True → put restored face(s) back into original image
        restored_img, _, _ = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

    # Convert back to RGB PIL
    restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    restored_pil = Image.fromarray(restored_rgb)

    # ========== Show side-by-side ==========
    c1, c2 = st.columns(2)

    with c1:
        st.image(
            pil_img,
            caption=f"Original (uploaded) – {w}×{h}",
            use_column_width=True
        )
    with c2:
        st.image(
            restored_pil,
            caption=f"Restored / Enhanced by GFPGAN – upscale ×{upscale_factor}",
            use_column_width=True
        )

    st.markdown("---")
    st.subheader("2️⃣ Download enhanced image")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download PNG (restored)",
            data=pil_to_bytes(restored_pil, "PNG"),
            file_name="restored_gfpgan.png",
            mime="image/png"
        )
    with dl2:
        st.download_button(
            label="Download JPEG (restored)",
            data=pil_to_bytes(restored_pil, "JPEG"),
            file_name="restored_gfpgan.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload a face image to start GFPGAN restoration.")
