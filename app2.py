import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import streamlit as st
import matplotlib.pyplot as plt


# ==========================
# 1) Autoencoder definition
# ==========================

class CAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # Encoder: 3×112×112 → latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),   # 56×56
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),  # 28×28
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True), # 14×14
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True) # 7×7
        )
        self.flatten = nn.Flatten()
        self.fc_mu   = nn.Linear(256 * 7 * 7, latent_dim)

        # Decoder: latent_dim → 3×112×112
        self.fc_dec  = nn.Linear(latent_dim, 256 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),  # 14×14
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(inplace=True),  # 28×28
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(inplace=True),  # 56×56
            nn.ConvTranspose2d(32,  3,   4, 2, 1), nn.Tanh()               # 112×112 in [-1,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        z = self.fc_mu(self.flatten(h))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 256, 7, 7)
        xhat = self.dec(h)
        return xhat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


# ==========================================
# 2) Utility: transforms, metrics, denormal
# ==========================================

IMG_SIZE = 112

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]."""
    return (x.clamp(-1, 1) + 1) / 2


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """x: (C,H,W) in [0,1] → np.ndarray (H,W,C) in [0,1]."""
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """PSNR = 10 * log10( max_val^2 / MSE )."""
    mse = max(mse, 1e-12)
    return 10.0 * math.log10((max_val ** 2) / mse)


def enhance_pil(img: Image.Image, upscale: int = 2) -> Image.Image:
    """
    Simple enhancement: upscale + unsharp mask.
    No training; pure post-processing.
    """
    w, h = img.size
    new_size = (w * upscale, h * upscale)
    up = img.resize(new_size, Image.LANCZOS)
    # Slight sharpening
    sharp = up.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return sharp


# ===================================
# 3) Model loading with Streamlit cache
# ===================================

@st.cache_resource
def load_model(weights_path: str = "model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAE(latent_dim=128)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ==========================
# 4) Streamlit UI
# ==========================

st.set_page_config(page_title="Face Autoencoder + Enhancement Demo", layout="wide")

st.title("🧠 Convolutional Autoencoder – Face Reconstruction & Enhancement")
st.write(
    "Upload a face image. The trained autoencoder reconstructs it from a low-dimensional "
    "latent vector. Optionally, a simple post-processing enhancement (upscale + sharpening) "
    "improves visual quality **without any extra training**."
)

# Load model once
try:
    model, device = load_model("model.pth")   # your existing weights file
    st.success(f"Autoencoder model loaded on **{device.type.upper()}**")
except Exception as e:
    st.error(
        "Could not load `model.pth`. Make sure it is in the same folder as `app.py` "
        "or update the path in `load_model(...)`."
    )
    st.exception(e)
    st.stop()

st.markdown("---")
st.subheader("1️⃣ Upload an image")

uploaded = st.file_uploader(
    "Choose an image file (preferably a single face) – formats: JPG / JPEG / PNG",
    type=["jpg", "jpeg", "png"]
)

upscale_factor = st.slider("Enhancement upscale factor", min_value=1, max_value=4, value=2, step=1)
apply_enhancement = st.checkbox("Apply enhancement (upscale + sharpen)", value=True)

if uploaded is not None:
    # Read image
    pil_img = Image.open(uploaded).convert("RGB")

    # Run through autoencoder
    with st.spinner("Running autoencoder…"):
        x = preprocess(pil_img).unsqueeze(0).to(device)  # (1,3,112,112)

        with torch.no_grad():
            x_hat, z = model(x)  # x_hat: (1,3,112,112)

        # MSE in training scale [-1,1]
        mse_tensor = F.mse_loss(x_hat, x, reduction="mean")
        mse_val = float(mse_tensor.item())

        # For visualization, move to [0,1]
        x_vis = denorm(x[0])      # (3,H,W)
        xh_vis = denorm(x_hat[0])

        vis_mse = F.mse_loss(x_vis, xh_vis, reduction="mean").item()
        psnr_val = compute_psnr(vis_mse, max_val=1.0)

        # Error map
        err_map = (x_vis - xh_vis).abs().mean(dim=0)  # (H,W)

        # Convert reconstruction to PIL
        recon_pil = Image.fromarray((to_numpy_image(xh_vis) * 255).astype(np.uint8))

        if apply_enhancement:
            enhanced_pil = enhance_pil(recon_pil, upscale=upscale_factor)
        else:
            enhanced_pil = recon_pil

    # --------- Show images ----------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(pil_img, caption="Original (uploaded)", use_column_width=True)

    with c2:
        st.image(recon_pil, caption="Reconstruction (Autoencoder)", use_column_width=True)

    with c3:
        st.image(enhanced_pil,
                 caption=f"Enhanced (upscale ×{upscale_factor}, sharpen)"
                         if apply_enhancement else "Enhanced (disabled)",
                 use_column_width=True)

    st.markdown("---")
    st.subheader("2️⃣ Reconstruction metrics")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("MSE (input vs reconstruction)", f"{mse_val:.6f}")
    with m2:
        st.metric("PSNR (on [0,1])", f"{psnr_val:.2f} dB")

    st.caption(
        "- Smaller MSE → better reconstruction\n"
        "- Larger PSNR → reconstruction closer to the original image"
    )

    st.markdown("---")
    st.subheader("3️⃣ Error heatmap")

    st.write(
        "Mean absolute error between the original and reconstructed images, "
        "averaged over RGB channels."
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(err_map.cpu().numpy(), cmap="inferno")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    st.pyplot(fig)

else:
    st.info("Upload an image to start.")
