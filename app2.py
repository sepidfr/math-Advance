import math
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import streamlit as st
import matplotlib.pyplot as plt


# ======================================================
# UNet-style Autoencoder (same as in training)
# ======================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if necessary
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetAutoencoder(nn.Module):
    def __init__(self, img_channels: int = 3):
        super().__init__()

        self.enc1 = ConvBlock(img_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)

        self.final_conv = nn.Conv2d(64, img_channels, kernel_size=1)
        self.act_out = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        xb = self.bottleneck(self.pool4(x4))

        x = self.up4(xb, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.final_conv(x)
        x = self.act_out(x)
        return x


# ======================================================
# Utilities
# ======================================================

IMG_SIZE = 112

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1]."""
    return (x.clamp(-1, 1) + 1) / 2


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) in [0,1] -> numpy (H,W,C) in [0,1]."""
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    mse = max(mse, 1e-12)
    return 10.0 * math.log10((max_val ** 2) / mse)


# ======================================================
# Load model (Streamlit cache)
# ======================================================

@st.cache_resource
def load_model(weights_path: str = "unet_ae.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetAutoencoder(img_channels=3)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(page_title="UNet Autoencoder Face Demo", layout="wide")

st.title("🧠 UNet-style Convolutional Autoencoder – Face Reconstruction Demo")
st.write(
    "Upload a face image. The trained UNet autoencoder will reconstruct it. "
    "You will see the original vs reconstruction, plus MSE, PSNR and an error heatmap."
)

try:
    model, device = load_model("unet_ae.pth")
    st.success(f"Model loaded on **{device.type.upper()}**")
except Exception as e:
    st.error(
        "Could not load `unet_ae.pth`. Make sure it is in the same folder as `app.py`, "
        "or update the path in `load_model()`."
    )
    st.exception(e)
    st.stop()

st.markdown("---")
st.subheader("1️⃣ Upload an image")

uploaded = st.file_uploader(
    "Choose an image file (preferably a single face) – JPG / JPEG / PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption="Original (uploaded)", use_column_width=True)

    with st.spinner("Running UNet autoencoder…"):
        x = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            x_hat = model(x)

        # MSE in normalized space [-1,1]
        mse_tensor = F.mse_loss(x_hat, x, reduction="mean")
        mse_val = float(mse_tensor.item())

        # For visualization, move to [0,1]
        x_vis = denorm(x[0])
        xh_vis = denorm(x_hat[0])

        vis_mse = F.mse_loss(x_vis, xh_vis, reduction="mean").item()
        psnr_val = compute_psnr(vis_mse, max_val=1.0)

        err_map = (x_vis - xh_vis).abs().mean(dim=0)

    with col2:
        st.image(
            to_numpy_image(xh_vis),
            caption="Reconstruction (UNet AE)",
            use_column_width=True,
        )

    st.markdown("---")
    st.subheader("2️⃣ Reconstruction metrics")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("MSE (input vs reconstruction)", f"{mse_val:.6f}")
    with c2:
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
