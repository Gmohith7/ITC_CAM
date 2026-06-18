#!/bin/bash
# install.sh — One-shot setup for Raspberry Pi 5 running Bookworm (64-bit)
# Run from the project root:  bash setup/install.sh
set -e

echo "[Setup] Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "[Setup] Installing system-level dependencies..."
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    python3-picamera2 \
    python3-lgpio lgpio \
    libatlas-base-dev \
    libjpeg-dev

# NOTE: RPi.GPIO is intentionally NOT installed — it does not support Pi 5 (RP1 chip).
# NOTE: python3-opencv from apt is fine for display; pip version used in venv for latest features.

echo "[Setup] Creating Python virtual environment (with system picamera2 visible)..."
# --system-site-packages lets the venv see the apt-installed picamera2 / libcamera bindings
# which cannot be installed via pip on aarch64.
python3 -m venv --system-site-packages venv
source venv/bin/activate

echo "[Setup] Upgrading pip..."
python3 -m pip install --upgrade pip

echo "[Setup] Installing pip packages..."
pip install numpy Pillow flask python-dotenv rapidocr_onnxruntime gpiozero opencv-python

echo ""
echo "[Setup] Done."
echo "  Activate : source venv/bin/activate"
echo "  Detect   : python detection/detector.py"
echo "  Dashboard: python run.py --dashboard"
echo ""
echo "[Setup] Verify camera:"
echo "  rpicam-hello --list-cameras"
