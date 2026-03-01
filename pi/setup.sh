#!/bin/bash
# Pi Zero setup script for swing-cam
set -e

echo "=== swing-cam Pi Setup ==="

# Ensure camera is enabled
if ! rpicam-hello --list-cameras 2>/dev/null | grep -q "Available"; then
    echo "WARNING: No camera detected. Make sure:"
    echo "  1. The camera ribbon cable is connected properly"
    echo "  2. The camera interface is enabled (sudo raspi-config)"
    echo ""
fi

# Prompt for config
read -p "Server IP address [192.168.1.100]: " SERVER_IP
SERVER_IP="${SERVER_IP:-192.168.1.100}"

read -p "Server port [8888]: " PORT
PORT="${PORT:-8888}"

read -p "Camera ID (e.g. face-on, dtl) [cam1]: " CAMERA_ID
CAMERA_ID="${CAMERA_ID:-cam1}"

# Write environment file for the systemd service
sudo tee /etc/default/swing-cam > /dev/null <<EOF
SERVER_IP=${SERVER_IP}
PORT=${PORT}
CAMERA_ID=${CAMERA_ID}
EOF

# Install systemd service
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sudo cp "${SCRIPT_DIR}/swing-cam-stream.service" /etc/systemd/system/
chmod +x "${SCRIPT_DIR}/stream.sh"

# Update the service file with the actual script path
sudo sed -i "s|/home/pi/swing-cam|${SCRIPT_DIR}|g" /etc/systemd/system/swing-cam-stream.service

sudo systemctl daemon-reload
sudo systemctl enable swing-cam-stream
sudo systemctl start swing-cam-stream

echo ""
echo "=== Setup complete ==="
echo "Stream service is running. Check status with:"
echo "  sudo systemctl status swing-cam-stream"
echo "  journalctl -u swing-cam-stream -f"
