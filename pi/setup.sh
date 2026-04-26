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

read -p "Camera angle (ff = front-facing, dtl = down-the-line) [ff]: " ANGLE
ANGLE="${ANGLE:-ff}"
case "$ANGLE" in
    ff|front-facing)   ANGLE="ff" ;;
    dtl|down-the-line) ANGLE="dtl" ;;
    *) echo "Invalid angle: $ANGLE (must be ff or dtl)"; exit 1 ;;
esac

# Write environment file for the systemd service
sudo tee /etc/default/swing-cam > /dev/null <<EOF
SWINGCAM_SERVER=${SERVER_IP}
SWINGCAM_ANGLE=${ANGLE}
EOF

# Install systemd service
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
sudo cp "${SCRIPT_DIR}/swing-cam-stream.service" /etc/systemd/system/
chmod +x "${REPO_DIR}/swingcam"

# Update the service file with the actual swingcam path
sudo sed -i "s|/home/pi/swing-cam/swingcam|${REPO_DIR}/swingcam|g" /etc/systemd/system/swing-cam-stream.service

sudo systemctl daemon-reload
sudo systemctl enable swing-cam-stream
sudo systemctl start swing-cam-stream

echo ""
echo "=== Setup complete ==="
echo "Stream service is running. Check status with:"
echo "  sudo systemctl status swing-cam-stream"
echo "  journalctl -u swing-cam-stream -f"
