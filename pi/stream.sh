#!/bin/bash
# swing-cam Pi Zero streaming script
# Streams H.264 video over TCP to the server.
#
# Usage: ./stream.sh [server_ip] [port]
#
# The Pi's hardware encoder handles H.264 so CPU usage stays low.
# 720p @ 30fps is the safe default for Pi Zero W. If you have a
# Pi Zero 2 W, you can bump to 60fps.

SERVER_IP="${1:-192.168.1.100}"
PORT="${2:-8888}"
CAMERA_ID="${CAMERA_ID:-cam1}"

# --- Tunable parameters ---
WIDTH=1280
HEIGHT=720
FRAMERATE=30          # 30 for Pi Zero W, 60 for Pi Zero 2 W
BITRATE=4000000       # 4 Mbps — good balance of quality vs bandwidth
KEYFRAME_PERIOD=1000  # Keyframe every 1s (in ms) — helps with clip cutting

echo "swing-cam streamer starting"
echo "  Camera ID:   ${CAMERA_ID}"
echo "  Resolution:  ${WIDTH}x${HEIGHT} @ ${FRAMERATE}fps"
echo "  Bitrate:     $((BITRATE / 1000)) kbps"
echo "  Streaming to: ${SERVER_IP}:${PORT}"

# Retry loop — if the server isn't up yet, keep trying
while true; do
    echo "Connecting to ${SERVER_IP}:${PORT}..."

    rpicam-vid \
        -t 0 \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --framerate "$FRAMERATE" \
        --bitrate "$BITRATE" \
        --codec h264 \
        --profile high \
        --level 4.1 \
        --inline \
        --flush \
        --keyframe "$KEYFRAME_PERIOD" \
        -o "tcp://${SERVER_IP}:${PORT}"

    EXIT_CODE=$?
    echo "Stream ended (exit code: ${EXIT_CODE}). Retrying in 5s..."
    sleep 5
done
