#!/usr/bin/env bash
#
# Sends a synthetic video stream to the swing-cam server for testing
# without a Raspberry Pi. Requires ffmpeg.
#
# Usage:
#   ./test_stream.sh [server_host] [port]
#
# The stream uses the "life" source (Conway's Game of Life) which produces
# organic bursts of motion — good for triggering swing detection.

set -euo pipefail

SERVER_HOST="${1:-localhost}"
SERVER_PORT="${2:-9556}"

if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Install it first."
    exit 1
fi

echo "==================================="
echo "  swing-cam test stream"
echo "==================================="
echo "Target: ${SERVER_HOST}:${SERVER_PORT}"
echo ""
echo "Make sure the server is running first (cd server && python main.py)"
echo "Press Ctrl+C to stop."
echo ""

# "life" generates Conway's Game of Life — creates bursts of pixel changes
# that look like motion events, which is ideal for testing the detector.
# The stream runs at 1280x720@30fps to match the Pi camera config.
ffmpeg \
    -re \
    -f lavfi \
    -i "life=size=320x180:rate=30:rule=S23/B3:random_seed=42,scale=1280x720" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -g 30 \
    -f h264 \
    "tcp://${SERVER_HOST}:${SERVER_PORT}"
