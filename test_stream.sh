#!/usr/bin/env bash
#
# Sends a synthetic video stream to the swing-cam server for testing
# without a Raspberry Pi. Requires ffmpeg.
#
# Usage:
#   ./test_stream.sh [server_host] [port]
#
# Simulates swing-like motion: 8 seconds of stillness, then a fast
# burst of motion (~0.5s), then stillness again. This pattern matches
# what the multi-stage detector expects (still → spike → still).

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
echo "Pattern: 8s still → 0.5s motion burst → repeat"
echo "This should trigger the swing detector every ~10 seconds."
echo ""
echo "Make sure the server is running first."
echo "Press Ctrl+C to stop."
echo ""

# Generate a pattern that simulates a golf swing:
# - Static background (still) for 8 seconds
# - Fast-moving object (swing burst) for 0.5 seconds
# - Repeat
#
# Uses drawbox with time-based enable to create motion bursts:
# A box sweeps across the frame rapidly during the burst window,
# then disappears. The 'between' function enables it for 0.5s
# every 10 seconds.
ffmpeg \
    -re \
    -f lavfi \
    -i "color=c=0x1a1a1a:s=1280x720:r=30:d=3600,
        drawbox=x='if(between(mod(t,10),8,8.5), (mod(t,10)-8)*2560, -200)':
              y=200:w=120:h=320:c=white:t=fill:
              enable='between(mod(t,10),8,8.5)',
        drawbox=x=500:y=600:w=280:h=40:c=0x333333:t=fill" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -g 30 \
    -f h264 \
    "tcp://${SERVER_HOST}:${SERVER_PORT}"
