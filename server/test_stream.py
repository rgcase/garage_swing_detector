"""
Synthetic test stream for the swing-cam server.

Generates frames with a clear still→burst→still pattern that exercises
the multi-stage swing detector (motion gate + optical flow + duration
filter). Pipes the frames through FFmpeg for H.264 + TCP delivery.

Invoked via the swingcam CLI: `swingcam test [host] [port]`
"""

import math
import subprocess
import sys
import time

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 30

CYCLE_SECONDS = 10.0   # full cycle (still + burst + still)
BURST_START = 8.0      # when in the cycle the burst begins
BURST_DURATION = 0.4   # length of the motion burst


def make_frame(t: float) -> np.ndarray:
    """Build a single frame at time t (seconds since start)."""
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

    # Static "ground" stripe — gives the still phase a stable reference
    cv2.rectangle(frame, (440, 600), (840, 640), (70, 70, 70), -1)
    cv2.putText(
        frame, "swing-cam test", (530, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2,
    )

    cycle_pos = t % CYCLE_SECONDS
    in_burst = BURST_START <= cycle_pos < BURST_START + BURST_DURATION

    if in_burst:
        # Fast arc-shaped sweep across the frame — simulates club + arms
        progress = (cycle_pos - BURST_START) / BURST_DURATION  # 0..1
        x = int(progress * (WIDTH + 200) - 100)
        y_arc = int(420 - 120 * math.sin(progress * math.pi))
        cv2.rectangle(
            frame, (x - 30, y_arc - 200), (x + 30, y_arc + 50),
            (255, 255, 255), -1,
        )
        cv2.line(
            frame, (x - 80, y_arc - 200), (x + 30, y_arc - 200),
            (180, 180, 180), 8,
        )

    return frame


def run(host: str, port: str):
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", "30",
        "-f", "h264",
        f"tcp://{host}:{port}",
    ]

    print(f"swing-cam test stream → tcp://{host}:{port}")
    print(
        f"Pattern: {BURST_START:.0f}s still → "
        f"{BURST_DURATION*1000:.0f}ms motion burst → repeat every {CYCLE_SECONDS:.0f}s"
    )
    print("First swing should detect ~9s after frames start arriving.")
    print("Press Ctrl+C to stop.\n", flush=True)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    start = time.time()
    frame_idx = 0

    try:
        while True:
            t = frame_idx / FPS
            frame = make_frame(t)
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("\nFFmpeg disconnected.")
                break

            frame_idx += 1

            # Pace to real-time so the server gets frames at FPS rate
            expected = start + frame_idx / FPS
            now = time.time()
            if expected > now:
                time.sleep(expected - now)

            # Status print at the start of each burst
            cycle_pos = t % CYCLE_SECONDS
            if abs(cycle_pos - BURST_START) < 1 / FPS:
                print(f"[{int(t)}s] BURST", flush=True)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = sys.argv[2] if len(sys.argv) > 2 else "9556"
    run(host, port)
