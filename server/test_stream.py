"""
Synthetic test stream for the swing-cam server.

Generates frames with a clear still→burst→still pattern that exercises
the multi-stage swing detector (motion gate + optical flow + duration
filter). Pipes the frames through FFmpeg for H.264 + TCP delivery.

Invoked via the swingcam CLI: `swingcam stream --test <ff|dtl> [host]`
"""

import subprocess
import sys
import time

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 30

CYCLE_SECONDS = 10.0    # full cycle (still + burst + still)
BURST_START = 8.0       # when in the cycle the burst begins
BURST_DURATION = 0.7    # length of the motion burst (seconds)

# Object that sweeps across the frame during the burst.
# Sized so it fills ~6% of the frame, which lands in the optical-flow
# detector's "concentrated motion" sweet spot (5-40%).
OBJECT_W = 240
OBJECT_H = 460


def make_frame(t: float) -> np.ndarray:
    """Build a single frame at time t (seconds since start)."""
    # Dark gray background
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

    # Static "ground" stripe so the still phase is not perfectly black
    cv2.rectangle(frame, (440, 620), (840, 660), (70, 70, 70), -1)
    cv2.putText(
        frame, "swing-cam test", (530, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2,
    )

    cycle_pos = t % CYCLE_SECONDS
    in_burst = BURST_START <= cycle_pos < BURST_START + BURST_DURATION

    if in_burst:
        progress = (cycle_pos - BURST_START) / BURST_DURATION  # 0..1
        # Sweep horizontally; object enters from off-frame on the left
        # and exits off-frame on the right.
        total_x = WIDTH + OBJECT_W * 2
        cx = int(progress * total_x - OBJECT_W)
        cy = HEIGHT // 2 - 30

        # Bright object body — high contrast against the background
        cv2.rectangle(
            frame,
            (cx - OBJECT_W // 2, cy - OBJECT_H // 2),
            (cx + OBJECT_W // 2, cy + OBJECT_H // 2),
            (240, 240, 240), -1,
        )
        # A second offset object to add more total motion area
        cv2.rectangle(
            frame,
            (cx - OBJECT_W // 2 - 80, cy - OBJECT_H // 2 - 60),
            (cx + OBJECT_W // 2 - 80, cy - OBJECT_H // 2 + 20),
            (200, 200, 200), -1,
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
    last_logged_cycle = -1  # so we only print once per burst

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

            # Print exactly once at the start of each burst
            cycle_idx = int(t // CYCLE_SECONDS)
            cycle_pos = t - cycle_idx * CYCLE_SECONDS
            if cycle_pos >= BURST_START and cycle_idx != last_logged_cycle:
                last_logged_cycle = cycle_idx
                print(f"[t={t:.1f}s] BURST", flush=True)
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
