# CLAUDE.md

## Project Overview

**garage_swing_detector** is an automated golf swing recording system. Raspberry Pi Zero cameras stream H.264 video to a Mac Mini server, which runs motion-based swing detection, saves clips around detected swings, and serves a web UI for reviewing and tagging recordings.

The goal is a "set and forget" system: cameras run continuously, swings are detected and clipped automatically, and the user reviews/tags them later via web UI or phone.

## Architecture

```
Pi Zero (face-on cam) ──TCP/H.264──►  Mac Mini (macOS)
Pi Zero (dtl cam)     ──TCP/H.264──►  - FFmpeg decodes streams
                                      - Circular frame buffer per camera
                                      - Frame-differencing swing detection
                                      - Clip extraction around triggers
                                      - SQLite database for swing records
                                      - FastAPI web UI for review/tagging
```

**Key design decisions:**
- Pi Zero is too weak for any CV/detection — it only streams raw H.264 via the hardware encoder
- Server uses FFmpeg subprocess to receive TCP streams and decode to raw frames
- Circular buffer holds ~20s of decoded frames per camera; on swing detection, a time window (pre + post trigger) is extracted and written to MP4
- Multi-camera correlation: swing events from different cameras within a configurable time window are grouped into one swing record

## Directory Structure

- `pi/` — Scripts and systemd service for the Pi Zero cameras. Bash only, no Python dependencies.
- `server/` — Python server application. Entry point is `main.py`.
  - `stream_receiver.py` — TCP listener, FFmpeg subprocess, circular frame buffer
  - `swing_detector.py` — Frame differencing motion detection
  - `clip_saver.py` — Extracts frames from buffer, writes H.264 MP4 via FFmpeg
  - `db.py` — SQLite schema and queries (swings table + clips table)
  - `web.py` — FastAPI app with Jinja2 templates
  - `config.yaml` — All tunable parameters
  - `templates/index.html` — Dark-themed review/tagging UI

## Development

### Server

```bash
cd server/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Requires FFmpeg installed on the host (`brew install ffmpeg`).

Note: macOS firewall must allow FFmpeg TCP connections (System Settings > Privacy & Security > Firewall).

Web UI runs at `http://localhost:8080` by default.

### Pi

The Pi runs Raspberry Pi OS Lite (Bookworm). `rpicam-vid` is used for streaming (part of the default install). No Python needed on the Pi.

```bash
# Stream manually (angle → port: ff=9556, dtl=9557)
./swingcam stream ff <server_ip>
./swingcam stream dtl <server_ip>
```

`pi/setup.sh` installs the stream as a systemd service.

### Testing without a Pi

You can simulate a camera stream with FFmpeg from any machine:

```bash
# Stream a test video file to the server (ff = 9556, dtl = 9557)
ffmpeg -re -i test_swing.mp4 -c:v libx264 -f h264 tcp://<server_ip>:9556
```

Or generate a synthetic stream:

```bash
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 -c:v libx264 -f h264 tcp://<server_ip>:9556
```

## Configuration

All tuning is in `server/config.yaml`. Key parameters:

- `detection.motion_threshold` (default 25) — pixel diff threshold, lower = more sensitive
- `detection.motion_area_pct` (default 1.5) — % of frame pixels that must exceed threshold to trigger
- `detection.cooldown_seconds` (default 5) — minimum gap between swing detections
- `detection.roi` — optional region of interest as [x, y, w, h] normalized 0-1, set this to the hitting area to reduce false positives
- `clips.pre_seconds` / `clips.post_seconds` — how much video to keep around the trigger point

## Planned Features

- [ ] Auto-play latest swing in web UI (SSE live updates)
- [ ] Swing comparison view (side-by-side two swings)
- [ ] Gesture detection (thumbs up/down via MediaPipe) for quick swing rating
- [ ] MediaPipe Pose estimation for swing phase detection and metrics
- [ ] Export tagged swings as compilation video
- [ ] Mobile-friendly PWA UI improvements
