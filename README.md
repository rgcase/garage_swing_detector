# swing-cam

Automated golf swing recording system using Raspberry Pi Zero cameras and a Linux server for motion detection, clip saving, and multi-camera coordination.

## Architecture

```
┌──────────────┐        TCP/H.264        ┌──────────────────────┐
│  Pi Zero (1) │ ───────────────────────► │                      │
│  face-on cam │                          │   Ubuntu Server      │
└──────────────┘                          │                      │
                                          │  - Stream receiver   │
┌──────────────┐        TCP/H.264        │  - Swing detection   │
│  Pi Zero (2) │ ───────────────────────► │  - Clip extraction   │
│  down-the-   │                          │  - Web UI for review │
│  line cam    │                          │                      │
└──────────────┘                          └──────────────────────┘
```

**Pi Zero** — Streams H.264 video over TCP. Runs as a systemd service on boot.
No processing happens on the Pi.

**Server** — Receives streams, maintains a circular buffer per camera,
runs frame-differencing-based swing detection, saves clips around detected
swings, and serves a web UI for reviewing and tagging clips.

## Quick Start

### Pi Zero Setup

1. Flash Raspberry Pi OS Lite (Bookworm) and enable the camera interface.

2. Copy the `pi/` directory to the Pi:
   ```bash
   scp -r pi/ pi@swingcam-1.local:~/swing-cam/
   ```

3. Run the setup script:
   ```bash
   ssh pi@swingcam-1.local
   cd ~/swing-cam
   chmod +x setup.sh
   ./setup.sh
   ```

4. The stream starts automatically on boot. To test manually:
   ```bash
   ./swingcam stream ff <server_ip>    # or: dtl
   ```

### Server Setup

1. Install dependencies:
   ```bash
   cd server/
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Edit `config.yaml` with your camera IPs and preferences.

3. Run:
   ```bash
   python main.py
   ```

4. Open the web UI at `http://localhost:8080`

## Configuration

See `server/config.yaml` for all options. Key settings:

- `cameras` — List of camera streams (name, host, port, angle)
- `detection.motion_threshold` — Sensitivity for swing detection (lower = more sensitive)
- `detection.cooldown_seconds` — Minimum time between detected swings
- `clips.pre_seconds` — Seconds of video to keep before the swing trigger
- `clips.post_seconds` — Seconds of video to keep after the swing trigger
- `clips.output_dir` — Where to save clip files

## Project Structure

```
swing-cam/
├── pi/
│   ├── setup.sh                  # Pi setup/install script
│   └── swing-cam-stream.service  # systemd unit file (calls `swingcam stream`)
├── server/
│   ├── main.py                   # Entry point
│   ├── config.yaml               # Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── stream_receiver.py        # TCP stream ingestion + circular buffer
│   ├── swing_detector.py         # Motion-based swing detection
│   ├── clip_saver.py             # FFmpeg clip extraction
│   ├── db.py                     # SQLite swing database
│   ├── web.py                    # FastAPI web UI + API
│   └── templates/
│       └── index.html            # Review/tagging UI
└── README.md
```
