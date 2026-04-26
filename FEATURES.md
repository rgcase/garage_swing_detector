# Feature Roadmap

## Completed

- [x] Multi-camera streaming (Pi TCP, RTSP, USB webcam)
- [x] Multi-stage swing detection (motion gate + optical flow + duration filter)
- [x] Multi-camera correlation (group same swing from different angles)
- [x] Synchronized side-by-side clip playback
- [x] Side-by-side video export
- [x] Web UI with tagging (good/bad), notes, delete
- [x] Swing comparison view (pick two swings to compare)
- [x] Live MJPEG stream view
- [x] SSE live updates (auto-reload on new swing)
- [x] Keyboard shortcuts (j/k navigate, g/b tag, space play)
- [x] ROI calibration (web-based, draw on snapshot)
- [x] Motion debug dashboard (real-time Chart.js graph)
- [x] Gesture detection (thumbs up/down via MediaPipe Hands)
- [x] Pose analysis (swing phases, tempo, head stability, rotation)
- [x] Mobile-friendly responsive UI with PWA meta tags
- [x] Relative timestamps ("3m ago")
- [x] Log rotation (10MB, 5 backups)
- [x] Clip storage limit (auto-delete oldest)
- [x] Daemon management CLI (swingcam start/stop/restart/update/config)
- [x] Configurable stream settings (resolution, fps)

## Up Next

### Slow-motion playback
Playback rate control (0.25x, 0.5x, 1x, 2x) on the video player. Golf swings happen in ~0.3s so slow-mo is essential for actually reviewing mechanics.

### Practice sessions
Group swings within 30 minutes of each other into sessions. Show session summary: date, duration, swing count, good/bad breakdown. Makes it easy to review a whole practice.

### Auto-trim clips
Use detection timestamps to trim clips to just the swing itself (~2-3s) instead of the full 9s window. Saves storage and speeds up review. Keep the full clip available as an option.

## Planned

### Trend charts
Plot tempo, head stability, hip/shoulder rotation over weeks and months. Track consistency and improvement over time. Filter by club if tagged.

### Club tagging
Quick-select which club you're hitting (D, 3W, 7i, PW, etc.) per swing or per session. Filter and compare metrics by club. A driver swing has very different metrics than a wedge.

### Push notifications
Send a notification to your phone when a swing is recorded. Tap the notification to jump directly to the swing in the web UI.

### Impact sound detection
Add USB microphone support as a secondary detection signal. The crack of ball impact is very distinctive and would reduce false positives. Could also distinguish real swings from practice swings (no ball).

## Ideas

### Swing overlay
Ghost-overlay a reference "good" swing on top of the current one with adjustable transparency. More useful than side-by-side for spotting differences in positions.

### Auto-highlight reel
Weekly auto-generated compilation of swings tagged "good", exported as a single video with transitions.

### Swing sequence export
Export a swing as a strip of key frames (address, top, impact, finish) as a single image. Classic golf instruction format.

### Launch monitor integration
If the user has a launch monitor (Garmin R10, Rapsodo, etc.), correlate swing clips with launch data (ball speed, carry distance, spin). Would require API integration per device.

### Voice commands
"Hey swingcam, that was a seven iron" — tag the last swing with club info via voice. Uses the same mic as impact sound detection.
