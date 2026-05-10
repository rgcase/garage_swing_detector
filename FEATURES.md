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
- [x] Slow-motion playback (0.1x to 2x speed controls)
- [x] Practice sessions (group swings by time, session headers with stats)
- [x] Auto-trim ("Swing Only" button seeks to detected swing bounds)
- [x] Side-by-side video export
- [x] RTSP and USB camera support
- [x] Trend charts (tempo, head stability, rotation, daily volume over time)
- [x] Swing sequence export (4-up address/top/impact/finish strip image)
- [x] Push notifications (ntfy.sh integration; fires on swing detection)
- [x] Swing overlay (ghost-overlay two swings on /compare with adjustable opacity)
- [x] Improved pose phase detection (proper address/finish, audio-anchored impact, quality gate)
- [x] Auto-highlight reel (concat all "good" swings in last N days into a single MP4)

## Up Next

## Planned

### Club tagging
Quick-select which club you're hitting (D, 3W, 7i, PW, etc.) per swing or per session. Filter and compare metrics by club. A driver swing has very different metrics than a wedge.

### Impact sound detection
Add USB microphone support as a secondary detection signal. The crack of ball impact is very distinctive and would reduce false positives. Could also distinguish real swings from practice swings (no ball).

## Ideas

### Launch monitor integration
If the user has a launch monitor (Garmin R10, Rapsodo, etc.), correlate swing clips with launch data (ball speed, carry distance, spin). Would require API integration per device.

### Voice commands
"Hey swingcam, that was a seven iron" — tag the last swing with club info via voice. Uses the same mic as impact sound detection.
