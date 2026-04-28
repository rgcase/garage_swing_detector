"""
swing-cam server: main entry point.

Wires up stream receivers, swing detectors, clip savers,
database, and the web UI.
"""

import logging
import logging.handlers
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
import yaml

from audio_receiver import AudioReceiver
from clip_saver import ClipSaver
from db import SwingDB
from gesture_detector import GestureDetector
from pose_analyzer import PoseAnalyzer
from stream_receiver import CircularFrameBuffer, StreamReceiver
from swing_detector import SwingDetector, SwingEvent
from web import create_app

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _expand_path(p: str) -> str:
    """Expand ~ and make path absolute."""
    return str(Path(p).expanduser().resolve())


def _setup_logging(log_cfg: dict | None = None):
    log_cfg = log_cfg or {}
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any existing handlers (notably the basicConfig bootstrap stream
    # handler installed below at import time). Otherwise stderr ends up with
    # one handler from basicConfig plus one from here, doubling every line in
    # the captured .err file.
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    # Always log to console
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file log
    log_dir = Path(_expand_path(log_cfg.get("log_dir", "~/.swingcam/logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "swingcam.log",
        maxBytes=log_cfg.get("max_bytes", 10 * 1024 * 1024),
        backupCount=log_cfg.get("backup_count", 5),
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


# Basic console logging until config is loaded
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
logger = logging.getLogger("swing-cam")


class SwingCamServer:
    """Orchestrates all components."""

    def __init__(self, config_path: str = "~/.swingcam/config.yaml"):
        self._config_path = _expand_path(config_path)
        with open(self._config_path) as f:
            self.config = yaml.safe_load(f)

        # Set up logging from config (replaces the basic bootstrap logger)
        _setup_logging(self.config.get("logging"))

        # Stream settings with defaults
        stream_cfg = self.config.get("stream", {})
        self._fps = stream_cfg.get("fps", 30)
        self._width = stream_cfg.get("width", 1280)
        self._height = stream_cfg.get("height", 720)

        # Expand ~ in data paths
        self._db_path = _expand_path(self.config["database"]["path"])
        self._clips_dir = _expand_path(self.config["clips"]["output_dir"])

        self.db = SwingDB(self._db_path, clips_dir=self._clips_dir)
        clips_cfg = self.config["clips"]
        self.clip_saver = ClipSaver(
            output_dir=self._clips_dir,
            pre_seconds=clips_cfg["pre_seconds"],
            post_seconds=clips_cfg["post_seconds"],
            fps=self._fps,
            width=self._width,
            height=self._height,
            max_storage_mb=clips_cfg.get("max_storage_mb"),
        )

        self.receivers: list[StreamReceiver] = []
        self.detectors: list[SwingDetector] = []
        self.buffers: dict[str, CircularFrameBuffer] = {}

        # Gesture detection (thumbs up/down after swing)
        self.gesture_detector = GestureDetector(
            watch_seconds=10.0,
            on_gesture=self._on_gesture_detected,
        )

        # Pose analysis (runs offline on saved clips)
        self.pose_analyzer = PoseAnalyzer(
            clips_dir=self._clips_dir,
        )
        self.pose_analyzer.on_analysis = self._on_analysis_complete

        # Audio receiver (optional; one mic feeds all cameras for impact corroboration)
        audio_cfg = self.config.get("audio", {}) or {}
        self.audio_receiver: AudioReceiver | None = None
        if audio_cfg.get("enabled", True):
            self.audio_receiver = AudioReceiver(
                host=audio_cfg.get("host", "0.0.0.0"),
                port=audio_cfg.get("port", 9558),
                sample_rate=audio_cfg.get("sample_rate", 16000),
                buffer_seconds=audio_cfg.get("buffer_seconds", 10.0),
            )
        self._audio_window = audio_cfg.get("correlation_window", 0.6)
        self._audio_threshold = audio_cfg.get("impact_threshold", 0.4)

        # For multi-camera swing correlation
        self._pending_events: list[SwingEvent] = []
        self._pending_lock = threading.Lock()
        self._app = None  # Set when web app is created, for SSE notifications

        self._setup_cameras()

    def _setup_cameras(self):
        """Create a receiver + detector + buffer for each configured camera."""
        det_cfg = self.config["detection"]
        buf_cfg = self.config["buffer"]

        for cam in self.config["cameras"]:
            name = cam["name"]

            # Circular buffer
            buf = CircularFrameBuffer(
                max_seconds=buf_cfg["max_seconds"], fps=self._fps
            )
            self.buffers[name] = buf

            # Stream receiver — supports both formats:
            #   source: "rtsp://192.168.4.50:554/stream1"  (RTSP/URL pull)
            #   host + port: "0.0.0.0" / 9556              (TCP listen for Pi push)
            if "source" in cam:
                source = cam["source"]
            else:
                source = f"tcp://{cam['host']}:{cam['port']}"

            receiver = StreamReceiver(
                name=name,
                source=source,
                buffer=buf,
                width=self._width,
                height=self._height,
                fps=self._fps,
            )

            # Swing detector
            detector = SwingDetector(
                camera_name=name,
                motion_threshold=det_cfg["motion_threshold"],
                motion_area_pct=det_cfg["motion_area_pct"],
                cooldown_seconds=det_cfg["cooldown_seconds"],
                roi=det_cfg.get("roi"),
                confidence_threshold=det_cfg.get("confidence_threshold", 0.5),
                spike_max_seconds=det_cfg.get("spike_max_seconds", 1.0),
                flow_score_min=det_cfg.get("flow_score_min", 0.4),
                on_swing=self._on_swing_detected,
            )

            # Wire: each decoded frame goes to the detector and gesture detector
            receiver.on_frame(detector.process_frame)
            receiver.on_frame(lambda tf: self.gesture_detector.process_frame(tf.frame))

            self.receivers.append(receiver)
            self.detectors.append(detector)

            logger.info(f"Configured camera: {name} ({cam['angle']}) on port {cam['port']}")

    def _record_impact(self, swing_id: str, trigger_time: float, spike_duration: float):
        """Look for an impact peak in the audio buffer near the swing trigger
        and persist it on the swing record. No-op if audio isn't running."""
        if self.audio_receiver is None:
            return
        # Center on the middle of the spike (impact usually near swing peak,
        # not the very first frame of motion). Window is ±0.3s by default.
        center = trigger_time + spike_duration / 2
        result = self.audio_receiver.find_impact(
            center_time=center,
            window=self._audio_window,
            threshold=self._audio_threshold,
        )
        if result is None:
            logger.info(f"[audio] No impact peak near swing {swing_id[:8]}")
            return
        offset_from_center, peak = result
        # Store offset relative to trigger_time, not center, for clarity.
        offset_from_trigger = (center - trigger_time) + offset_from_center
        self.db.set_impact(swing_id, offset_from_trigger, peak)
        logger.info(
            f"[audio] Impact for swing {swing_id[:8]}: "
            f"peak={peak:.2f} at trigger+{offset_from_trigger:+.3f}s"
        )

    def _on_swing_detected(self, event: SwingEvent):
        """
        Called when any camera detects a swing.
        For single-camera: immediately create a swing record and save a clip.
        For multi-camera: correlate events within a time window.
        """
        logger.info(
            f"Swing event from {event.camera_name}: "
            f"confidence={event.confidence:.2f} motion={event.motion_level:.1f}%"
        )
        num_cameras = len(self.config["cameras"])

        if num_cameras == 1:
            # Simple path: one camera, one swing
            swing_id = self.db.generate_swing_id()
            self.db.create_swing(swing_id, event.trigger_time)
            self._record_impact(swing_id, event.trigger_time, event.spike_duration)
            self.gesture_detector.start_watching(swing_id)
            cam_cfg = self.config["cameras"][0]
            angle = cam_cfg.get("angle", "unknown")
            self.clip_saver.save_clip_async(
                buffer=self.buffers[event.camera_name],
                trigger_time=event.trigger_time,
                camera_name=event.camera_name,
                swing_id=swing_id,
                spike_start=event.trigger_time,
                spike_duration=event.spike_duration,
                callback=lambda sid, cname, fp, ts, te, a=angle:
                    self._on_clip_saved(sid, cname, a, fp, trim_start=ts, trim_end=te),
            )
        else:
            self._handle_multi_camera_event(event)

    def _handle_multi_camera_event(self, event: SwingEvent):
        """
        Correlate swing detections across cameras.
        When events from different cameras fall within the correlation window,
        treat them as the same swing.
        """
        window = self.config["correlation"]["window_seconds"]

        with self._pending_lock:
            # Check if this event correlates with a pending event
            matched = None
            for pending in self._pending_events:
                if (
                    pending.camera_name != event.camera_name
                    and abs(pending.trigger_time - event.trigger_time) <= window
                ):
                    matched = pending
                    break

            if matched:
                # We have events from two cameras — create a correlated swing
                self._pending_events.remove(matched)
                swing_id = self.db.generate_swing_id()
                # Use the earliest trigger as the shared reference point
                # so both clips cover the same time window and play in sync
                shared_trigger = min(matched.trigger_time, event.trigger_time)
                self.db.create_swing(swing_id, shared_trigger)
                # Use the longer spike for impact correlation; either camera
                # may have caught the actual contact moment.
                impact_spike = max(matched.spike_duration, event.spike_duration)
                self._record_impact(swing_id, shared_trigger, impact_spike)
                self.gesture_detector.start_watching(swing_id)

                for evt in [matched, event]:
                    cam_cfg = next(
                        c for c in self.config["cameras"] if c["name"] == evt.camera_name
                    )
                    angle = cam_cfg.get("angle", "unknown")
                    # Each camera passes its own spike_start so trim values
                    # land on the actual swing frames in that camera's clip,
                    # even if its trigger lagged the shared anchor.
                    self.clip_saver.save_clip_async(
                        buffer=self.buffers[evt.camera_name],
                        trigger_time=shared_trigger,
                        camera_name=evt.camera_name,
                        swing_id=swing_id,
                        spike_start=evt.trigger_time,
                        spike_duration=evt.spike_duration,
                        callback=lambda sid, cname, fp, ts, te, a=angle:
                            self._on_clip_saved(sid, cname, a, fp, trim_start=ts, trim_end=te),
                    )
                logger.info(f"Correlated swing {swing_id} from {matched.camera_name} + {event.camera_name}")
            else:
                # No match yet — add to pending and set a timer
                self._pending_events.append(event)
                threading.Timer(
                    window + 0.5,
                    self._flush_pending_event,
                    args=[event],
                ).start()

    # Single-camera events need higher confidence since there's
    # no corroborating camera to back them up.
    SINGLE_CAM_CONFIDENCE_MIN = 0.6

    def _flush_pending_event(self, event: SwingEvent):
        """
        If a pending event wasn't correlated within the time window,
        save it as a single-camera swing — but require higher confidence.
        """
        with self._pending_lock:
            if event in self._pending_events:
                self._pending_events.remove(event)

                if event.confidence < self.SINGLE_CAM_CONFIDENCE_MIN:
                    logger.info(
                        f"Single-camera event from {event.camera_name} rejected: "
                        f"confidence {event.confidence:.2f} < {self.SINGLE_CAM_CONFIDENCE_MIN} "
                        f"(no corroboration from second camera)"
                    )
                    return
                swing_id = self.db.generate_swing_id()
                self.db.create_swing(swing_id, event.trigger_time)
                self._record_impact(swing_id, event.trigger_time, event.spike_duration)
                self.gesture_detector.start_watching(swing_id)
                cam_cfg = next(
                    c for c in self.config["cameras"] if c["name"] == event.camera_name
                )
                angle = cam_cfg.get("angle", "unknown")
                self.clip_saver.save_clip_async(
                    buffer=self.buffers[event.camera_name],
                    trigger_time=event.trigger_time,
                    camera_name=event.camera_name,
                    swing_id=swing_id,
                    spike_start=event.trigger_time,
                    spike_duration=event.spike_duration,
                    callback=lambda sid, cname, fp, ts, te, a=angle:
                        self._on_clip_saved(sid, cname, a, fp, trim_start=ts, trim_end=te),
                )
                logger.info(f"Single-camera swing {swing_id} from {event.camera_name} (no correlation)")

    def _on_gesture_detected(self, swing_id: str, gesture: str):
        """Called when a thumbs up/down gesture is confirmed."""
        tag = gesture  # 'good' or 'bad'
        self.db.tag_swing(swing_id, tag)
        logger.info(f"Swing {swing_id[:8]} tagged as '{tag}' via gesture")
        if self._app and hasattr(self._app, "notify_new_swing"):
            self._app.notify_new_swing(swing_id)

    def _on_analysis_complete(self, result: dict):
        """Called when pose analysis finishes for a clip."""
        import json
        self.db.save_analysis(
            swing_id=result["swing_id"],
            camera_name=result["camera_name"],
            tempo_ratio=result.get("tempo_ratio"),
            backswing_frames=result.get("backswing_frames"),
            downswing_frames=result.get("downswing_frames"),
            head_stability=result.get("head_stability"),
            hip_rotation=result.get("hip_rotation"),
            shoulder_rotation=result.get("shoulder_rotation"),
            phases_json=json.dumps(result.get("phases", [])),
        )

    def _on_clip_saved(
        self, swing_id: str, camera_name: str, angle: str, filepath: str | None,
        trim_start: float | None = None, trim_end: float | None = None,
    ):
        if filepath:
            self.db.add_clip(swing_id, camera_name, angle, filepath,
                             trim_start=trim_start, trim_end=trim_end)
            logger.info(f"Clip saved for swing {swing_id}: {filepath}")
            # Queue for pose analysis
            self.pose_analyzer.queue_analysis(swing_id, camera_name, filepath)
            # Notify SSE subscribers of the new/updated swing
            if self._app and hasattr(self._app, "notify_new_swing"):
                self._app.notify_new_swing(swing_id)
        else:
            logger.warning(f"Failed to save clip for swing {swing_id} ({camera_name})")

    @staticmethod
    def _lan_ip() -> str:
        """Best-effort LAN IP, for printing actionable stream URLs."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "localhost"
        finally:
            s.close()

    def _log_stream_targets(self):
        lan_ip = self._lan_ip()
        for cam in self.config["cameras"]:
            host = cam["host"]
            if host in ("0.0.0.0", "::"):
                host = lan_ip
            logger.info(
                f"Stream target [{cam['name']}/{cam['angle']}]: "
                f"tcp://{host}:{cam['port']}"
            )
        if self.audio_receiver is not None:
            host = self.audio_receiver.host
            if host in ("0.0.0.0", "::"):
                host = lan_ip
            logger.info(
                f"Stream target [audio]: tcp://{host}:{self.audio_receiver.port}"
            )

    def run(self):
        """Start everything."""
        logger.info("=" * 50)
        logger.info("  swing-cam server starting")
        logger.info("=" * 50)

        # Start stream receivers
        for receiver in self.receivers:
            receiver.start()
        if self.audio_receiver is not None:
            self.audio_receiver.start()

        # Start web server in a thread
        web_cfg = self.config["server"]
        app = create_app(
            self.db,
            self._clips_dir,
            receivers=self.receivers,
            buffers=self.buffers,
            detectors=self.detectors,
            config=self.config,
            config_path=self._config_path,
        )
        self._app = app

        server_thread = threading.Thread(
            target=lambda: uvicorn.run(
                app,
                host=web_cfg["host"],
                port=web_cfg["web_port"],
                log_level="warning",
            ),
            daemon=True,
            name="web",
        )
        server_thread.start()
        logger.info(f"Web UI available at http://localhost:{web_cfg['web_port']}")
        self._log_stream_targets()

        # Block until Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            for receiver in self.receivers:
                receiver.stop()


def main():
    # Support: python main.py --config /path/to/config.yaml
    # or:     python main.py /path/to/config.yaml (legacy)
    config_path = "~/.swingcam/config.yaml"
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            config_path = args[idx + 1]
    elif args and not args[0].startswith("-"):
        config_path = args[0]

    config_path = _expand_path(config_path)
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        logger.error("Run 'swingcam start' to create the default config at ~/.swingcam/config.yaml")
        sys.exit(1)

    server = SwingCamServer(config_path)
    server.run()


if __name__ == "__main__":
    main()
