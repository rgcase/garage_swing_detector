"""
swing-cam server: main entry point.

Wires up stream receivers, swing detectors, clip savers,
database, and the web UI.
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path

import uvicorn
import yaml

from clip_saver import ClipSaver
from db import SwingDB
from gesture_detector import GestureDetector
from pose_analyzer import PoseAnalyzer
from stream_receiver import CircularFrameBuffer, StreamReceiver
from swing_detector import SwingDetector, SwingEvent
from web import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("swing-cam")


class SwingCamServer:
    """Orchestrates all components."""

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = SwingDB(
            self.config["database"]["path"],
            clips_dir=self.config["clips"]["output_dir"],
        )
        self.clip_saver = ClipSaver(
            output_dir=self.config["clips"]["output_dir"],
            pre_seconds=self.config["clips"]["pre_seconds"],
            post_seconds=self.config["clips"]["post_seconds"],
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
            clips_dir=self.config["clips"]["output_dir"],
        )
        self.pose_analyzer.on_analysis = self._on_analysis_complete

        # For multi-camera swing correlation
        self._pending_events: list[SwingEvent] = []
        self._pending_lock = threading.Lock()
        self._app = None  # Set when web app is created, for SSE notifications

        self._setup_cameras()

    def _setup_cameras(self):
        """Create a receiver + detector + buffer for each configured camera."""
        det_cfg = self.config["detection"]
        buf_cfg = self.config["buffer"]
        fps = 30.0  # Match the Pi stream FPS

        for cam in self.config["cameras"]:
            name = cam["name"]

            # Circular buffer
            buf = CircularFrameBuffer(
                max_seconds=buf_cfg["max_seconds"], fps=fps
            )
            self.buffers[name] = buf

            # Stream receiver
            receiver = StreamReceiver(
                name=name,
                host=cam["host"],
                port=cam["port"],
                buffer=buf,
                fps=fps,
            )

            # Swing detector
            detector = SwingDetector(
                camera_name=name,
                motion_threshold=det_cfg["motion_threshold"],
                motion_area_pct=det_cfg["motion_area_pct"],
                cooldown_seconds=det_cfg["cooldown_seconds"],
                roi=det_cfg.get("roi"),
                on_swing=self._on_swing_detected,
            )

            # Wire: each decoded frame goes to the detector and gesture detector
            receiver.on_frame(detector.process_frame)
            receiver.on_frame(lambda tf: self.gesture_detector.process_frame(tf.frame))

            self.receivers.append(receiver)
            self.detectors.append(detector)

            logger.info(f"Configured camera: {name} ({cam['angle']}) on port {cam['port']}")

    def _on_swing_detected(self, event: SwingEvent):
        """
        Called when any camera detects a swing.
        For single-camera: immediately create a swing record and save a clip.
        For multi-camera: correlate events within a time window.
        """
        num_cameras = len(self.config["cameras"])

        if num_cameras == 1:
            # Simple path: one camera, one swing
            swing_id = self.db.generate_swing_id()
            self.db.create_swing(swing_id, event.trigger_time)
            self.gesture_detector.start_watching(swing_id)
            cam_cfg = self.config["cameras"][0]
            self.clip_saver.save_clip_async(
                buffer=self.buffers[event.camera_name],
                trigger_time=event.trigger_time,
                camera_name=event.camera_name,
                swing_id=swing_id,
                callback=lambda sid, cname, fp: self._on_clip_saved(
                    sid, cname, cam_cfg.get("angle", "unknown"), fp
                ),
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
                self.db.create_swing(swing_id, min(matched.trigger_time, event.trigger_time))
                self.gesture_detector.start_watching(swing_id)

                for evt in [matched, event]:
                    cam_cfg = next(
                        c for c in self.config["cameras"] if c["name"] == evt.camera_name
                    )
                    self.clip_saver.save_clip_async(
                        buffer=self.buffers[evt.camera_name],
                        trigger_time=evt.trigger_time,
                        camera_name=evt.camera_name,
                        swing_id=swing_id,
                        callback=lambda sid, cname, fp, angle=cam_cfg.get("angle", "unknown"):
                            self._on_clip_saved(sid, cname, angle, fp),
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

    def _flush_pending_event(self, event: SwingEvent):
        """
        If a pending event wasn't correlated within the time window,
        save it as a single-camera swing.
        """
        with self._pending_lock:
            if event in self._pending_events:
                self._pending_events.remove(event)
                swing_id = self.db.generate_swing_id()
                self.db.create_swing(swing_id, event.trigger_time)
                self.gesture_detector.start_watching(swing_id)
                cam_cfg = next(
                    c for c in self.config["cameras"] if c["name"] == event.camera_name
                )
                self.clip_saver.save_clip_async(
                    buffer=self.buffers[event.camera_name],
                    trigger_time=event.trigger_time,
                    camera_name=event.camera_name,
                    swing_id=swing_id,
                    callback=lambda sid, cname, fp, angle=cam_cfg.get("angle", "unknown"):
                        self._on_clip_saved(sid, cname, angle, fp),
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

    def _on_clip_saved(self, swing_id: str, camera_name: str, angle: str, filepath: str | None):
        if filepath:
            self.db.add_clip(swing_id, camera_name, angle, filepath)
            logger.info(f"Clip saved for swing {swing_id}: {filepath}")
            # Queue for pose analysis
            self.pose_analyzer.queue_analysis(swing_id, camera_name, filepath)
            # Notify SSE subscribers of the new/updated swing
            if self._app and hasattr(self._app, "notify_new_swing"):
                self._app.notify_new_swing(swing_id)
        else:
            logger.warning(f"Failed to save clip for swing {swing_id} ({camera_name})")

    def run(self):
        """Start everything."""
        logger.info("=" * 50)
        logger.info("  swing-cam server starting")
        logger.info("=" * 50)

        # Start stream receivers
        for receiver in self.receivers:
            receiver.start()

        # Start web server in a thread
        web_cfg = self.config["server"]
        app = create_app(
            self.db,
            self.config["clips"]["output_dir"],
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

        # Block until Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            for receiver in self.receivers:
                receiver.stop()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    server = SwingCamServer(config_path)
    server.run()


if __name__ == "__main__":
    main()
