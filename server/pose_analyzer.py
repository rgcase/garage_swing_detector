"""
Pose analyzer: runs MediaPipe Pose on saved swing clips to detect
swing phases and extract metrics.

Runs offline (on saved clips, not live frames) to avoid CPU contention
with the real-time detection pipeline.
"""

import json
import logging
import math
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning(
        "mediapipe not installed — pose analysis disabled. "
        "Install with: pip install mediapipe"
    )

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def _angle_between(a, b):
    """Angle in degrees between two 2D vectors."""
    dot = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0]**2 + a[1]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_angle))


def _midpoint(lm_a, lm_b):
    return ((lm_a.x + lm_b.x) / 2, (lm_a.y + lm_b.y) / 2)


class SwingPhase:
    ADDRESS = "address"
    BACKSWING = "backswing"
    TOP = "top"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"


class PoseAnalyzer:
    """Analyzes swing clips using MediaPipe Pose."""

    def __init__(self, clips_dir: str):
        self.clips_dir = Path(clips_dir)
        self._queue: deque[tuple[str, str, str]] = deque()  # (swing_id, camera_name, filepath)
        self._lock = threading.Lock()
        self._running = False

        if HAS_MEDIAPIPE:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._pose = None

    def queue_analysis(self, swing_id: str, camera_name: str, filepath: str):
        """Queue a clip for background analysis."""
        if not HAS_MEDIAPIPE:
            return
        with self._lock:
            self._queue.append((swing_id, camera_name, filepath))
            if not self._running:
                self._running = True
                threading.Thread(
                    target=self._process_queue,
                    daemon=True,
                    name="pose-analysis",
                ).start()

    def _process_queue(self):
        """Process queued clips one at a time."""
        while True:
            with self._lock:
                if not self._queue:
                    self._running = False
                    return
                swing_id, camera_name, filepath = self._queue.popleft()

            try:
                result = self.analyze_clip(filepath)
                if result:
                    result["swing_id"] = swing_id
                    result["camera_name"] = camera_name
                    logger.info(
                        f"Analysis complete for {swing_id[:8]}/{camera_name}: "
                        f"tempo={result.get('tempo_ratio', 'N/A')}"
                    )
                    # Store via callback (set by main.py)
                    if self.on_analysis:
                        self.on_analysis(result)
            except Exception as e:
                logger.error(f"Pose analysis failed for {filepath}: {e}")

    on_analysis = None  # Callback set by main.py

    def analyze_clip(self, filepath: str) -> dict | None:
        """Run pose estimation on a clip and extract swing metrics.

        Returns dict with: tempo_ratio, backswing_frames, downswing_frames,
        head_stability, hip_rotation, shoulder_rotation, phases.
        """
        if not self._pose:
            return None

        clip_path = self.clips_dir / filepath
        if not clip_path.exists():
            logger.warning(f"Clip not found: {clip_path}")
            return None

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        landmarks_per_frame = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            if results.pose_landmarks:
                landmarks_per_frame.append((frame_idx, results.pose_landmarks.landmark))
            else:
                landmarks_per_frame.append((frame_idx, None))
            frame_idx += 1

        cap.release()

        if len(landmarks_per_frame) < 30:
            return None

        return self._extract_metrics(landmarks_per_frame, fps)

    def _extract_metrics(self, frames: list, fps: float) -> dict:
        """Extract swing phases and metrics from pose landmarks."""

        # Track key positions over time
        wrist_y = []       # Lead wrist y-position (left wrist for right-handed)
        nose_positions = []
        hip_midpoints = []
        shoulder_angles = []
        hip_angles = []

        for idx, lm in frames:
            if lm is None:
                wrist_y.append(None)
                nose_positions.append(None)
                hip_midpoints.append(None)
                shoulder_angles.append(None)
                hip_angles.append(None)
                continue

            wrist_y.append(lm[LEFT_WRIST].y)
            nose_positions.append((lm[NOSE].x, lm[NOSE].y))
            hip_midpoints.append(_midpoint(lm[LEFT_HIP], lm[RIGHT_HIP]))

            # Shoulder line angle relative to horizontal
            shoulder_vec = (
                lm[RIGHT_SHOULDER].x - lm[LEFT_SHOULDER].x,
                lm[RIGHT_SHOULDER].y - lm[LEFT_SHOULDER].y,
            )
            shoulder_angles.append(math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0])))

            # Hip line angle
            hip_vec = (
                lm[RIGHT_HIP].x - lm[LEFT_HIP].x,
                lm[RIGHT_HIP].y - lm[LEFT_HIP].y,
            )
            hip_angles.append(math.degrees(math.atan2(hip_vec[1], hip_vec[0])))

        # Find swing phases based on wrist trajectory
        # Filter None values for analysis
        valid_wrist = [(i, y) for i, y in enumerate(wrist_y) if y is not None]
        if len(valid_wrist) < 30:
            return {"phases": [], "tempo_ratio": None}

        # Find top of backswing: minimum wrist y (highest point, y increases downward)
        min_wrist_idx = min(valid_wrist, key=lambda x: x[1])[0]

        # Find approximate impact: maximum wrist velocity after top
        wrist_velocity = []
        for i in range(1, len(wrist_y)):
            if wrist_y[i] is not None and wrist_y[i-1] is not None:
                wrist_velocity.append((i, wrist_y[i] - wrist_y[i-1]))
            else:
                wrist_velocity.append((i, 0))

        # Impact is max downward velocity after top of backswing
        post_top_vel = [(i, v) for i, v in wrist_velocity if i > min_wrist_idx]
        impact_idx = max(post_top_vel, key=lambda x: x[1])[0] if post_top_vel else min_wrist_idx + 5

        # Phases
        phases = [
            {"phase": SwingPhase.ADDRESS, "start_frame": 0, "end_frame": max(0, min_wrist_idx - 10)},
            {"phase": SwingPhase.BACKSWING, "start_frame": max(0, min_wrist_idx - 10), "end_frame": min_wrist_idx},
            {"phase": SwingPhase.TOP, "start_frame": min_wrist_idx, "end_frame": min_wrist_idx + 1},
            {"phase": SwingPhase.DOWNSWING, "start_frame": min_wrist_idx, "end_frame": impact_idx},
            {"phase": SwingPhase.IMPACT, "start_frame": impact_idx, "end_frame": impact_idx + 1},
            {"phase": SwingPhase.FOLLOW_THROUGH, "start_frame": impact_idx, "end_frame": len(frames) - 1},
        ]

        # Tempo: backswing frames / downswing frames
        backswing_frames = min_wrist_idx - phases[1]["start_frame"]
        downswing_frames = impact_idx - min_wrist_idx
        tempo_ratio = round(backswing_frames / max(1, downswing_frames), 2)

        # Head stability: std dev of nose position
        valid_nose = [p for p in nose_positions if p is not None]
        head_stability = None
        if valid_nose:
            nose_x = [p[0] for p in valid_nose]
            nose_y = [p[1] for p in valid_nose]
            head_stability = round(
                float(np.std(nose_x) + np.std(nose_y)) * 100, 2
            )

        # Rotation: difference between address and top-of-backswing angles
        hip_rotation = None
        shoulder_rotation = None
        if hip_angles[0] is not None and hip_angles[min_wrist_idx] is not None:
            hip_rotation = round(abs(hip_angles[min_wrist_idx] - hip_angles[0]), 1)
        if shoulder_angles[0] is not None and shoulder_angles[min_wrist_idx] is not None:
            shoulder_rotation = round(abs(shoulder_angles[min_wrist_idx] - shoulder_angles[0]), 1)

        return {
            "phases": phases,
            "tempo_ratio": tempo_ratio,
            "backswing_frames": backswing_frames,
            "downswing_frames": downswing_frames,
            "head_stability": head_stability,
            "hip_rotation": hip_rotation,
            "shoulder_rotation": shoulder_rotation,
        }
