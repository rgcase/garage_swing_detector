"""
Pose analyzer: runs MediaPipe Pose on saved swing clips to detect
swing phases and extract metrics.

Uses the MediaPipe Tasks API (PoseLandmarker). Runs offline (on saved
clips, not live frames) to avoid CPU contention with the real-time
detection pipeline.
"""

import json
import logging
import math
import threading
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".swingcam" / "models"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning(
        "mediapipe not installed — pose analysis disabled. "
        "Install with: pip install mediapipe"
    )

# MediaPipe Pose landmark indices (same in Tasks API)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def _download_model(url: str, dest: Path):
    """Download a model file if it doesn't exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model to {dest}...")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Model downloaded: {dest}")


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
        self._detector = None

        if HAS_MEDIAPIPE:
            try:
                _download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(POSE_MODEL_PATH)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            except Exception as e:
                logger.warning(f"Failed to initialize pose landmarker: {e}")

    def queue_analysis(
        self,
        swing_id: str,
        camera_name: str,
        filepath: str,
        pre_seconds: float = 3.0,
        impact_offset: float | None = None,
    ):
        """Queue a clip for background analysis.

        pre_seconds and impact_offset come from the server config and the
        recorded swing record respectively; they let us anchor the impact
        frame to the audio peak when one was heard.
        """
        if not self._detector:
            return
        with self._lock:
            self._queue.append(
                (swing_id, camera_name, filepath, pre_seconds, impact_offset)
            )
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
                swing_id, camera_name, filepath, pre_seconds, impact_offset = (
                    self._queue.popleft()
                )

            try:
                result = self.analyze_clip(
                    filepath, pre_seconds=pre_seconds, impact_offset=impact_offset
                )
                if result:
                    result["swing_id"] = swing_id
                    result["camera_name"] = camera_name
                    logger.info(
                        f"Analysis complete for {swing_id[:8]}/{camera_name}: "
                        f"tempo={result.get('tempo_ratio', 'N/A')} "
                        f"quality={result.get('quality', 'N/A')}"
                    )
                    # Store via callback (set by main.py)
                    if self.on_analysis:
                        self.on_analysis(result)
            except Exception as e:
                logger.error(f"Pose analysis failed for {filepath}: {e}")

    on_analysis = None  # Callback set by main.py

    def analyze_clip(
        self,
        filepath: str,
        pre_seconds: float = 3.0,
        impact_offset: float | None = None,
    ) -> dict | None:
        """Run pose estimation on a clip and extract swing metrics.

        Returns dict with: tempo_ratio, backswing_frames, downswing_frames,
        head_stability, hip_rotation, shoulder_rotation, phases, quality.
        """
        if not self._detector:
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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires a monotonically increasing timestamp in ms
            timestamp_ms = int(frame_idx * (1000 / fps))
            result = self._detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks_per_frame.append((frame_idx, result.pose_landmarks[0]))
            else:
                landmarks_per_frame.append((frame_idx, None))
            frame_idx += 1

        cap.release()

        if len(landmarks_per_frame) < 30:
            return None

        return self._extract_metrics(
            landmarks_per_frame, fps,
            pre_seconds=pre_seconds, impact_offset=impact_offset,
        )

    # ── Phase detection thresholds ──
    # Wrist must stay below this per-frame displacement (normalized coords)
    # for STILLNESS_SECONDS seconds to count as quiescent (address or finish).
    QUIESCENCE_THRESHOLD = 0.012
    QUIESCENCE_SECONDS = 0.4
    # If less than this fraction of frames in the swing window have valid
    # landmarks, the analysis is flagged "low" quality.
    QUALITY_VALID_RATIO = 0.75

    def _extract_metrics(
        self,
        frames: list,
        fps: float,
        pre_seconds: float = 3.0,
        impact_offset: float | None = None,
    ) -> dict:
        """Extract swing phases and metrics from pose landmarks.

        Detects address (wrist quiescence before top), top of backswing
        (combined-hands y-extremum), impact (audio-anchored when available,
        else wrist-velocity peak), and finish (wrist quiescence after impact).
        Metrics are computed within the actual swing window — pre-swing
        standing-around no longer contaminates head stability or rotation.
        """

        # Use the midpoint of both wrists as "the hands" — more robust than
        # one wrist when MediaPipe flickers, and tracks the club end better.
        hands_x: list[float | None] = []
        hands_y: list[float | None] = []
        nose_positions: list[tuple[float, float] | None] = []
        shoulder_angles: list[float | None] = []
        hip_angles: list[float | None] = []

        for idx, lm in frames:
            if lm is None:
                hands_x.append(None); hands_y.append(None)
                nose_positions.append(None)
                shoulder_angles.append(None); hip_angles.append(None)
                continue

            hands_x.append((lm[LEFT_WRIST].x + lm[RIGHT_WRIST].x) / 2)
            hands_y.append((lm[LEFT_WRIST].y + lm[RIGHT_WRIST].y) / 2)
            nose_positions.append((lm[NOSE].x, lm[NOSE].y))
            shoulder_angles.append(math.degrees(math.atan2(
                lm[RIGHT_SHOULDER].y - lm[LEFT_SHOULDER].y,
                lm[RIGHT_SHOULDER].x - lm[LEFT_SHOULDER].x,
            )))
            hip_angles.append(math.degrees(math.atan2(
                lm[RIGHT_HIP].y - lm[LEFT_HIP].y,
                lm[RIGHT_HIP].x - lm[LEFT_HIP].x,
            )))

        valid_hands = [(i, y) for i, y in enumerate(hands_y) if y is not None]
        if len(valid_hands) < 30:
            return self._empty_result(quality="low")

        # ── Top of backswing: minimum y (highest point on screen) ──
        # Restrict the search to the post-trigger window when the trigger is
        # within the clip — eliminates spurious "tops" from pre-swing motion
        # like setting up or waggling the club.
        trigger_idx = int(round(pre_seconds * fps))
        search_pool = [
            (i, y) for i, y in valid_hands
            if i >= max(0, trigger_idx - int(0.5 * fps))
        ]
        if len(search_pool) < 10:
            search_pool = valid_hands
        top_idx = min(search_pool, key=lambda p: p[1])[0]

        # ── Impact: prefer the audio peak; fall back to wrist velocity ──
        impact_idx = None
        if impact_offset is not None:
            candidate = int(round((pre_seconds + impact_offset) * fps))
            if top_idx < candidate < len(frames):
                impact_idx = candidate
        if impact_idx is None:
            impact_idx = self._wrist_velocity_impact(hands_y, top_idx)
        if impact_idx is None:
            impact_idx = min(top_idx + int(0.25 * fps), len(frames) - 1)

        # ── Address: walk back from the top until the hands have been still ──
        stillness_frames = max(5, int(self.QUIESCENCE_SECONDS * fps))
        address_idx = self._find_quiescence_back(
            hands_x, hands_y, top_idx, stillness_frames, self.QUIESCENCE_THRESHOLD,
        )
        if address_idx is None:
            address_idx = max(0, top_idx - int(0.7 * fps))

        # ── Finish: walk forward from impact until the hands plateau ──
        finish_idx = self._find_quiescence_forward(
            hands_x, hands_y, impact_idx, stillness_frames, self.QUIESCENCE_THRESHOLD,
        )
        if finish_idx is None:
            finish_idx = len(frames) - 1

        # ── Quality gate ──
        swing_window = list(range(address_idx, finish_idx + 1))
        valid_in_window = sum(1 for i in swing_window if hands_y[i] is not None)
        ratio = valid_in_window / max(1, len(swing_window))
        quality = "high" if ratio >= self.QUALITY_VALID_RATIO else "low"

        # ── Tempo ──
        backswing_frames = max(0, top_idx - address_idx)
        downswing_frames = max(1, impact_idx - top_idx)
        tempo_ratio = round(backswing_frames / downswing_frames, 2) if backswing_frames > 0 else None

        # ── Head stability across the actual swing (address → impact) ──
        head_stability = None
        head_pts = [
            nose_positions[i] for i in range(address_idx, impact_idx + 1)
            if 0 <= i < len(nose_positions) and nose_positions[i] is not None
        ]
        if len(head_pts) >= 5:
            head_stability = round(
                float(np.std([p[0] for p in head_pts]) +
                      np.std([p[1] for p in head_pts])) * 100, 2
            )

        # ── Rotation: address vs top, with a small search window for missing frames ──
        hip_rotation = self._angle_diff(hip_angles, address_idx, top_idx)
        shoulder_rotation = self._angle_diff(shoulder_angles, address_idx, top_idx)

        phases = [
            {"phase": SwingPhase.ADDRESS, "start_frame": address_idx, "end_frame": address_idx + 1},
            {"phase": SwingPhase.BACKSWING, "start_frame": address_idx, "end_frame": top_idx},
            {"phase": SwingPhase.TOP, "start_frame": top_idx, "end_frame": top_idx + 1},
            {"phase": SwingPhase.DOWNSWING, "start_frame": top_idx, "end_frame": impact_idx},
            {"phase": SwingPhase.IMPACT, "start_frame": impact_idx, "end_frame": impact_idx + 1},
            {"phase": SwingPhase.FOLLOW_THROUGH, "start_frame": impact_idx, "end_frame": finish_idx},
        ]

        result = {
            "phases": phases,
            "tempo_ratio": tempo_ratio,
            "backswing_frames": backswing_frames,
            "downswing_frames": downswing_frames,
            "head_stability": head_stability,
            "hip_rotation": hip_rotation,
            "shoulder_rotation": shoulder_rotation,
            "quality": quality,
        }

        # Don't publish meaningless metrics when MediaPipe lost the body.
        if quality == "low":
            for k in ("tempo_ratio", "head_stability", "hip_rotation", "shoulder_rotation"):
                result[k] = None
        return result

    @staticmethod
    def _empty_result(quality: str) -> dict:
        return {
            "phases": [],
            "tempo_ratio": None,
            "backswing_frames": None,
            "downswing_frames": None,
            "head_stability": None,
            "hip_rotation": None,
            "shoulder_rotation": None,
            "quality": quality,
        }

    @staticmethod
    def _wrist_velocity_impact(ys: list, top_idx: int) -> int | None:
        """Frame with maximum downward wrist velocity after the top."""
        best = None
        for i in range(top_idx + 1, len(ys)):
            if ys[i] is None or ys[i - 1] is None:
                continue
            v = ys[i] - ys[i - 1]
            if best is None or v > best[1]:
                best = (i, v)
        return best[0] if best else None

    @staticmethod
    def _find_quiescence_back(
        xs: list, ys: list, end_idx: int, window: int, threshold: float,
    ) -> int | None:
        """Walk back from end_idx looking for `window` consecutive frames where
        per-frame hand displacement stayed below `threshold`. Returns the index
        at the *end* of that quiet window (i.e. the latest still frame), which
        is where the swing actually started moving from."""
        for i in range(end_idx - 1, window, -1):
            if not all(
                xs[j] is not None and ys[j] is not None
                for j in range(i - window, i + 1)
            ):
                continue
            max_disp = max(
                math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
                for j in range(i - window, i)
            )
            if max_disp < threshold:
                return i
        return None

    @staticmethod
    def _find_quiescence_forward(
        xs: list, ys: list, start_idx: int, window: int, threshold: float,
    ) -> int | None:
        """Walk forward from start_idx looking for `window` consecutive frames
        where per-frame hand displacement stayed below `threshold`. Returns
        the index at the *start* of the plateau (first still frame after
        follow-through stops)."""
        n = len(xs)
        for i in range(start_idx + 1, n - window):
            if not all(
                xs[j] is not None and ys[j] is not None
                for j in range(i, i + window + 1)
            ):
                continue
            max_disp = max(
                math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
                for j in range(i, i + window)
            )
            if max_disp < threshold:
                return i
        return None

    @classmethod
    def _angle_diff(cls, angles: list, idx_a: int, idx_b: int) -> float | None:
        """Absolute angle difference between two indices, with a small search
        window if the landmark is missing at the exact frame."""
        a = cls._nearest_valid(angles, idx_a)
        b = cls._nearest_valid(angles, idx_b)
        if a is None or b is None:
            return None
        return round(abs(b - a), 1)

    @staticmethod
    def _nearest_valid(vals: list, idx: int, search_radius: int = 3):
        """Value at idx, or the nearest non-None within ±search_radius."""
        if 0 <= idx < len(vals) and vals[idx] is not None:
            return vals[idx]
        for r in range(1, search_radius + 1):
            for di in (-r, r):
                j = idx + di
                if 0 <= j < len(vals) and vals[j] is not None:
                    return vals[j]
        return None
