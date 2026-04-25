"""
Swing detector: multi-stage pipeline to detect golf swings.

A golf swing in a garage is distinctive: a burst of very fast, concentrated
motion (the club + arms, ~0.3s) bracketed by stillness (address and finish).
We use multiple signals to distinguish swings from other motion:

Stage 1 — Motion gate (cheap):
    Frame differencing detects any motion. Requires a still→spike pattern.

Stage 2 — Optical flow analysis:
    Dense optical flow on the ROI checks that the motion is fast,
    concentrated, and brief — like a swing, not like walking.

Stage 3 — Duration filter:
    A real swing's high-motion phase is very short (0.2-0.8s).
    Sustained motion (>1s) is rejected.

Each stage produces a confidence score. The final confidence must exceed
a threshold to trigger. Multi-camera agreement (handled in main.py)
can boost confidence.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from stream_receiver import TimestampedFrame

logger = logging.getLogger(__name__)


@dataclass
class SwingEvent:
    """A detected swing event."""
    trigger_time: float    # Timestamp of the trigger frame
    camera_name: str
    motion_level: float    # Peak motion % that triggered detection
    confidence: float      # 0.0-1.0 detection confidence


class SwingDetector:
    """Per-camera swing detector using a multi-stage pipeline."""

    # How long the scene must be still before a spike counts
    STILL_REQUIRED_SECONDS = 1.0
    # Motion pct below this is considered "still"
    STILL_THRESHOLD_FACTOR = 0.3

    # Optical flow thresholds
    FLOW_SPEED_MIN = 3.0       # Min avg flow magnitude in the active region
    FLOW_CONCENTRATION_MIN = 0.15  # Min fraction of flow in the peak region

    # Duration filter: high-motion phase must be this short
    SPIKE_MAX_SECONDS = 1.0

    # Minimum confidence to trigger (0.0-1.0)
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        camera_name: str,
        motion_threshold: int = 25,
        motion_area_pct: float = 1.5,
        cooldown_seconds: float = 5.0,
        roi: list[float] | None = None,
        confidence_threshold: float = 0.5,
        spike_max_seconds: float = 1.0,
        on_swing=None,
    ):
        self.camera_name = camera_name
        self.motion_threshold = motion_threshold
        self.motion_area_pct = motion_area_pct
        self.cooldown_seconds = cooldown_seconds
        self.roi = roi
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.SPIKE_MAX_SECONDS = spike_max_seconds
        self.on_swing = on_swing

        self._prev_gray: np.ndarray | None = None
        self._prev_gray_full: np.ndarray | None = None  # for optical flow
        self._last_trigger_time: float = 0
        self._analysis_scale = 0.25

        # Temporal tracking
        self._motion_history: deque[tuple[float, float]] = deque(maxlen=120)
        self._still_threshold = self.motion_area_pct * self.STILL_THRESHOLD_FACTOR

        # Track the current spike for duration measurement
        self._spike_start: float | None = None
        self._spike_peak: float = 0
        self._spike_reported = False

        # Rolling motion data exposed for the debug dashboard
        self.recent_motion: deque[tuple[float, float]] = deque(maxlen=300)

    def _get_roi_slice(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Convert normalized ROI to pixel coordinates."""
        if self.roi:
            rx, ry, rw, rh = self.roi
            return int(ry * h), int((ry + rh) * h), int(rx * w), int((rx + rw) * w)
        return 0, h, 0, w

    def _was_still_before(self, now: float) -> bool:
        """Check if scene was still for STILL_REQUIRED_SECONDS before now."""
        cutoff = now - self.STILL_REQUIRED_SECONDS
        still_samples = [
            pct for ts, pct in self._motion_history
            if ts < now and ts >= cutoff
        ]
        if len(still_samples) < 5:
            return False
        return all(pct < self._still_threshold for pct in still_samples)

    def _compute_optical_flow_score(self, gray: np.ndarray) -> float:
        """Compute optical flow on the ROI and return a swing-likeness score.

        A golf swing produces:
        - High peak flow magnitude (club/arms moving fast)
        - Concentrated flow (motion in a small region, not diffuse)
        - Dominant vertical/arc direction

        Returns 0.0-1.0 score.
        """
        if self._prev_gray_full is None:
            return 0.0

        h, w = gray.shape
        y1, y2, x1, x2 = self._get_roi_slice(h, w)
        roi_curr = gray[y1:y2, x1:x2]
        roi_prev = self._prev_gray_full[y1:y2, x1:x2]

        if roi_curr.size == 0 or roi_prev.size == 0:
            return 0.0

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )

        # Flow magnitude
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # 1. Speed check: is the fast part fast enough?
        # Look at the top 10% of flow magnitudes
        top_10_pct = np.percentile(mag, 90)
        if top_10_pct < self.FLOW_SPEED_MIN:
            return 0.0

        speed_score = min(1.0, top_10_pct / (self.FLOW_SPEED_MIN * 4))

        # 2. Concentration check: is motion concentrated or diffuse?
        # A swing has high flow in a small area; walking has low flow everywhere
        threshold_mag = top_10_pct * 0.5
        active_pixels = np.count_nonzero(mag > threshold_mag)
        total_pixels = mag.size
        active_fraction = active_pixels / total_pixels

        # Sweet spot: 5-40% of pixels are active (concentrated but not too sparse)
        if active_fraction < 0.02 or active_fraction > 0.6:
            concentration_score = 0.2
        elif active_fraction < 0.05:
            concentration_score = 0.5
        elif active_fraction <= 0.4:
            concentration_score = 1.0
        else:
            concentration_score = 0.4

        # 3. Directionality: does the flow have a dominant direction?
        # A swing has coherent direction; random motion doesn't
        active_mask = mag > threshold_mag
        if np.any(active_mask):
            active_flow_x = flow[..., 0][active_mask]
            active_flow_y = flow[..., 1][active_mask]
            mean_x = np.mean(active_flow_x)
            mean_y = np.mean(active_flow_y)
            mean_mag = np.sqrt(mean_x**2 + mean_y**2)
            mean_individual_mag = np.mean(mag[active_mask])
            # Ratio of mean vector to mean magnitude — 1.0 = all same direction
            coherence = mean_mag / max(mean_individual_mag, 0.01)
            direction_score = min(1.0, coherence * 1.5)
        else:
            direction_score = 0.0

        # Weighted combination
        score = (speed_score * 0.4 + concentration_score * 0.3 + direction_score * 0.3)
        return score

    def process_frame(self, tf: TimestampedFrame):
        """Called for each new frame. Multi-stage swing detection."""
        frame = tf.frame
        now = tf.timestamp

        # Downscale for frame differencing
        small = cv2.resize(
            frame, None,
            fx=self._analysis_scale,
            fy=self._analysis_scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray_blurred
            self._prev_gray_full = gray
            return

        # ── Stage 1: Frame differencing (motion gate) ──
        delta = cv2.absdiff(self._prev_gray, gray_blurred)

        # Apply ROI for motion percentage calculation
        if self.roi:
            h, w = delta.shape
            rx, ry, rw, rh = self.roi
            x1, y1 = int(rx * w), int(ry * h)
            x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
            delta_roi = delta[y1:y2, x1:x2]
        else:
            delta_roi = delta

        _, thresh = cv2.threshold(delta_roi, self.motion_threshold, 255, cv2.THRESH_BINARY)
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_pct = (motion_pixels / total_pixels) * 100

        # Track history
        self._motion_history.append((now, motion_pct))
        self.recent_motion.append((now, motion_pct))

        is_moving = motion_pct >= self.motion_area_pct

        # ── Stage 3: Duration tracking ──
        # Track spike start/end to measure how long the motion lasts
        if is_moving:
            if self._spike_start is None:
                self._spike_start = now
                self._spike_peak = motion_pct
                self._spike_reported = False
            else:
                self._spike_peak = max(self._spike_peak, motion_pct)
        else:
            if self._spike_start is not None and not self._spike_reported:
                # Spike just ended — evaluate it
                spike_duration = now - self._spike_start
                self._evaluate_spike(now, spike_duration, gray)
            if not is_moving:
                if self._spike_start is not None and (now - self._spike_start) > self.SPIKE_MAX_SECONDS * 2:
                    # Spike has been going too long, abandon it
                    pass
                if not is_moving:
                    self._spike_start = None

        # Also check for spikes that have been going too long
        if self._spike_start is not None:
            spike_duration = now - self._spike_start
            if spike_duration > self.SPIKE_MAX_SECONDS * 2:
                # Too long — this is sustained motion, not a swing
                logger.debug(
                    f"[{self.camera_name}] Rejected: sustained motion "
                    f"({spike_duration:.1f}s > {self.SPIKE_MAX_SECONDS}s)"
                )
                self._spike_start = None
                self._spike_reported = True

        self._prev_gray = gray_blurred
        self._prev_gray_full = gray

    def _evaluate_spike(self, now: float, spike_duration: float, gray: np.ndarray):
        """Evaluate a completed motion spike to determine if it's a swing."""
        self._spike_reported = True

        elapsed = now - self._last_trigger_time
        if elapsed < self.cooldown_seconds:
            return

        # Check 1: Was it still before the spike?
        if not self._was_still_before(self._spike_start):
            logger.debug(
                f"[{self.camera_name}] Rejected: not still before spike"
            )
            return

        # Check 2: Duration — swing spikes are short
        if spike_duration > self.SPIKE_MAX_SECONDS:
            logger.debug(
                f"[{self.camera_name}] Rejected: spike too long "
                f"({spike_duration:.2f}s > {self.SPIKE_MAX_SECONDS}s)"
            )
            return

        duration_score = 1.0 if spike_duration < 0.6 else max(0.3, 1.0 - (spike_duration - 0.6))

        # Check 3: Optical flow — is this swing-like motion?
        flow_score = self._compute_optical_flow_score(gray)

        # Check 4: Motion intensity — swings produce high peak motion
        intensity_score = min(1.0, self._spike_peak / (self.motion_area_pct * 5))

        # Combined confidence
        confidence = (
            flow_score * 0.40 +
            duration_score * 0.30 +
            intensity_score * 0.30
        )

        logger.info(
            f"[{self.camera_name}] Spike analyzed: "
            f"flow={flow_score:.2f} duration={duration_score:.2f}({spike_duration:.2f}s) "
            f"intensity={intensity_score:.2f} → confidence={confidence:.2f} "
            f"(threshold={self.CONFIDENCE_THRESHOLD})"
        )

        if confidence >= self.CONFIDENCE_THRESHOLD:
            self._last_trigger_time = now
            event = SwingEvent(
                trigger_time=self._spike_start,
                camera_name=self.camera_name,
                motion_level=self._spike_peak,
                confidence=confidence,
            )
            logger.info(
                f"[{self.camera_name}] Swing detected! "
                f"confidence={confidence:.2f} peak_motion={self._spike_peak:.1f}%"
            )
            if self.on_swing:
                self.on_swing(event)
        else:
            logger.info(
                f"[{self.camera_name}] Motion rejected (confidence {confidence:.2f} "
                f"< {self.CONFIDENCE_THRESHOLD})"
            )
