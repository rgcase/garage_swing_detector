"""
Swing detector: uses frame differencing to detect golf swings.

A golf swing in a garage is a distinctive event: a burst of fast
motion (the swing itself, ~1-2 seconds) bracketed by relative
stillness (address and finish positions). We detect this by:

1. Downscale frames for speed
2. Convert to grayscale + Gaussian blur
3. Compute absolute difference between consecutive frames
4. Threshold to get binary motion mask
5. Calculate % of ROI pixels that are "moving"
6. Trigger when motion exceeds threshold, with cooldown

This works extremely well in a controlled environment like a garage
with consistent lighting and static background.
"""

import logging
import time
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


class SwingDetector:
    """Per-camera swing detector using frame differencing."""

    def __init__(
        self,
        camera_name: str,
        motion_threshold: int = 25,
        motion_area_pct: float = 1.5,
        cooldown_seconds: float = 5.0,
        roi: list[float] | None = None,
        on_swing=None,
    ):
        self.camera_name = camera_name
        self.motion_threshold = motion_threshold
        self.motion_area_pct = motion_area_pct
        self.cooldown_seconds = cooldown_seconds
        self.roi = roi  # [x, y, w, h] normalized 0.0-1.0
        self.on_swing = on_swing  # callback(SwingEvent)

        self._prev_gray: np.ndarray | None = None
        self._last_trigger_time: float = 0
        self._analysis_scale = 0.25  # Process at 1/4 resolution for speed

    def process_frame(self, tf: TimestampedFrame):
        """Called for each new frame. Checks for swing-like motion."""
        frame = tf.frame

        # Downscale for speed
        small = cv2.resize(
            frame, None,
            fx=self._analysis_scale,
            fy=self._analysis_scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return

        # Frame differencing
        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        # Apply ROI if configured
        if self.roi:
            h, w = delta.shape
            rx, ry, rw, rh = self.roi
            x1, y1 = int(rx * w), int(ry * h)
            x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
            delta = delta[y1:y2, x1:x2]

        # Threshold
        _, thresh = cv2.threshold(delta, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # Calculate motion percentage
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_pct = (motion_pixels / total_pixels) * 100

        # Check trigger conditions
        if motion_pct >= self.motion_area_pct:
            now = tf.timestamp
            elapsed = now - self._last_trigger_time
            if elapsed >= self.cooldown_seconds:
                self._last_trigger_time = now
                event = SwingEvent(
                    trigger_time=now,
                    camera_name=self.camera_name,
                    motion_level=motion_pct,
                )
                logger.info(
                    f"[{self.camera_name}] Swing detected! "
                    f"motion={motion_pct:.1f}% (threshold={self.motion_area_pct}%)"
                )
                if self.on_swing:
                    self.on_swing(event)
