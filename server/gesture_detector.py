"""
Gesture detector: watches for thumbs up/down after a swing is detected.

Uses MediaPipe Hands to detect hand landmarks and classify gestures.
Only runs during a short window after a swing to save CPU and avoid
false positives.
"""

import logging
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning(
        "mediapipe not installed — gesture detection disabled. "
        "Install with: pip install mediapipe"
    )


class GestureDetector:
    """Detects thumbs up/down gestures from video frames."""

    # Minimum consecutive frames with same gesture to confirm
    CONFIRM_FRAMES = 10
    # Process every Nth frame to save CPU
    FRAME_SKIP = 5

    def __init__(
        self,
        watch_seconds: float = 10.0,
        on_gesture=None,
    ):
        """
        Args:
            watch_seconds: How long to watch for gestures after a swing.
            on_gesture: Callback(swing_id, gesture) where gesture is 'good' or 'bad'.
        """
        self.watch_seconds = watch_seconds
        self.on_gesture = on_gesture

        self._active = False
        self._active_swing_id: str | None = None
        self._active_until: float = 0
        self._frame_counter = 0
        self._gesture_streak: dict[str, int] = {}
        self._lock = threading.Lock()

        if HAS_MEDIAPIPE:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        else:
            self._hands = None

    def start_watching(self, swing_id: str):
        """Begin gesture detection window after a swing."""
        if not HAS_MEDIAPIPE:
            return
        with self._lock:
            self._active = True
            self._active_swing_id = swing_id
            self._active_until = time.time() + self.watch_seconds
            self._frame_counter = 0
            self._gesture_streak = {}
            logger.info(f"Gesture detection active for {self.watch_seconds}s (swing {swing_id[:8]})")

    def process_frame(self, frame):
        """Process a frame during the gesture detection window.

        Args:
            frame: BGR numpy array from the camera.
        """
        if not self._active or not self._hands:
            return

        now = time.time()
        with self._lock:
            if now > self._active_until:
                self._active = False
                logger.debug("Gesture detection window expired")
                return

            self._frame_counter += 1
            if self._frame_counter % self.FRAME_SKIP != 0:
                return

            swing_id = self._active_swing_id

        # Run hand detection (outside lock — this is the expensive part)
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._gesture_streak.clear()
            return

        hand = results.multi_hand_landmarks[0]
        gesture = self._classify_gesture(hand)

        if gesture:
            count = self._gesture_streak.get(gesture, 0) + 1
            self._gesture_streak[gesture] = count

            if count >= self.CONFIRM_FRAMES:
                with self._lock:
                    self._active = False
                logger.info(f"Gesture confirmed: {gesture} for swing {swing_id[:8]}")
                self._play_feedback(gesture)
                if self.on_gesture:
                    self.on_gesture(swing_id, gesture)
        else:
            self._gesture_streak.clear()

    @staticmethod
    def _classify_gesture(hand) -> str | None:
        """Classify hand landmarks as thumbs up, thumbs down, or None.

        MediaPipe hand landmarks:
            0: WRIST
            1: THUMB_CMC, 2: THUMB_MCP, 3: THUMB_IP, 4: THUMB_TIP
            5: INDEX_MCP, 6: INDEX_PIP, 7: INDEX_DIP, 8: INDEX_TIP
            9: MIDDLE_MCP, 10: MIDDLE_PIP, 11: MIDDLE_DIP, 12: MIDDLE_TIP
            13: RING_MCP, 14: RING_PIP, 15: RING_DIP, 16: RING_TIP
            17: PINKY_MCP, 18: PINKY_PIP, 19: PINKY_DIP, 20: PINKY_TIP
        """
        lm = hand.landmark

        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        # Check if other fingers are curled (tip below PIP joint)
        fingers_curled = all(
            lm[tip].y > lm[pip].y  # y increases downward in image coords
            for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
        )

        if not fingers_curled:
            return None

        # Thumb extended upward: tip significantly above MCP
        thumb_up = (thumb_mcp.y - thumb_tip.y) > 0.05
        # Thumb extended downward: tip significantly below MCP
        thumb_down = (thumb_tip.y - thumb_mcp.y) > 0.05

        if thumb_up:
            return "good"
        elif thumb_down:
            return "bad"
        return None

    @staticmethod
    def _play_feedback(gesture: str):
        """Play audio feedback on macOS."""
        if sys.platform != "darwin":
            return
        # Different sounds for good vs bad
        sound = (
            "/System/Library/Sounds/Glass.aiff" if gesture == "good"
            else "/System/Library/Sounds/Basso.aiff"
        )
        try:
            subprocess.Popen(
                ["afplay", sound],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass
