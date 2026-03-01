"""
Clip saver: when a swing is detected, waits for the post-swing
buffer to fill, then extracts the relevant frames from the
circular buffer and writes them as an MP4 file via OpenCV.
"""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from stream_receiver import CircularFrameBuffer, TimestampedFrame

logger = logging.getLogger(__name__)


class ClipSaver:
    """Saves clips from the circular buffer to disk."""

    def __init__(
        self,
        output_dir: str,
        pre_seconds: float = 3.0,
        post_seconds: float = 6.0,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps
        self.width = width
        self.height = height

    def save_clip(
        self,
        buffer: CircularFrameBuffer,
        trigger_time: float,
        camera_name: str,
        swing_id: str,
    ) -> str | None:
        """
        Wait for post-swing frames, extract from buffer, write MP4.
        Returns the output file path, or None on failure.

        This is meant to be called in a separate thread so it doesn't
        block the frame processing pipeline.
        """
        # Wait for post-swing buffer to fill
        wait_time = self.post_seconds + 0.5  # small margin
        logger.info(
            f"[{camera_name}] Waiting {wait_time:.1f}s for post-swing frames..."
        )
        time.sleep(wait_time)

        # Extract frames from buffer
        start_time = trigger_time - self.pre_seconds
        end_time = trigger_time + self.post_seconds
        frames = buffer.get_range(start_time, end_time)

        if len(frames) < 10:
            logger.warning(
                f"[{camera_name}] Only got {len(frames)} frames, skipping clip"
            )
            return None

        # Generate filename
        dt = datetime.fromtimestamp(trigger_time)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        filename = f"swing_{timestamp_str}_{camera_name}.mp4"
        filepath = self.output_dir / filename

        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(filepath), fourcc, self.fps, (self.width, self.height)
        )

        for tf in frames:
            writer.write(tf.frame)

        writer.release()

        duration = frames[-1].timestamp - frames[0].timestamp
        logger.info(
            f"[{camera_name}] Saved {filepath.name} "
            f"({len(frames)} frames, {duration:.1f}s)"
        )
        return str(filepath)

    def save_clip_async(
        self,
        buffer: CircularFrameBuffer,
        trigger_time: float,
        camera_name: str,
        swing_id: str,
        callback=None,
    ):
        """
        Non-blocking clip save. Spawns a thread, calls callback(filepath)
        when done.
        """
        def _run():
            filepath = self.save_clip(buffer, trigger_time, camera_name, swing_id)
            if callback:
                callback(swing_id, camera_name, filepath)

        t = threading.Thread(target=_run, daemon=True, name=f"clip-{swing_id}")
        t.start()
