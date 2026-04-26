"""
Clip saver: when a swing is detected, waits for the post-swing
buffer to fill, then extracts the relevant frames from the
circular buffer and writes them as an H.264 MP4 file via FFmpeg.
"""

import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

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
        max_storage_mb: float | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps
        self.width = width
        self.height = height
        self.max_storage_bytes = int(max_storage_mb * 1024 * 1024) if max_storage_mb else None

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

        # Write H.264 MP4 via FFmpeg (plays in Safari/iOS unlike mp4v)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            # Dense keyframes (~1 per second at 30fps) so the web UI can
            # scrub frame-accurately. Default GOP of 250 leaves ~one keyframe
            # per clip, which makes mid-clip seeks render frozen frames.
            "-g", str(int(self.fps)),
            "-keyint_min", str(int(self.fps)),
            "-sc_threshold", "0",
            "-movflags", "+faststart",
            str(filepath),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        for tf in frames:
            proc.stdin.write(tf.frame.tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace")
            logger.error(f"[{camera_name}] FFmpeg clip encode failed: {stderr}")
            return None

        duration = frames[-1].timestamp - frames[0].timestamp
        logger.info(
            f"[{camera_name}] Saved {filepath.name} "
            f"({len(frames)} frames, {duration:.1f}s)"
        )

        self._enforce_storage_limit()
        return filename

    def _enforce_storage_limit(self):
        """Delete oldest clips if total storage exceeds the configured limit."""
        if not self.max_storage_bytes:
            return

        clips = sorted(self.output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in clips)

        while total > self.max_storage_bytes and len(clips) > 1:
            oldest = clips.pop(0)
            size = oldest.stat().st_size
            oldest.unlink()
            total -= size
            logger.info(f"Storage limit: deleted {oldest.name} ({size // 1024}KB)")

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
