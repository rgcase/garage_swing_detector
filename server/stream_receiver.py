"""
Stream receiver: accepts an H.264 TCP stream from a Pi,
decodes frames via OpenCV/FFmpeg, and stores them in a
thread-safe circular buffer.
"""

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimestampedFrame:
    """A decoded frame with its capture timestamp."""
    frame: np.ndarray
    timestamp: float  # time.time()
    index: int        # monotonic frame counter


class CircularFrameBuffer:
    """Thread-safe ring buffer of decoded frames."""

    def __init__(self, max_seconds: float, fps: float):
        self.max_frames = int(max_seconds * fps)
        self._buffer: deque[TimestampedFrame] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self._frame_count = 0

    def push(self, frame: np.ndarray, timestamp: float) -> TimestampedFrame:
        tf = TimestampedFrame(
            frame=frame, timestamp=timestamp, index=self._frame_count
        )
        with self._lock:
            self._buffer.append(tf)
            self._frame_count += 1
        return tf

    def get_range(self, start_time: float, end_time: float) -> list[TimestampedFrame]:
        """Return all frames within the given time window."""
        with self._lock:
            return [
                f for f in self._buffer
                if start_time <= f.timestamp <= end_time
            ]

    def get_latest(self) -> TimestampedFrame | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    @property
    def frame_count(self) -> int:
        return self._frame_count


class StreamReceiver:
    """
    Listens for an incoming H.264 TCP stream from a Pi,
    decodes it, and feeds frames into a circular buffer.

    Uses FFmpeg as a subprocess to handle the TCP listening
    and H.264 → raw frame decoding, then reads raw frames
    from FFmpeg's stdout via pipe.
    """

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        buffer: CircularFrameBuffer,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ):
        self.name = name
        self.host = host
        self.port = port
        self.buffer = buffer
        self.width = width
        self.height = height
        self.fps = fps

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_frame_callbacks: list = []

    def on_frame(self, callback):
        """Register a callback that receives each decoded TimestampedFrame."""
        self._on_frame_callbacks.append(callback)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name=f"recv-{self.name}", daemon=True
        )
        self._thread.start()
        logger.info(f"[{self.name}] Receiver started, listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """Outer loop: restarts FFmpeg if the stream disconnects."""
        while self._running:
            try:
                self._receive_stream()
            except Exception as e:
                logger.warning(f"[{self.name}] Stream error: {e}. Reconnecting in 3s...")
                time.sleep(3)

    def _receive_stream(self):
        """
        Launch FFmpeg to listen on TCP, decode H.264, and pipe raw
        BGR24 frames to us via stdout.
        """
        frame_size = self.width * self.height * 3  # BGR24

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            # Input: listen for TCP connection from the Pi
            "-f", "h264",
            "-i", f"tcp://{self.host}:{self.port}?listen=1",
            # Output: raw BGR frames to stdout
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",  # no audio
            "pipe:1",
        ]

        logger.info(f"[{self.name}] Starting FFmpeg: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        while self._running:
            raw = self._process.stdout.read(frame_size)
            if len(raw) < frame_size:
                logger.warning(f"[{self.name}] Short read ({len(raw)}/{frame_size}), stream likely ended")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            timestamp = time.time()
            tf = self.buffer.push(frame, timestamp)

            for cb in self._on_frame_callbacks:
                try:
                    cb(tf)
                except Exception as e:
                    logger.error(f"[{self.name}] Frame callback error: {e}")

        self._process.terminate()
        self._process.wait()
        self._process = None
