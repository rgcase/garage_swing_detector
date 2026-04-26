"""
Stream receiver: accepts video from Pi cameras (TCP push) or IP cameras
(RTSP/URL pull), decodes frames via FFmpeg, and stores them in a
thread-safe circular buffer.

Supported source formats:
  - TCP listen (Pi push):  "tcp://0.0.0.0:9556"  or  host + port in config
  - RTSP pull (IP camera): "rtsp://192.168.4.50:554/stream1"
  - Any FFmpeg input URL:  "http://...", "udp://...", etc.
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
    Receives video from a camera source, decodes it via FFmpeg,
    and feeds frames into a circular buffer.

    Supports two modes:
      - TCP listen: server waits for Pi to push H.264 (source starts with "tcp://")
      - Pull: server connects to an RTSP/HTTP/etc URL (anything else)
    """

    def __init__(
        self,
        name: str,
        source: str,
        buffer: CircularFrameBuffer,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ):
        self.name = name
        self.source = source
        self.buffer = buffer
        self.width = width
        self.height = height
        self.fps = fps

        # For status display
        self.port = self._extract_port()

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_frame_callbacks: list = []

    def _extract_port(self) -> int | None:
        """Extract port number from source URL for display purposes."""
        try:
            if ":" in self.source:
                # tcp://0.0.0.0:9556?listen=1 or rtsp://host:554/path
                part = self.source.split("://", 1)[-1]
                port_str = part.split(":")[1].split("/")[0].split("?")[0]
                return int(port_str)
        except (IndexError, ValueError):
            pass
        return None

    @property
    def is_connected(self) -> bool:
        """True if FFmpeg is running and receiving frames."""
        return self._running and self._process is not None

    def on_frame(self, callback):
        """Register a callback that receives each decoded TimestampedFrame."""
        self._on_frame_callbacks.append(callback)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name=f"recv-{self.name}", daemon=True
        )
        self._thread.start()
        logger.info(f"[{self.name}] Receiver started, source: {self.source}")

    def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def _build_ffmpeg_cmd(self) -> list[str]:
        """Build the FFmpeg command based on source type."""
        source = self.source
        is_tcp_listen = source.startswith("tcp://")
        is_rtsp = source.startswith("rtsp://")

        cmd = ["ffmpeg", "-y", "-loglevel", "warning"]

        if is_tcp_listen:
            # Pi push mode: listen for incoming TCP H.264 stream
            if "?listen" not in source:
                source += "?listen=1"
            cmd += ["-f", "h264", "-i", source]
        elif is_rtsp:
            # RTSP pull mode: connect to IP camera
            cmd += [
                "-rtsp_transport", "tcp",  # TCP is more reliable than UDP
                "-i", source,
            ]
        else:
            # Generic URL (http, udp, file, device, etc.)
            cmd += ["-i", source]

        # Output: raw BGR frames to stdout
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-an",
            "pipe:1",
        ]
        return cmd

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
        Launch FFmpeg to receive and decode the stream, piping raw
        BGR24 frames to us via stdout.
        """
        frame_size = self.width * self.height * 3  # BGR24
        cmd = self._build_ffmpeg_cmd()

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
