"""
Audio receiver: listens for a single PCM TCP stream from a Pi mic and
maintains a rolling buffer of recent samples for impact detection.

The server only accepts ONE audio source at a time — once a client is
connected, the listen socket is closed so additional Pis get ECONNREFUSED
and can keep retrying. When the active source disconnects we re-open the
listener and the next retrying Pi takes over.
"""

import logging
import socket
import threading
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class AudioReceiver:
    """TCP listener for raw S16LE mono PCM, with a rolling sample buffer."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9558,
        sample_rate: int = 16000,
        buffer_seconds: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds

        # (timestamp, samples_array). Each chunk's timestamp is the wall
        # clock at the moment the last sample of the chunk was received,
        # which is good enough for ±0.3s correlation against video swings.
        max_chunks = int(buffer_seconds * sample_rate / 1024) + 16
        self._chunks: deque[tuple[float, np.ndarray]] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

        self._running = False
        self._thread: threading.Thread | None = None
        self._client_addr: tuple | None = None
        self._last_chunk_at: float = 0.0

    @property
    def is_connected(self) -> bool:
        # Heuristic: we got data in the last 2 seconds.
        return self._client_addr is not None and (time.time() - self._last_chunk_at) < 2.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="audio-recv", daemon=True
        )
        self._thread.start()
        logger.info(f"AudioReceiver listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                self._accept_and_read()
            except Exception as e:
                logger.warning(f"AudioReceiver error: {e}; retrying in 2s")
                time.sleep(2)

    def _accept_and_read(self):
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.settimeout(1.0)
        try:
            listen.bind((self.host, self.port))
            listen.listen(1)
            client = None
            while self._running and client is None:
                try:
                    client, addr = listen.accept()
                except socket.timeout:
                    continue
            if client is None:
                return
            # Closing the listen socket means subsequent connect()s get
            # ECONNREFUSED — exactly the "reject the second Pi" behavior.
            listen.close()
        finally:
            try:
                listen.close()
            except Exception:
                pass

        self._client_addr = addr
        logger.info(f"Audio source connected: {addr[0]}:{addr[1]}")
        self._last_chunk_at = time.time()
        try:
            client.settimeout(5.0)
            while self._running:
                # 16-bit mono samples → 2 bytes/sample. 4096 bytes = 2048 samples ≈ 128ms at 16kHz.
                raw = client.recv(4096)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.int16)
                ts = time.time()
                with self._lock:
                    self._chunks.append((ts, samples))
                self._last_chunk_at = ts
        except socket.timeout:
            logger.warning(f"Audio source {addr[0]}:{addr[1]} timed out")
        except Exception as e:
            logger.warning(f"Audio source {addr[0]}:{addr[1]} error: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass
            logger.info(f"Audio source disconnected: {addr[0]}:{addr[1]}")
            self._client_addr = None

    def find_impact(
        self,
        center_time: float,
        window: float = 0.3,
        threshold: float = 0.4,
    ) -> tuple[float, float] | None:
        """
        Search the buffer for a transient peak within [center-window, center+window].
        Returns (impact_time_offset_from_center, peak_amplitude_0_to_1) or None.

        threshold is normalized amplitude (1.0 = full-scale int16). Tune for
        your gain stage; 0.4 is a reasonable default for a club-on-ball strike
        on a USB lavalier within ~3m.
        """
        t_lo = center_time - window
        t_hi = center_time + window
        with self._lock:
            chunks = [(ts, s) for ts, s in self._chunks if ts >= t_lo - 0.5 and ts <= t_hi + 0.5]
        if not chunks:
            return None

        # Stitch into a single array along with per-sample timestamps.
        # Each chunk's recorded ts is the wall time at end-of-chunk; spread
        # samples uniformly back from there at the configured sample rate.
        all_samples: list[np.ndarray] = []
        all_ts: list[np.ndarray] = []
        for ts, s in chunks:
            n = len(s)
            if n == 0:
                continue
            chunk_ts = ts - np.arange(n - 1, -1, -1, dtype=np.float64) / self.sample_rate
            all_samples.append(s)
            all_ts.append(chunk_ts)
        if not all_samples:
            return None

        samples = np.concatenate(all_samples)
        ts_arr = np.concatenate(all_ts)
        mask = (ts_arr >= t_lo) & (ts_arr <= t_hi)
        if not np.any(mask):
            return None
        windowed = samples[mask]
        windowed_ts = ts_arr[mask]

        # Smooth slightly to find peaks (10ms window) and normalize to 0-1.
        abs_norm = np.abs(windowed.astype(np.float32)) / 32768.0
        smooth_n = max(1, self.sample_rate // 100)
        kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
        smoothed = np.convolve(abs_norm, kernel, mode="same")

        peak_idx = int(np.argmax(smoothed))
        peak_val = float(smoothed[peak_idx])
        if peak_val < threshold:
            return None
        peak_ts = float(windowed_ts[peak_idx])
        return peak_ts - center_time, peak_val
