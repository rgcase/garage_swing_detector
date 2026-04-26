"""
Web server: FastAPI app serving the review/tagging UI and API.
"""

import asyncio
import io
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from db import SwingDB
from stream_receiver import CircularFrameBuffer, StreamReceiver
from swing_detector import SwingDetector

logger = logging.getLogger(__name__)


class TagRequest(BaseModel):
    tag: str  # "good", "bad", or "neutral"


class NotesRequest(BaseModel):
    notes: str


class ROIRequest(BaseModel):
    x: float
    y: float
    w: float
    h: float


def _camera_status(receivers: list[StreamReceiver], buffers: dict[str, CircularFrameBuffer]) -> list[dict]:
    """Build per-camera status info."""
    now = time.time()
    cameras = []
    for recv in receivers:
        buf = buffers.get(recv.name)
        latest = buf.get_latest() if buf else None
        last_ts = latest.timestamp if latest else None
        # Consider "connected" if we got a frame in the last 5 seconds
        connected = last_ts is not None and (now - last_ts) < 5.0
        cameras.append({
            "name": recv.name,
            "port": recv.port,
            "connected": connected,
            "frame_count": buf.frame_count if buf else 0,
            "seconds_ago": round(now - last_ts, 1) if last_ts else None,
        })
    return cameras


def create_app(
    db: SwingDB,
    clips_dir: str,
    receivers: list[StreamReceiver] | None = None,
    buffers: dict[str, CircularFrameBuffer] | None = None,
    detectors: list[SwingDetector] | None = None,
    config: dict | None = None,
    config_path: str | None = None,
) -> FastAPI:
    app = FastAPI(title="swing-cam", version="0.1.0")

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    _receivers = receivers or []
    _buffers = buffers or {}
    _detectors = {d.camera_name: d for d in (detectors or [])}
    _config = config or {}
    _config_path = config_path
    _sse_subscribers: list[asyncio.Queue] = []

    # Serve clip files
    clips_path = Path(clips_dir)
    clips_path.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_path)), name="clips")

    # ── Pages ──

    def _group_into_sessions(swings, gap_minutes=30):
        """Group swings into practice sessions based on time gaps."""
        if not swings:
            return []

        sessions = []
        current_session = {"swings": [swings[0]]}

        for swing in swings[1:]:
            try:
                prev_ts = datetime.fromisoformat(current_session["swings"][-1].timestamp)
                curr_ts = datetime.fromisoformat(swing.timestamp)
                gap = abs((prev_ts - curr_ts).total_seconds())
            except (ValueError, AttributeError):
                gap = 0

            if gap > gap_minutes * 60:
                sessions.append(current_session)
                current_session = {"swings": [swing]}
            else:
                current_session["swings"].append(swing)

        sessions.append(current_session)

        # Add session summaries
        for session in sessions:
            s = session["swings"]
            session["start"] = s[-1].timestamp  # oldest (swings are desc)
            session["end"] = s[0].timestamp
            session["count"] = len(s)
            session["good"] = sum(1 for sw in s if sw.tag == "good")
            session["bad"] = sum(1 for sw in s if sw.tag == "bad")

        return sessions

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, tag: str | None = None):
        swings = db.list_swings(limit=200, tag_filter=tag)
        stats = db.get_stats()
        cameras = _camera_status(_receivers, _buffers)
        sessions = _group_into_sessions(swings)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "sessions": sessions,
                "stats": stats,
                "active_filter": tag,
                "cameras": cameras,
            },
        )

    @app.get("/live", response_class=HTMLResponse)
    async def live_page(request: Request):
        cameras = _camera_status(_receivers, _buffers)
        return templates.TemplateResponse(
            "live.html",
            {"request": request, "cameras": cameras},
        )

    @app.get("/api/stream/{camera_name}")
    async def api_stream(camera_name: str):
        """MJPEG stream of live frames from a camera."""
        buf = _buffers.get(camera_name)
        if not buf:
            raise HTTPException(404, f"Camera '{camera_name}' not found")

        async def mjpeg_generator():
            last_index = -1
            while True:
                latest = buf.get_latest()
                if latest is None or latest.index == last_index:
                    await asyncio.sleep(0.033)  # ~30fps cap
                    continue
                last_index = latest.index
                _, jpeg = cv2.imencode(
                    ".jpg", latest.frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── API ──

    @app.get("/api/swings")
    async def api_list_swings(limit: int = 50, offset: int = 0, tag: str | None = None):
        return {
            "swings": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp,
                    "tag": s.tag,
                    "notes": s.notes,
                    "clips": s.clips,
                }
                for s in db.list_swings(limit=limit, offset=offset, tag_filter=tag)
            ],
            "stats": db.get_stats(),
        }

    @app.get("/api/swings/{swing_id}")
    async def api_get_swing(swing_id: str):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        return {
            "id": swing.id,
            "timestamp": swing.timestamp,
            "tag": swing.tag,
            "notes": swing.notes,
            "clips": swing.clips,
        }

    @app.post("/api/swings/{swing_id}/tag")
    async def api_tag_swing(swing_id: str, req: TagRequest):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        db.tag_swing(swing_id, req.tag)
        return {"ok": True, "swing_id": swing_id, "tag": req.tag}

    @app.post("/api/swings/{swing_id}/notes")
    async def api_update_notes(swing_id: str, req: NotesRequest):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        db.update_notes(swing_id, req.notes)
        return {"ok": True, "swing_id": swing_id}

    @app.get("/api/swings/{swing_id}/export")
    async def api_export_swing(swing_id: str):
        """Export a swing as a single side-by-side MP4 (or single clip if only one camera)."""
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        if not swing.clips:
            raise HTTPException(404, "No clips for this swing")

        clips_path = Path(clips_dir)

        if len(swing.clips) == 1:
            # Single camera — just serve the file
            clip_file = clips_path / swing.clips[0]["filepath"]
            if not clip_file.exists():
                raise HTTPException(404, "Clip file not found")
            return FileResponse(
                clip_file,
                media_type="video/mp4",
                filename=f"swing_{swing_id}.mp4",
            )

        # Multiple cameras — stitch side-by-side with FFmpeg
        input_files = []
        for clip in swing.clips:
            p = clips_path / clip["filepath"]
            if not p.exists():
                raise HTTPException(404, f"Clip file not found: {clip['filepath']}")
            input_files.append(str(p))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # FFmpeg hstack filter: place videos side-by-side
        inputs = []
        for f in input_files:
            inputs.extend(["-i", f])

        # Scale all inputs to the same height, then hstack
        n = len(input_files)
        filter_parts = []
        for i in range(n):
            filter_parts.append(f"[{i}:v]scale=-2:720[v{i}]")
        stack_inputs = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{stack_inputs}hstack=inputs={n}[out]")
        filter_str = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_path,
        ]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            logger.error(f"Export FFmpeg failed: {proc.stderr.decode(errors='replace')}")
            raise HTTPException(500, "Failed to generate export")

        return FileResponse(
            tmp_path,
            media_type="video/mp4",
            filename=f"swing_{swing_id}_combined.mp4",
        )

    @app.delete("/api/swings/{swing_id}")
    async def api_delete_swing(swing_id: str):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        db.delete_swing(swing_id)
        return {"ok": True, "deleted": swing_id}

    @app.get("/api/swings/{swing_id}/analysis")
    async def api_swing_analysis(swing_id: str):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        return db.get_analysis(swing_id)

    @app.get("/api/stats")
    async def api_stats():
        return db.get_stats()

    @app.get("/api/status")
    async def api_status():
        cameras = _camera_status(_receivers, _buffers)
        return {
            "cameras": cameras,
            "stats": db.get_stats(),
        }

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_page(request: Request, a: str = "", b: str = ""):
        swing_a = db.get_swing(a) if a else None
        swing_b = db.get_swing(b) if b else None
        all_swings = db.list_swings(limit=200)
        return templates.TemplateResponse(
            "compare.html",
            {
                "request": request,
                "swing_a": swing_a,
                "swing_b": swing_b,
                "swings": all_swings,
            },
        )

    # ── Debug / Tuning ──

    @app.get("/api/motion/{camera_name}")
    async def api_motion(camera_name: str):
        """Return recent motion levels for the debug dashboard."""
        detector = _detectors.get(camera_name)
        if not detector:
            raise HTTPException(404, f"Camera '{camera_name}' not found")
        data = list(detector.recent_motion)
        return {
            "camera": camera_name,
            "threshold": detector.motion_area_pct,
            "motion": [{"t": t, "pct": round(p, 2)} for t, p in data],
        }

    @app.get("/debug", response_class=HTMLResponse)
    async def debug_page(request: Request):
        cameras = _camera_status(_receivers, _buffers)
        return templates.TemplateResponse(
            "debug.html",
            {"request": request, "cameras": cameras},
        )

    # ── Calibration ──

    @app.get("/api/snapshot/{camera_name}")
    async def api_snapshot(camera_name: str):
        """Return a JPEG snapshot of the latest frame from a camera."""
        buf = _buffers.get(camera_name)
        if not buf:
            raise HTTPException(404, f"Camera '{camera_name}' not found")
        latest = buf.get_latest()
        if latest is None:
            raise HTTPException(503, f"No frames from '{camera_name}'")
        _, jpeg = cv2.imencode(".jpg", latest.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")

    @app.get("/calibrate", response_class=HTMLResponse)
    async def calibrate_page(request: Request):
        cameras = _camera_status(_receivers, _buffers)
        current_roi = _config.get("detection", {}).get("roi")
        return templates.TemplateResponse(
            "calibrate.html",
            {
                "request": request,
                "cameras": cameras,
                "current_roi": current_roi,
            },
        )

    @app.post("/api/roi/{camera_name}")
    async def api_set_roi(camera_name: str, req: ROIRequest):
        """Update the ROI in config and apply it live to the detector."""
        roi = [round(req.x, 4), round(req.y, 4), round(req.w, 4), round(req.h, 4)]

        # Apply to running detector
        detector = _detectors.get(camera_name)
        if detector:
            detector.roi = roi

        # Save to config file
        if _config_path:
            _config["detection"]["roi"] = roi
            with open(_config_path, "w") as f:
                yaml.dump(_config, f, default_flow_style=False, sort_keys=False)

        return {"ok": True, "roi": roi}

    @app.delete("/api/roi")
    async def api_clear_roi():
        """Clear the ROI (use full frame)."""
        for det in _detectors.values():
            det.roi = None
        if _config_path:
            _config["detection"]["roi"] = None
            with open(_config_path, "w") as f:
                yaml.dump(_config, f, default_flow_style=False, sort_keys=False)
        return {"ok": True, "roi": None}

    # ── Server-Sent Events ──

    @app.get("/api/events")
    async def api_events():
        """SSE stream for live swing notifications."""
        queue: asyncio.Queue = asyncio.Queue()
        _sse_subscribers.append(queue)

        async def event_stream():
            try:
                while True:
                    # Send heartbeat every 15s to keep connection alive
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                _sse_subscribers.remove(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def notify_new_swing(swing_id: str):
        """Called from main.py (sync context) to push a swing event to SSE clients."""
        swing = db.get_swing(swing_id)
        if not swing:
            return
        event = {
            "type": "new_swing",
            "swing": {
                "id": swing.id,
                "timestamp": swing.timestamp,
                "tag": swing.tag,
                "clips": swing.clips,
            },
        }
        for q in list(_sse_subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # Expose the notify function on the app for main.py to call
    app.notify_new_swing = notify_new_swing

    return app
