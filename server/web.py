"""
Web server: FastAPI app serving the review/tagging UI and API.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from db import SwingDB

logger = logging.getLogger(__name__)


class TagRequest(BaseModel):
    tag: str  # "good", "bad", or "neutral"


class NotesRequest(BaseModel):
    notes: str


def create_app(db: SwingDB, clips_dir: str) -> FastAPI:
    app = FastAPI(title="swing-cam", version="0.1.0")

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Serve clip files
    clips_path = Path(clips_dir)
    clips_path.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=str(clips_path)), name="clips")

    # ── Pages ──

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, tag: str | None = None):
        swings = db.list_swings(limit=100, tag_filter=tag)
        stats = db.get_stats()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "swings": swings,
                "stats": stats,
                "active_filter": tag,
            },
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

    @app.delete("/api/swings/{swing_id}")
    async def api_delete_swing(swing_id: str):
        swing = db.get_swing(swing_id)
        if not swing:
            raise HTTPException(404, "Swing not found")
        db.delete_swing(swing_id)
        return {"ok": True, "deleted": swing_id}

    @app.get("/api/stats")
    async def api_stats():
        return db.get_stats()

    return app
