"""
SQLite database for swing records and clip metadata.
"""

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SwingRecord:
    id: str
    timestamp: str          # ISO format
    tag: str | None         # "good", "bad", or None (untagged)
    notes: str | None
    clips: list[dict]       # [{camera_name, angle, filepath}]


class SwingDB:
    def __init__(self, db_path: str = "./swings.db", clips_dir: str = "./clips"):
        self.db_path = db_path
        self.clips_dir = Path(clips_dir)
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS swings (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                tag TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                swing_id TEXT NOT NULL REFERENCES swings(id),
                camera_name TEXT NOT NULL,
                angle TEXT,
                filepath TEXT,
                trim_start REAL,
                trim_end REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_swings_timestamp ON swings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_clips_swing_id ON clips(swing_id);

            CREATE TABLE IF NOT EXISTS swing_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                swing_id TEXT NOT NULL REFERENCES swings(id),
                camera_name TEXT NOT NULL,
                tempo_ratio REAL,
                backswing_frames INTEGER,
                downswing_frames INTEGER,
                head_stability REAL,
                hip_rotation REAL,
                shoulder_rotation REAL,
                phases_json TEXT,
                landmarks_summary TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_analysis_swing ON swing_analysis(swing_id);
        """)

        # Migrations: ALTER TABLE for columns added after the original schema.
        # CREATE TABLE IF NOT EXISTS doesn't add columns to an existing table,
        # so we have to check and add explicitly.
        clip_cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)").fetchall()}
        if "trim_start" not in clip_cols:
            conn.execute("ALTER TABLE clips ADD COLUMN trim_start REAL")
        if "trim_end" not in clip_cols:
            conn.execute("ALTER TABLE clips ADD COLUMN trim_end REAL")

        conn.commit()
        conn.close()

    def generate_swing_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def create_swing(self, swing_id: str, timestamp: float) -> str:
        iso_ts = datetime.fromtimestamp(timestamp).isoformat()
        self._conn.execute(
            "INSERT INTO swings (id, timestamp) VALUES (?, ?)",
            (swing_id, iso_ts),
        )
        self._conn.commit()
        return swing_id

    def add_clip(
        self, swing_id: str, camera_name: str, angle: str, filepath: str,
        trim_start: float | None = None, trim_end: float | None = None,
    ):
        self._conn.execute(
            "INSERT INTO clips (swing_id, camera_name, angle, filepath, trim_start, trim_end) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (swing_id, camera_name, angle, filepath, trim_start, trim_end),
        )
        self._conn.commit()

    def tag_swing(self, swing_id: str, tag: str | None):
        self._conn.execute(
            "UPDATE swings SET tag = ? WHERE id = ?", (tag, swing_id)
        )
        self._conn.commit()

    def update_notes(self, swing_id: str, notes: str):
        self._conn.execute(
            "UPDATE swings SET notes = ? WHERE id = ?", (notes, swing_id)
        )
        self._conn.commit()

    def delete_swing(self, swing_id: str):
        """Delete a swing and its associated clips."""
        # Get clip filepaths to delete files
        clips = self._conn.execute(
            "SELECT filepath FROM clips WHERE swing_id = ?", (swing_id,)
        ).fetchall()
        for clip in clips:
            if clip["filepath"]:
                path = self.clips_dir / clip["filepath"]
                if path.exists():
                    path.unlink()

        self._conn.execute("DELETE FROM clips WHERE swing_id = ?", (swing_id,))
        self._conn.execute("DELETE FROM swings WHERE id = ?", (swing_id,))
        self._conn.commit()

    def get_swing(self, swing_id: str) -> SwingRecord | None:
        row = self._conn.execute(
            "SELECT * FROM swings WHERE id = ?", (swing_id,)
        ).fetchone()
        if not row:
            return None

        clips = self._conn.execute(
            "SELECT camera_name, angle, filepath, trim_start, trim_end FROM clips WHERE swing_id = ?",
            (swing_id,),
        ).fetchall()

        return SwingRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            tag=row["tag"],
            notes=row["notes"],
            clips=[dict(c) for c in clips],
        )

    def list_swings(
        self, limit: int = 50, offset: int = 0, tag_filter: str | None = None
    ) -> list[SwingRecord]:
        if tag_filter:
            rows = self._conn.execute(
                "SELECT * FROM swings WHERE tag = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (tag_filter, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM swings ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        records = []
        for row in rows:
            clips = self._conn.execute(
                "SELECT camera_name, angle, filepath, trim_start, trim_end FROM clips WHERE swing_id = ?",
                (row["id"],),
            ).fetchall()
            records.append(SwingRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                tag=row["tag"],
                notes=row["notes"],
                clips=[dict(c) for c in clips],
            ))
        return records

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM swings").fetchone()[0]
        good = self._conn.execute(
            "SELECT COUNT(*) FROM swings WHERE tag = 'good'"
        ).fetchone()[0]
        bad = self._conn.execute(
            "SELECT COUNT(*) FROM swings WHERE tag = 'bad'"
        ).fetchone()[0]
        untagged = self._conn.execute(
            "SELECT COUNT(*) FROM swings WHERE tag IS NULL"
        ).fetchone()[0]
        return {
            "total": total,
            "good": good,
            "bad": bad,
            "untagged": untagged,
        }

    def save_analysis(
        self,
        swing_id: str,
        camera_name: str,
        tempo_ratio: float | None,
        backswing_frames: int | None,
        downswing_frames: int | None,
        head_stability: float | None,
        hip_rotation: float | None,
        shoulder_rotation: float | None,
        phases_json: str | None,
    ):
        self._conn.execute(
            """INSERT INTO swing_analysis
               (swing_id, camera_name, tempo_ratio, backswing_frames,
                downswing_frames, head_stability, hip_rotation,
                shoulder_rotation, phases_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (swing_id, camera_name, tempo_ratio, backswing_frames,
             downswing_frames, head_stability, hip_rotation,
             shoulder_rotation, phases_json),
        )
        self._conn.commit()

    def get_analysis(self, swing_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM swing_analysis WHERE swing_id = ?", (swing_id,)
        ).fetchall()
        return [dict(r) for r in rows]
