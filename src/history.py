"""Analysis history: compact session records in SQLite, keyed by an anonymous browser id.

A record holds everything the app shows (features, prediction, per-game features, a slim
per-swing table and the heart-rate trace), so past sessions open without the .fit file.
The same JSON record is mirrored into the browser's IndexedDB by app.py as a fallback when
the server database is gone (e.g. a restarted Hugging Face Space).
"""
import json
import math
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("BADAK_DB", os.path.join(ROOT, ".appdata", "history.sqlite"))
SCHEMA = 1


def clean(x):
    """JSON-safe copy: NaN/inf -> None, tuples -> lists (browsers reject NaN in JSON)."""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def compact(session, prediction):
    """Session dict from parser.parse_fit + model prediction -> JSON-able record."""
    sw = session["swings"]
    return clean({
        "schema": SCHEMA,
        "id": session["file_id"],
        "activity_id": session["activity_id"],
        "file_name": session["file_name"],
        "start_local": session["start_local"],
        "model": session["model"],
        "meta": {k: session["meta"].get(k) for k in ("total_timer_time", "total_calories", "avg_heart_rate",
                                                      "max_heart_rate", "total_training_effect",
                                                      "total_anaerobic_training_effect", "user_max_heart_rate",
                                                      "user_resting_heart_rate")},
        "games": session["games"],
        "features": session["features"],
        "per_game": session["per_game"],
        "prediction": prediction,
        # slim swing table: [time_s, game, code, racket_kmh, saturated]
        "swings": [[s["time_s"], s["game"], s["code"], s["racket_kmh"], int(s["saturated"])] for s in sw],
        "hr": [[round(t, 3), h] for t, h in session["hr"][::2]],
    })


class History:
    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            user TEXT NOT NULL, id TEXT NOT NULL, start_local TEXT, added REAL, record TEXT NOT NULL,
            PRIMARY KEY (user, id))""")
        self.db.commit()

    def save(self, user, rec):
        self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
                        (user, rec["id"], rec["start_local"], time.time(), json.dumps(clean(rec), separators=(",", ":"), allow_nan=False)))
        self.db.commit()

    def all(self, user):
        cur = self.db.execute("SELECT record FROM sessions WHERE user = ? ORDER BY start_local", (user,))
        return [json.loads(r[0]) for r in cur]

    def delete(self, user, sid):
        self.db.execute("DELETE FROM sessions WHERE user = ? AND id = ?", (user, sid))
        self.db.commit()

    def clear(self, user):
        self.db.execute("DELETE FROM sessions WHERE user = ?", (user,))
        self.db.commit()
