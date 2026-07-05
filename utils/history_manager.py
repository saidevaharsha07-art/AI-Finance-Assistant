"""SQLite history helpers with comparison support."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("finance_history.db")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS finance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )


def save_record(inputs: dict, result: dict) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO finance_records (created_at, input_json, result_json) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), json.dumps(inputs), json.dumps(result)),
        )
        return int(cursor.lastrowid)


def get_record(record_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM finance_records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["input"] = json.loads(item.pop("input_json"))
    item["result"] = json.loads(item.pop("result_json"))
    return item


def recent_records(limit: int = 8) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM finance_records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    records = []
    previous = None
    for row in rows:
        item = dict(row)
        item["input"] = json.loads(item.pop("input_json"))
        item["result"] = json.loads(item.pop("result_json"))
        if previous:
            item["comparison"] = compare_records(item, previous)
        else:
            older = _older_than(int(item["id"]))
            item["comparison"] = compare_records(item, older) if older else None
        previous = item
        records.append(item)
    return records


def _older_than(record_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM finance_records WHERE id < ? ORDER BY id DESC LIMIT 1",
            (record_id,),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["input"] = json.loads(item.pop("input_json"))
    item["result"] = json.loads(item.pop("result_json"))
    return item


def compare_records(current: dict, previous: dict) -> dict:
    cur = current["result"]
    prev = previous["result"]
    return {
        "score_change": round(float(cur.get("health_score", 0)) - float(prev.get("health_score", 0)), 2),
        "savings_change": round(float(cur.get("actual_savings", 0)) - float(prev.get("actual_savings", 0)), 2),
        "class_change": f"{prev.get('spender_class', '-') } -> {cur.get('spender_class', '-')}",
    }
