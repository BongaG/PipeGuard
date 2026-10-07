import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    node TEXT NOT NULL,
    pressure_kpa REAL NOT NULL,
    twin_kpa REAL,
    source TEXT NOT NULL DEFAULT 'sim'
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    inlet_lps REAL NOT NULL,
    metered_lps REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at REAL NOT NULL,
    closed_at REAL,
    kind TEXT NOT NULL,
    pipe TEXT,
    pipe_confidence REAL,
    event_confidence REAL,
    detected_by TEXT NOT NULL,
    latency_s REAL,
    status TEXT NOT NULL DEFAULT 'open',
    note TEXT
);
CREATE TABLE IF NOT EXISTS valve_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    pipe TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path):
        self.path = path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def executemany(self, sql, rows):
        with self.lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def prune(self, before_ts):
        with self.lock:
            self.conn.execute("DELETE FROM readings WHERE ts < ?", (before_ts,))
            self.conn.execute("DELETE FROM flows WHERE ts < ?", (before_ts,))
            self.conn.commit()
