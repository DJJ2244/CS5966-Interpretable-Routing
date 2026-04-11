"""
database_util.py - Database initialization and seeding.

Call init_db() once to create all tables and populate them with the
HumanEval-XL dataset and a default 80/20 stratified split.
Safe to call multiple times — uses INSERT OR IGNORE throughout.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from util.database_connection_util import get_connection

DATA_PATH = Path("data/humaneval_xl_english.jsonl")
SEED = 42
TEST_SIZE = 0.2


def init_db() -> None:
    conn = get_connection()
    _create_tables(conn)
    _seed_tasks(conn)
    split_id = _ensure_default_split(conn)
    _seed_task_split(conn, split_id)
    conn.commit()
    conn.close()
    print("Database initialized.")


def _create_tables(conn: object) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id                  TEXT PRIMARY KEY,
            prompt              TEXT NOT NULL,
            entry_point         TEXT NOT NULL,
            test                TEXT NOT NULL,
            description         TEXT,
            language            TEXT,
            canonical_solution  TEXT,
            natural_language    TEXT,
            programming_language TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS split (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_split (
            split_id    INTEGER NOT NULL REFERENCES split(id),
            task_id     TEXT    NOT NULL REFERENCES tasks(id),
            is_test     INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (split_id, task_id)
        );

        CREATE TABLE IF NOT EXISTS model_task_result (
            task_id         TEXT    NOT NULL REFERENCES tasks(id),
            model_name      TEXT    NOT NULL,
            result          TEXT,
            run_millis      INTEGER,
            extracted_code  TEXT,
            passed          INTEGER,
            PRIMARY KEY (task_id, model_name)
        );

        CREATE TABLE IF NOT EXISTS runs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            weak_model_name     TEXT    NOT NULL,
            strong_model_name   TEXT    NOT NULL,
            split_id            INTEGER NOT NULL REFERENCES split(id),
            route_llm_threshold REAL,
            created_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)


def _seed_tasks(conn: object) -> None:
    if not DATA_PATH.exists():
        print(f"Warning: {DATA_PATH} not found — skipping task seeding.")
        return

    inserted = 0
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks
                    (id, prompt, entry_point, test, description, language,
                     canonical_solution, natural_language, programming_language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["task_id"],
                    rec.get("prompt", ""),
                    rec.get("entry_point", ""),
                    rec.get("test", ""),
                    rec.get("description", ""),
                    rec.get("language", ""),
                    rec.get("canonical_solution", ""),
                    rec.get("natural_language", ""),
                    rec.get("programming_language", "python"),
                ),
            )
            inserted += 1

    print(f"Seeded {inserted} tasks (existing rows skipped).")


def _ensure_default_split(conn: object) -> int:
    conn.execute("INSERT OR IGNORE INTO split (name) VALUES ('default')")
    row = conn.execute("SELECT id FROM split WHERE name = 'default'").fetchone()
    return row["id"]


def _seed_task_split(conn, split_id: int) -> None:
    existing = conn.execute(
        "SELECT COUNT(*) as cnt FROM task_split WHERE split_id = ?", (split_id,)
    ).fetchone()["cnt"]
    if existing > 0:
        print(f"Split {split_id} already has {existing} rows — skipping split seeding.")
        return

    # Load all tasks grouped by programming_language
    buckets: dict = defaultdict(list)
    rows = conn.execute("SELECT id, programming_language FROM tasks").fetchall()
    for row in rows:
        buckets[row["programming_language"]].append(row["id"])

    rng = random.Random(SEED)
    train_ids, test_ids = [], []

    for lang in sorted(buckets):
        ids = buckets[lang][:]
        rng.shuffle(ids)
        split_idx = int(len(ids) * (1 - TEST_SIZE))
        train_ids.extend(ids[:split_idx])
        test_ids.extend(ids[split_idx:])

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    rows_to_insert = (
        [(split_id, task_id, 0) for task_id in train_ids]
        + [(split_id, task_id, 1) for task_id in test_ids]
    )
    conn.executemany(
        "INSERT OR IGNORE INTO task_split (split_id, task_id, is_test) VALUES (?, ?, ?)",
        rows_to_insert,
    )
    print(f"Created split '{split_id}': {len(train_ids)} train, {len(test_ids)} test.")
