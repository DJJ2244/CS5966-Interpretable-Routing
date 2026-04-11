import sqlite3
from dataclasses import dataclass
from typing import Optional

from util.database_connection_util import get_connection


TABLE = "runs"
F_ID = TABLE + ".id"
F_WEAK_MODEL_ID = TABLE + ".weak_model_id"
F_STRONG_MODEL_ID = TABLE + ".strong_model_id"
F_SPLIT_ID = TABLE + ".split_id"
F_THRESHOLD = TABLE + ".route_llm_threshold"
F_CREATED_AT = TABLE + ".created_at"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class Run:
    id: int
    weak_model_id: int
    strong_model_id: int
    split_id: int
    route_llm_threshold: Optional[float]
    created_at: str


def _map(row: sqlite3.Row) -> Run:
    return Run(
        id=row[_col(F_ID)],
        weak_model_id=row[_col(F_WEAK_MODEL_ID)],
        strong_model_id=row[_col(F_STRONG_MODEL_ID)],
        split_id=row[_col(F_SPLIT_ID)],
        route_llm_threshold=row[_col(F_THRESHOLD)],
        created_at=row[_col(F_CREATED_AT)],
    )


def _get_by_id(conn: sqlite3.Connection, run_id: int) -> Optional[Run]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (run_id,)
    ).fetchone()
    return _map(row) if row else None


def create(
    weak_model_id: int,
    strong_model_id: int,
    split_id: int,
    route_llm_threshold: Optional[float] = None,
) -> Run:
    conn = get_connection()
    try:
        cursor = conn.execute(
            f"""
            INSERT INTO {TABLE}
                ({_col(F_WEAK_MODEL_ID)}, {_col(F_STRONG_MODEL_ID)}, {_col(F_SPLIT_ID)}, {_col(F_THRESHOLD)})
            VALUES (?, ?, ?, ?)
            """,
            (weak_model_id, strong_model_id, split_id, route_llm_threshold),
        )
        conn.commit()
        return _get_by_id(conn, cursor.lastrowid)
    finally:
        conn.close()


def get_by_id(run_id: int) -> Optional[Run]:
    conn = get_connection()
    try:
        return _get_by_id(conn, run_id)
    finally:
        conn.close()


def get_latest() -> Optional[Run]:
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY {F_ID} DESC LIMIT 1"
        ).fetchone()
        return _map(row) if row else None
    finally:
        conn.close()
