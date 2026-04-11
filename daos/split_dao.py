import sqlite3
from dataclasses import dataclass
from typing import Optional

from util.database_connection_util import get_connection


TABLE = "split"
F_ID = TABLE + ".id"
F_NAME = TABLE + ".name"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class Split:
    id: int
    name: str


def _map(row: sqlite3.Row) -> Split:
    return Split(id=row[_col(F_ID)], name=row[_col(F_NAME)])


def _get_by_name(conn: sqlite3.Connection, name: str) -> Optional[Split]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_NAME} = ?", (name,)
    ).fetchone()
    return _map(row) if row else None


def get_by_name(name: str) -> Optional[Split]:
    conn = get_connection()
    try:
        return _get_by_name(conn, name)
    finally:
        conn.close()


def get_by_id(split_id: int) -> Optional[Split]:
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (split_id,)
        ).fetchone()
        return _map(row) if row else None
    finally:
        conn.close()


def create(name: str) -> Split:
    conn = get_connection()
    try:
        conn.execute(f"INSERT OR IGNORE INTO {TABLE} ({_col(F_NAME)}) VALUES (?)", (name,))
        conn.commit()
        return _get_by_name(conn, name)
    finally:
        conn.close()
