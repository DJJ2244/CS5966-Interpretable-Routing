import sqlite3
from dataclasses import dataclass
from typing import Optional


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


def get_by_name(conn: sqlite3.Connection, name: str) -> Optional[Split]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_NAME} = ?", (name,)
    ).fetchone()
    return _map(row) if row else None


def get_by_id(conn: sqlite3.Connection, split_id: int) -> Optional[Split]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (split_id,)
    ).fetchone()
    return _map(row) if row else None


def create(conn: sqlite3.Connection, name: str) -> Split:
    conn.execute(f"INSERT OR IGNORE INTO {TABLE} ({_col(F_NAME)}) VALUES (?)", (name,))
    conn.commit()
    return get_by_name(conn, name)
