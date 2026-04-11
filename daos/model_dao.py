import sqlite3
from dataclasses import dataclass
from typing import Optional

from util.database_connection_util import get_connection


TABLE = "models"
F_ID = TABLE + ".id"
F_NAME = TABLE + ".name"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class Model:
    id: int
    name: str


def _map(row: sqlite3.Row) -> Model:
    return Model(id=row[_col(F_ID)], name=row[_col(F_NAME)])


def _get_by_name(conn: sqlite3.Connection, name: str) -> Optional[Model]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_NAME} = ?", (name,)
    ).fetchone()
    return _map(row) if row else None


def _get_by_id(conn: sqlite3.Connection, model_id: int) -> Optional[Model]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (model_id,)
    ).fetchone()
    return _map(row) if row else None


def get_by_name(name: str) -> Optional[Model]:
    conn = get_connection()
    try:
        return _get_by_name(conn, name)
    finally:
        conn.close()


def get_by_id(model_id: int) -> Optional[Model]:
    conn = get_connection()
    try:
        return _get_by_id(conn, model_id)
    finally:
        conn.close()


def get_or_create(name: str) -> Model:
    conn = get_connection()
    try:
        existing = _get_by_name(conn, name)
        if existing:
            return existing
        conn.execute(f"INSERT INTO {TABLE} ({_col(F_NAME)}) VALUES (?)", (name,))
        conn.commit()
        return _get_by_name(conn, name)
    finally:
        conn.close()
