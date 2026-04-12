import sqlite3
from dataclasses import dataclass
from typing import Optional

import daos.task_split_dao as task_split_dao
from util.database_connection_util import get_connection


TABLE = "tasks"
F_ID = TABLE + ".id"
F_PROMPT = TABLE + ".prompt"
F_ENTRY_POINT = TABLE + ".entry_point"
F_TEST = TABLE + ".test"
F_DESCRIPTION = TABLE + ".description"
F_LANGUAGE = TABLE + ".language"
F_CANONICAL_SOLUTION = TABLE + ".canonical_solution"
F_NATURAL_LANGUAGE = TABLE + ".natural_language"
F_PROGRAMMING_LANGUAGE = TABLE + ".programming_language"
F_TOUGHNESS_SCORE = TABLE + ".toughness_score"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class Task:
    id: str
    prompt: str
    entry_point: str
    test: str
    description: str
    language: str
    canonical_solution: str
    natural_language: str
    programming_language: str
    toughness_score: Optional[float] = None


def _map(row: sqlite3.Row) -> Task:
    return Task(
        id=row[_col(F_ID)],
        prompt=row[_col(F_PROMPT)],
        entry_point=row[_col(F_ENTRY_POINT)],
        test=row[_col(F_TEST)],
        description=row[_col(F_DESCRIPTION)] or "",
        language=row[_col(F_LANGUAGE)] or "",
        canonical_solution=row[_col(F_CANONICAL_SOLUTION)] or "",
        natural_language=row[_col(F_NATURAL_LANGUAGE)] or "",
        programming_language=row[_col(F_PROGRAMMING_LANGUAGE)],
        toughness_score=row[_col(F_TOUGHNESS_SCORE)],
    )


def get_by_id(task_id: str) -> Optional[Task]:
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (task_id,)
        ).fetchone()
        return _map(row) if row else None
    finally:
        conn.close()


def get_all() -> list[Task]:
    conn = get_connection()
    try:
        rows = conn.execute(f"SELECT * FROM {TABLE}").fetchall()
        return [_map(r) for r in rows]
    finally:
        conn.close()


def get_all_for_split(split_id: int, is_test: bool) -> list[Task]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT {TABLE}.* FROM {TABLE}
            JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_ID}
            WHERE {task_split_dao.F_SPLIT_ID} = ? AND {task_split_dao.F_IS_TEST} = ?
            """,
            (split_id, 1 if is_test else 0),
        ).fetchall()
        return [_map(r) for r in rows]
    finally:
        conn.close()


def set_toughness_score(task_id: str, score: float) -> None:
    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE {TABLE} SET {_col(F_TOUGHNESS_SCORE)} = ? WHERE {_col(F_ID)} = ?",
            (score, task_id),
        )
        conn.commit()
    finally:
        conn.close()
