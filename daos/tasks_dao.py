import sqlite3
from dataclasses import dataclass
from typing import Optional

import daos.task_split_dao as task_split_dao


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
    )


def get_by_id(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute(
        f"SELECT * FROM {TABLE} WHERE {F_ID} = ?", (task_id,)
    ).fetchone()
    return _map(row) if row else None


def get_all(conn: sqlite3.Connection) -> list[Task]:
    rows = conn.execute(f"SELECT * FROM {TABLE}").fetchall()
    return [_map(r) for r in rows]


def get_all_for_split(
    conn: sqlite3.Connection, split_id: int, is_test: bool
) -> list[Task]:
    rows = conn.execute(
        f"""
        SELECT {TABLE}.* FROM {TABLE}
        JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_ID}
        WHERE {task_split_dao.F_SPLIT_ID} = ? AND {task_split_dao.F_IS_TEST} = ?
        """,
        (split_id, 1 if is_test else 0),
    ).fetchall()
    return [_map(r) for r in rows]
