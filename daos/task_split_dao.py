import sqlite3
from dataclasses import dataclass


TABLE = "task_split"
F_SPLIT_ID = TABLE + ".split_id"
F_TASK_ID = TABLE + ".task_id"
F_IS_TEST = TABLE + ".is_test"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class TaskSplit:
    split_id: int
    task_id: str
    is_test: bool


def _map(row: sqlite3.Row) -> TaskSplit:
    return TaskSplit(
        split_id=row[_col(F_SPLIT_ID)],
        task_id=row[_col(F_TASK_ID)],
        is_test=bool(row[_col(F_IS_TEST)]),
    )


def bulk_insert(conn: sqlite3.Connection, rows: list[TaskSplit]) -> None:
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO {TABLE}
            ({_col(F_SPLIT_ID)}, {_col(F_TASK_ID)}, {_col(F_IS_TEST)})
        VALUES (?, ?, ?)
        """,
        [(r.split_id, r.task_id, 1 if r.is_test else 0) for r in rows],
    )


def get_task_ids_for_split(
    conn: sqlite3.Connection, split_id: int, is_test: bool
) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT {F_TASK_ID} FROM {TABLE}
        WHERE {F_SPLIT_ID} = ? AND {F_IS_TEST} = ?
        """,
        (split_id, 1 if is_test else 0),
    ).fetchall()
    return [r[_col(F_TASK_ID)] for r in rows]
