import sqlite3
from dataclasses import dataclass
from typing import Optional

import daos.task_split_dao as task_split_dao
from util.database_connection_util import get_connection


TABLE = "model_task_result"
F_TASK_ID = TABLE + ".task_id"
F_MODEL_NAME = TABLE + ".model_name"
F_RESULT = TABLE + ".result"
F_RUN_MILLIS = TABLE + ".run_millis"
F_EXTRACTED_CODE = TABLE + ".extracted_code"
F_PASSED = TABLE + ".passed"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class ModelTaskResult:
    task_id: str
    model_name: str
    result: Optional[str]
    run_millis: Optional[int]
    extracted_code: Optional[str]
    passed: Optional[bool]


def _map(row: sqlite3.Row) -> ModelTaskResult:
    passed_val = row[_col(F_PASSED)]
    return ModelTaskResult(
        task_id=row[_col(F_TASK_ID)],
        model_name=row[_col(F_MODEL_NAME)],
        result=row[_col(F_RESULT)],
        run_millis=row[_col(F_RUN_MILLIS)],
        extracted_code=row[_col(F_EXTRACTED_CODE)],
        passed=bool(passed_val) if passed_val is not None else None,
    )


def _upsert(conn: sqlite3.Connection, r: ModelTaskResult) -> None:
    conn.execute(
        f"""
        INSERT INTO {TABLE}
            ({_col(F_TASK_ID)}, {_col(F_MODEL_NAME)}, {_col(F_RESULT)}, {_col(F_RUN_MILLIS)}, {_col(F_EXTRACTED_CODE)}, {_col(F_PASSED)})
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT({_col(F_TASK_ID)}, {_col(F_MODEL_NAME)}) DO UPDATE SET
            {_col(F_RESULT)}         = excluded.{_col(F_RESULT)},
            {_col(F_RUN_MILLIS)}     = excluded.{_col(F_RUN_MILLIS)},
            {_col(F_EXTRACTED_CODE)} = excluded.{_col(F_EXTRACTED_CODE)},
            {_col(F_PASSED)}         = excluded.{_col(F_PASSED)}
        """,
        (
            r.task_id,
            r.model_name,
            r.result,
            r.run_millis,
            r.extracted_code,
            1 if r.passed else (0 if r.passed is not None else None),
        ),
    )


def upsert(r: ModelTaskResult) -> None:
    conn = get_connection()
    try:
        _upsert(conn, r)
        conn.commit()
    finally:
        conn.close()


def bulk_upsert(results: list[ModelTaskResult]) -> None:
    conn = get_connection()
    try:
        for r in results:
            _upsert(conn, r)
        conn.commit()
    finally:
        conn.close()


@dataclass
class ModelPassRate:
    model_name: str
    total: int
    passed: int


def get_pass_rates_for_split(split_id: int, is_test: bool) -> list[ModelPassRate]:
    """Return per-model pass counts for all tasks in a split partition."""
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT {F_MODEL_NAME}, COUNT(*) as total, SUM({F_PASSED}) as passed
            FROM {TABLE}
            JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_TASK_ID}
            WHERE {task_split_dao.F_SPLIT_ID} = ? AND {task_split_dao.F_IS_TEST} = ?
            GROUP BY {F_MODEL_NAME}
            """,
            (split_id, 1 if is_test else 0),
        ).fetchall()
        return [
            ModelPassRate(
                model_name=row[_col(F_MODEL_NAME)],
                total=row["total"] or 0,
                passed=int(row["passed"] or 0),
            )
            for row in rows
        ]
    finally:
        conn.close()


def get_all_for_model_split(
    model_name: str,
    split_id: int,
    is_test: bool,
) -> list[ModelTaskResult]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT {TABLE}.* FROM {TABLE}
            JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_TASK_ID}
            WHERE {F_MODEL_NAME} = ?
              AND {task_split_dao.F_SPLIT_ID} = ?
              AND {task_split_dao.F_IS_TEST} = ?
            """,
            (model_name, split_id, 1 if is_test else 0),
        ).fetchall()
        return [_map(r) for r in rows]
    finally:
        conn.close()
