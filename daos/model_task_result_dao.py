import sqlite3
from dataclasses import dataclass
from typing import Optional

import daos.model_dao as model_dao
import daos.task_split_dao as task_split_dao


TABLE = "model_task_result"
F_TASK_ID = TABLE + ".task_id"
F_MODEL_ID = TABLE + ".model_id"
F_RESULT = TABLE + ".result"
F_RUN_MILLIS = TABLE + ".run_millis"
F_EXTRACTED_CODE = TABLE + ".extracted_code"
F_PASSED = TABLE + ".passed"


def _col(f: str) -> str:
    return f.rsplit(".", 1)[-1]


@dataclass
class ModelTaskResult:
    task_id: str
    model_id: int
    result: Optional[str]
    run_millis: Optional[int]
    extracted_code: Optional[str]
    passed: Optional[bool]


def _map(row: sqlite3.Row) -> ModelTaskResult:
    passed_val = row[_col(F_PASSED)]
    return ModelTaskResult(
        task_id=row[_col(F_TASK_ID)],
        model_id=row[_col(F_MODEL_ID)],
        result=row[_col(F_RESULT)],
        run_millis=row[_col(F_RUN_MILLIS)],
        extracted_code=row[_col(F_EXTRACTED_CODE)],
        passed=bool(passed_val) if passed_val is not None else None,
    )


def upsert(conn: sqlite3.Connection, r: ModelTaskResult) -> None:
    conn.execute(
        f"""
        INSERT INTO {TABLE}
            ({_col(F_TASK_ID)}, {_col(F_MODEL_ID)}, {_col(F_RESULT)}, {_col(F_RUN_MILLIS)}, {_col(F_EXTRACTED_CODE)}, {_col(F_PASSED)})
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT({_col(F_TASK_ID)}, {_col(F_MODEL_ID)}) DO UPDATE SET
            {_col(F_RESULT)}         = excluded.{_col(F_RESULT)},
            {_col(F_RUN_MILLIS)}     = excluded.{_col(F_RUN_MILLIS)},
            {_col(F_EXTRACTED_CODE)} = excluded.{_col(F_EXTRACTED_CODE)},
            {_col(F_PASSED)}         = excluded.{_col(F_PASSED)}
        """,
        (
            r.task_id,
            r.model_id,
            r.result,
            r.run_millis,
            r.extracted_code,
            1 if r.passed else (0 if r.passed is not None else None),
        ),
    )


def bulk_upsert(conn: sqlite3.Connection, results: list[ModelTaskResult]) -> None:
    for r in results:
        upsert(conn, r)


@dataclass
class ModelPassRate:
    model_name: str
    total: int
    passed: int


def get_pass_rates_for_split(
    conn: sqlite3.Connection,
    split_id: int,
    is_test: bool,
) -> list[ModelPassRate]:
    """Return per-model pass counts for all tasks in a split partition."""
    rows = conn.execute(
        f"""
        SELECT {model_dao.F_NAME}, COUNT(*) as total, SUM({F_PASSED}) as passed
        FROM {TABLE}
        JOIN {model_dao.TABLE} ON {model_dao.F_ID} = {F_MODEL_ID}
        JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_TASK_ID}
        WHERE {task_split_dao.F_SPLIT_ID} = ? AND {task_split_dao.F_IS_TEST} = ?
        GROUP BY {model_dao.F_ID}
        """,
        (split_id, 1 if is_test else 0),
    ).fetchall()
    return [
        ModelPassRate(
            model_name=row[_col(model_dao.F_NAME)],
            total=row["total"] or 0,
            passed=int(row["passed"] or 0),
        )
        for row in rows
    ]


def get_all_for_model_split(
    conn: sqlite3.Connection,
    model_id: int,
    split_id: int,
    is_test: bool,
) -> list[ModelTaskResult]:
    rows = conn.execute(
        f"""
        SELECT {TABLE}.* FROM {TABLE}
        JOIN {task_split_dao.TABLE} ON {task_split_dao.F_TASK_ID} = {F_TASK_ID}
        WHERE {F_MODEL_ID} = ?
          AND {task_split_dao.F_SPLIT_ID} = ?
          AND {task_split_dao.F_IS_TEST} = ?
        """,
        (model_id, split_id, 1 if is_test else 0),
    ).fetchall()
    return [_map(r) for r in rows]
