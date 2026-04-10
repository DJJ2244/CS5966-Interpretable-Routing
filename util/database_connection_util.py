import os
import sqlite3
from pathlib import Path


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the routing database.

    The database path is read from the DB_PATH environment variable,
    defaulting to data/routing.db relative to the project root.
    """
    db_path = Path(os.environ.get("DB_PATH", "data/routing.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
