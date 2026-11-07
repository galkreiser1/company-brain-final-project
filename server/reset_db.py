"""
Put runs/novaops.db back to the seed data, then print each table's row count.

Stop the server first. Run from the project root:
  python -m server.reset_db
"""

import sqlite3

from server import db


def main() -> None:
    db.reset_database()
    conn = sqlite3.connect(db.DB_PATH)
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )]
    print(f"Reset {db.DB_PATH}")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<24} {count} rows")
    conn.close()


if __name__ == "__main__":
    main()
