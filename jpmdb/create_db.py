from pathlib import Path
import duckdb


def create_db():
    db_path = Path(__file__).parents[1] / "data" / "jpmdb.duckdb"
    conn = duckdb.connect(str(db_path))

    cmds = [
        "create view imdb_title_basics as select * from delta_scan('data/silver/imdb/title_basics')",
        "create view imdb_title_ratings as select * from delta_scan('data/silver/imdb/title_ratings')",
    ]

    for cmd in cmds:
        conn.sql(cmd)


if __name__ == "__main__":
    create_db()
