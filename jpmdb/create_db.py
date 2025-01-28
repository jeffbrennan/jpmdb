from pathlib import Path
import duckdb


def create_db():
    db_path = Path(__file__).parents[1] / "data" / "jpmdb.duckdb"
    conn = duckdb.connect(str(db_path))

    base_dir = Path(__file__).parents[1] / "data" / "silver"
    tables = [
        "imdb/title_basics",
        "imdb/title_ratings",
        "jpmdb",
    ]

    for tbl in tables:
        tbl_name = tbl.split("/")[-1]
        cmd = f"create view if not exists {tbl_name} as select * from delta_scan('{base_dir.as_posix()}/{tbl}')"
        conn.sql(cmd)


if __name__ == "__main__":
    create_db()
