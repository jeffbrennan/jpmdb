from pathlib import Path

from deltalake import DeltaTable


def main():
    base_path = Path(__file__).parents[2] / "data"
    all_tables = [
        base_path / "silver" / "imdb" / "title_basics",
        base_path / "silver" / "imdb" / "title_ratings",
        base_path / "silver" / "jpmdb" / "jpmdb_cleaned",
        base_path / "silver" / "jpmdb" / "stg_jpmdb_combined",
        base_path / "gold" / "jpmdb",
    ]

    for tbl in all_tables:
        print("-" * 80)
        print(tbl)
        dt = DeltaTable(tbl.as_posix())
        print(dt.optimize.compact(target_size=1024 * 1024))
        print(
            dt.vacuum(
                retention_hours=0, enforce_retention_duration=False, dry_run=False
            )
        )
        print(dt.history()[-1])


if __name__ == "__main__":
    main()
