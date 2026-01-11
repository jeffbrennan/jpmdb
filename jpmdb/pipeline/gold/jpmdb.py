from pathlib import Path

import polars as pl


def main():
    base_path = Path(__file__).parents[1] / "data"
    jpmdb_silver = pl.read_delta(
        str(base_path / "silver" / "jpmdb" / "stg_jpmdb_combined")
    )
    imdb_data = (
        pl.read_delta(str(base_path / "silver" / "imdb" / "title_basics"))
        .join(
            pl.read_delta(str(base_path / "silver" / "imdb" / "title_ratings")),
            on="tconst",
        )
        .select(
            "tconst",
            "primaryTitle",
            "originalTitle",
            "titleType",
            "startYear",
            "runtimeMinutes",
            "genres",
            "averageRating",
            "numVotes",
        )
    )

    combined = (
        jpmdb_silver.join(imdb_data, on="tconst", how="left")
        .drop("manually_approved", "manually_reviewed_at")
        .sort("watched_id")
    )

    combined.write_delta(str(base_path / "gold" / "jpmdb"), mode="overwrite")


if __name__ == "__main__":
    main()
