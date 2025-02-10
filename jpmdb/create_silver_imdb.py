from pathlib import Path

import pandas as pd
import polars as pl
import requests


def get_imdb_datasets():
    base_url = "https://datasets.imdbws.com/"
    out_dir = Path(__file__).parents[1] / "data" / "bronze" / "imdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["title.basics.tsv.gz", "title.ratings.tsv.gz"]
    for dataset in datasets:
        url = base_url + dataset
        response = requests.get(url)
        out_path = out_dir / dataset
        with out_path.open("wb") as f:
            f.write(response.content)


def parse_imdb_title_basics() -> None:
    in_file_path = (
        Path(__file__).parents[1] / "data" / "bronze" / "imdb" / "title.basics.tsv.gz"
    )
    out_file_path = (
        Path(__file__).parents[1] / "data" / "silver" / "imdb" / "title_basics"
    )

    df = pl.DataFrame(
        pd.read_csv(
            in_file_path,
            sep="\t",
            na_values=["\\N"],
            encoding="latin1",
            low_memory=False,
        )
    )

    df_clean = (
        df.with_columns(
            pl.when(pl.col("runtimeMinutes").str.contains(r"[^0-9]"))
            .then(pl.col("runtimeMinutes"))
            .alias("non_numeric_runtime")
        )
        .with_columns(
            pl.when(pl.col("non_numeric_runtime").is_not_null())
            .then(pl.col("non_numeric_runtime"))
            .otherwise(pl.col("genres"))
            .alias("genres")
        )
        .with_columns(
            [
                pl.col("runtimeMinutes").str.extract(r"(\d+)").cast(pl.UInt32),
                pl.col("isAdult").cast(pl.Boolean),
                pl.col("startYear").cast(pl.UInt16),
                pl.col("endYear").cast(pl.UInt16),
                pl.col("genres").str.split(","),
            ]
        )
    ).drop("non_numeric_runtime")
    print(df_clean.head())

    df_clean.write_delta(out_file_path, mode="overwrite")


def parse_imdb_title_ratings() -> None:
    in_file_path = (
        Path(__file__).parents[1] / "data" / "bronze" / "imdb" / "title.ratings.tsv.gz"
    )

    out_file_path = (
        Path(__file__).parents[1] / "data" / "silver" / "imdb" / "title_ratings"
    )

    df = pl.read_csv(
        in_file_path, separator="\t", null_values=["\\N"], low_memory=False
    )

    df_clean = df.with_columns(
        pl.col("averageRating").cast(pl.Float32), pl.col("numVotes").cast(pl.UInt32)
    )
    print(df_clean.head())
    df_clean.write_delta(out_file_path, mode="overwrite")


def parse_imdb_datasets():
    parse_imdb_title_basics()
    parse_imdb_title_ratings()


if __name__ == "__main__":
    get_imdb_datasets()
    parse_imdb_datasets()
