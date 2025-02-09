from pathlib import Path

import pandas as pd
import polars as pl
from dash import dash_table, html


def get_records() -> pd.DataFrame:
    path = Path(__file__).parents[2] / "data" / "gold" / "jpmdb"
    df = (
        pl.read_delta(str(path))
        .with_columns(pl.col("genres").list.join(", ").alias("genres"))
        .with_columns(
            pl.coalesce(pl.col("primaryTitle"), pl.col("title")).alias("title")
        )
        .select(
            [
                "watched_id",
                "title",
                pl.col("startYear").alias("year"),
                "genres",
                "rating",
                pl.col("averageRating").round(1).cast(pl.String).alias("imdb_rating"),
                pl.col("numVotes").alias("imdb_votes"),
                (
                    (pl.col("rating") - pl.col("averageRating"))
                    .round(1)
                    .cast(pl.String)
                ).alias("rating_diff"),
            ]
        )
    )
    return df.to_pandas()


def layout():
    df = get_records()
    print(df.head())
    print(df.columns)
    return [
        html.Div(
            id="summary-table",
            children=dash_table.DataTable(data=df.to_dict("records")),
        )
    ]
