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
            pl.when(pl.col("primaryTitle").is_not_null())
            .then(
                pl.concat_str(
                    [
                        pl.lit("["),
                        pl.col("primaryTitle"),
                        pl.lit("]"),
                        pl.lit("("),
                        pl.lit("https://www.imdb.com/title/"),
                        pl.col("tconst"),
                        pl.lit(")"),
                    ]
                )
            )
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("["),
                        pl.col("title"),
                        pl.lit("]"),
                    ]
                )
            )
            .alias("title")
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
    tbl_cols = []
    for col in df.columns:
        if col == "title":
            tbl_cols.append(
                {"id": "title", "name": "title", "presentation": "markdown"}
            )
        else:
            tbl_cols.append({"id": col, "name": col})

    return [
        html.Div(
            id="summary-table",
            children=dash_table.DataTable(
                data=df.to_dict("records"),
                columns=tbl_cols,
            ),
        )
    ]
