from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import polars as pl
from dash import Input, Output, callback, dash_table, html

from jpmdb.styling import ScreenWidth, get_dt_style


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
            .otherwise(pl.col("title"))
            .alias("title")
        )
        .select(
            [
                pl.col("watched_id").alias("id"),
                "title",
                pl.col("titleType").alias("type"),
                pl.col("startYear").alias("year"),
                "season",
                "genres",
                "rating",
                pl.col("averageRating").alias("imdb rating"),
                (pl.col("rating") - pl.col("averageRating")).alias("rating diff"),
                pl.col("numVotes").alias("imdb votes"),
            ]
        )
    )
    return df.to_pandas()


@callback(
    [
        Output("summary-table", "children"),
        Output("summary-table", "style"),
        Output("summary-fade", "is_in"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_styled_summary_table(dark_mode: bool, breakpoint_name: str):
    df = get_records()
    summary_style = get_dt_style(dark_mode)
    summary_style["style_table"]["height"] = "auto"
    sm_margins = {"maxWidth": "90vw", "width": "90vw"}
    lg_margins = {"maxWidth": "50vw", "width": "50vw", "marginLeft": "20vw"}

    if breakpoint_name in [ScreenWidth.xs, ScreenWidth.sm]:
        summary_style["style_cell"]["font_size"] = "12px"
        summary_style["style_table"].update(sm_margins)

    else:
        summary_style["style_table"].update(lg_margins)

    tbl_cols = []
    markdown_style = {"presentation": "markdown"}
    float_style = {"type": "numeric", "format": {"specifier": ".1f"}}
    int_style = {"type": "numeric", "format": {"specifier": ",d"}}

    col_mapping = {
        "title": markdown_style,
        "rating": float_style,
        "imdb rating": float_style,
        "rating diff": float_style,
        "imdb votes": int_style,
    }

    width_mapping = {
        "id": 75,
        "title": 150,
        "type": 75,
        "year": 75,
        "season": 75,
        "genres": 100,
        "rating": 75,
        "imdb rating": 75,
        "rating diff": 75,
        "imdb votes": 75,
    }

    width_adjustment = [
        {
            "if": {"column_id": i},
            "minWidth": width_mapping[i],
            "maxWidth": width_mapping[i],
        }
        for i in width_mapping
    ]

    summary_style["style_cell_conditional"].extend(width_adjustment)
    for col in df.columns:
        if col in col_mapping:
            tbl_cols.append({**col_mapping[col], "id": col, "name": col})
        else:
            tbl_cols.append({"id": col, "name": col})

    tbl = dash_table.DataTable(
        df.to_dict("records"),
        columns=tbl_cols,
        **summary_style,
    )

    return tbl, {}, True


def layout():
    return [
        dbc.Fade(
            id="summary-fade",
            children=[html.Div(id="summary-table", style={"visibility": "hidden"})],
            style={"transition": "opacity 200ms ease-in", "minHeight": "100vh"},
            is_in=False,
        )
    ]
