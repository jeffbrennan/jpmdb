from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pypalettes
from dash import Input, Output, callback, dash_table, dcc, html
from plotly.colors import qualitative
from plotly.graph_objs import Figure
from statsmodels.nonparametric.smoothers_lowess import lowess

from jpmdb.common import timeit
from jpmdb.styling import ScreenWidth, get_dt_style, get_site_colors


def get_fig_margins(fig_type: str):
    sm_width = 90
    lg_width = 50
    margin_left = 20

    if fig_type == "timeseries":
        lg_width = 70
        margin_left = 15

    sm_margins = {"maxWidth": f"{sm_width}vw", "width": f"{sm_width}vw"}
    lg_margins = {
        "maxWidth": f"{lg_width}vw",
        "width": f"{lg_width}vw",
        "marginLeft": f"{margin_left}vw",
    }
    return sm_margins, lg_margins


@timeit
def get_records() -> pd.DataFrame:
    path = Path(__file__).parents[2] / "data" / "gold" / "jpmdb"
    df = (
        pl.read_delta(str(path))
        .with_columns(
            [
                pl.col("genres").list.join(", ").alias("genres"),
                pl.col("title").alias("jp_title"),
            ]
        )
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
            .otherwise(pl.col("jp_title"))
            .alias("title"),
        )
        .select(
            [
                pl.col("watched_id").alias("id"),
                "watched_year",
                "jp_title",
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


@timeit
def style_timeseries_fig(
    fig: Figure, watched_id_list: list[str], font_color: str, screen_width: str
) -> Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(categoryorder="array", categoryarray=watched_id_list)
    fig.update_yaxes(showline=False, showgrid=False, zeroline=False)

    trace_color = font_color.replace("rgb", "rgba").replace(")", ", 0.5)")
    fig.update_traces(marker=dict(line=dict(width=1, color=trace_color)))

    year_groups = {}
    for uid in watched_id_list:
        year = str(uid).split("_")[0]
        year_groups.setdefault(year, []).append(uid)
    tickvals = []
    ticktext = []
    for year in sorted(year_groups, key=lambda y: int(y)):
        group_ids = year_groups[year]
        mid_index = len(group_ids) // 2
        tickvals.append(group_ids[mid_index])
        ticktext.append(year)

    fig.update_xaxes(
        showline=False,
        showgrid=False,
        zeroline=False,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title="",
    )
    fig.update_layout(
        xaxis_range=[
            -5,
            len(watched_id_list) + 5,
        ],
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=font_color, width=2),
            )
        ],
        legend_title_text="",
    )

    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=200, r=200, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig


@timeit
def add_timeseries_trendline(
    df: pd.DataFrame, fig: Figure, watched_id_list: list[str], font_color: str
):
    years = sorted(set(i.split("_")[0] for i in watched_id_list))
    for year in years[1:]:
        boundary_id = watched_id_list.index(year + "_001")
        fig.add_vline(x=boundary_id, line_color=font_color, layer="below")
    smoothed = lowess(df["rating"], range(len(watched_id_list)), frac=0.025)
    x_trend = [watched_id_list[i] for i, _ in enumerate(smoothed[:, 0])]
    y_trend = smoothed[:, 1]
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(shape="spline", color=font_color, width=6),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(shape="spline", color="rgb(247, 111, 83)", width=3),
            showlegend=False,
        )
    )

    return fig


def get_color_palette(groups: list, n_colors: int = 10):
    n_groups = len(groups)
    repeats = (n_groups // n_colors) + 1
    colors = list(
        pypalettes.load_cmap("Tableau_10", cmap_type="discrete", repeat=repeats).rgb  # type: ignore
    )[0:n_groups]
    return [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in colors]


@callback(
    [
        Output("timeseries-viz", "figure"),
        Output("timeseries-viz", "style"),
        Output("summary-fade", "is_in"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
@timeit
def get_timeseries_viz(dark_mode: bool, screen_width: str):
    df = get_records()
    df["watched_year"] = df["watched_year"].astype(str)
    df["jp_title_hover"] = df["id"] + " - " + df["jp_title"]

    _, font_color = get_site_colors(dark_mode, contrast=False)

    watched_id_list = df["id"].to_list()
    colors = qualitative.Plotly

    types = list(df["type"].unique())
    colors = get_color_palette(types)
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(types)}

    template = "plotly_dark" if dark_mode else "plotly_white"
    fig = px.scatter(
        df,
        x="id",
        y="rating",
        color="type",
        color_discrete_map=color_map,
        hover_name="jp_title_hover",
        hover_data={
            "rating": ":.1f",
            "imdb rating": ":.1f",
            "rating diff": ":.1f",
            "imdb votes": ":,",
            "id": False,
        },
        template=template,
        opacity=0.8,
    )
    fig = style_timeseries_fig(fig, watched_id_list, font_color, screen_width)
    fig = add_timeseries_trendline(df, fig, watched_id_list, font_color)

    return fig, {}, True


@callback(
    [
        Output("rating-diff-viz", "figure"),
        Output("rating-diff-viz", "style"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_rating_diff_viz(dark_mode: bool, screen_width: str):
    df = get_records()[
        [
            "id",
            "jp_title",
            "title",
            "rating",
            "imdb rating",
            "rating diff",
            "imdb votes",
        ]
    ]

    df = df.dropna(subset=["rating", "imdb rating"])

    df["rated_higher"] = df["rating diff"] > 0
    df["rated_higher"] = np.where(
        df["rating diff"] > 0, "higher than imdb", "lower than imdb"
    )
    df["hover_title"] = df["id"] + " - " + df["jp_title"]
    jitter = 0.05

    df["rating_jitter"] = df["rating"] + jitter * np.random.randn(len(df))
    df["imdb rating_jitter"] = df["imdb rating"] + jitter * np.random.randn(len(df))

    min_marker = 5
    max_marker = 80
    min_votes = df["imdb votes"].min()
    max_votes = df["imdb votes"].max()
    df["scaled_votes"] = min_marker + (
        (df["imdb votes"] - min_votes) / (max_votes - min_votes)
    ) * (max_marker - min_marker)

    _, font_color = get_site_colors(dark_mode, contrast=False)
    fig = px.scatter(
        df,
        x="imdb rating_jitter",
        y="rating_jitter",
        color="rated_higher",
        hover_name="hover_title",
        hover_data={
            "rating": ":.1f",
            "imdb rating": ":.1f",
            "rating diff": ":.1f",
            "imdb votes": ":,",
            "rating_jitter": False,
            "imdb rating_jitter": False,
            "scaled_votes": False,
        },
        template="plotly_dark" if dark_mode else "plotly_white",
        size="scaled_votes",
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color=font_color)))
    fig.update_xaxes(title="rating")
    fig.update_yaxes(title="imdb rating")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="black", width=2),
            )
        ],
        legend_title_text="rating compared to imdb",
    )
    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=200, r=200, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig, {}


@timeit
def get_genre_df():
    path = Path(__file__).parents[2] / "data" / "gold" / "jpmdb"
    df = (
        pl.read_delta(str(path))
        .select(
            pl.col("watched_id").alias("id"),
            pl.col("primaryTitle").alias("title"),
            "rating",
            "genres",
        )
        .explode("genres")
        .rename({"genres": "genre"})
        .with_columns(pl.col("title").count().over("genre").alias("n_titles"))
        .with_columns(
            pl.when(pl.col("n_titles") < 100)
            .then(pl.lit("Other"))
            .otherwise(pl.col("genre"))
            .alias("genre")
        )
        .with_columns(pl.col("title").count().over("genre").alias("n_titles"))
        .with_columns(pl.col("rating").mean().over("genre").alias("rating_avg"))
        .sort("rating_avg", descending=True)
    )

    return df.to_pandas()


@callback(
    [
        Output("box-viz", "figure"),
        Output("box-viz", "style"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_box_genres(dark_mode: bool, screen_width: str):
    df = get_genre_df()
    df["title_hover"] = df["id"] + " - " + df["title"]
    bg_color, font_color = get_site_colors(dark_mode, contrast=False)
    genres = df["genre"].unique()
    n_genres = len(genres)
    n_colors = 10
    repeats = (n_genres // n_colors) + 1

    colors = list(
        pypalettes.load_cmap("Tableau_10", cmap_type="discrete", repeat=repeats).rgb  # type: ignore
    )[0:n_genres]

    colors = [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in colors]
    fig = px.box(
        df,
        color="genre",
        x="genre",
        y="rating",
        hover_name="title_hover",
        hover_data={"rating": ":.1f"},
        notched=True,
        template="plotly_dark" if dark_mode else "plotly_white",
        color_discrete_map={genre: color for genre, color in zip(genres, colors)},
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.5, line=dict(width=1, color=font_color)),
        boxpoints="all",
        jitter=0.8,
        hoverinfo="skip",
        selector=dict(type="box"),
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(range=[-0.5, 11.5], title="rating")

    for i, genre in enumerate(df["genre"].unique()):
        n_titles = df[df["genre"] == genre]["n_titles"].iloc[0]
        avg_rating = df[df["genre"] == genre]["rating_avg"].iloc[0]
        fig.add_annotation(
            x=i,
            y=10.75,
            text=f"n={n_titles}<br>avg={avg_rating:.1f}",
            showarrow=False,
            font=dict(color=font_color),
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=font_color, width=2),
            )
        ],
        legend_title_text="",
    )
    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=200, r=200, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig, {}


@callback(
    [
        Output("summary-table", "children"),
        Output("summary-table", "style"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_styled_summary_table(dark_mode: bool, breakpoint_name: str):
    df = get_records()
    df.drop(columns=["watched_year", "jp_title"], inplace=True)

    summary_style = get_dt_style(dark_mode)
    summary_style["style_table"]["height"] = "auto"
    sm_margins, lg_margins = get_fig_margins("table")

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

    return tbl, {}


def layout():
    return [
        dbc.Fade(
            id="summary-fade",
            children=[
                dcc.Graph(
                    id="timeseries-viz",
                    style={"visibility": "hidden"},
                    config={"displayModeBar": False},
                ),
                dcc.Graph(
                    id="box-viz",
                    style={"visibility": "hidden"},
                    config={"displayModeBar": False},
                ),
                dcc.Graph(
                    id="rating-diff-viz",
                    style={"visibility": "hidden"},
                    config={"displayModeBar": False},
                ),
                html.Br(),
                html.Div(id="summary-table", style={"visibility": "hidden"}),
            ],
            style={"transition": "opacity 200ms ease-in", "minHeight": "100vh"},
            is_in=False,
        )
    ]
