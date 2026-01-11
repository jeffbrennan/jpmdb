import datetime
from functools import lru_cache
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
    lg_width = 52.5
    margin_left = 0

    if fig_type == "timeseries":
        lg_width = 70
        margin_left = 0

    sm_margins = {"maxWidth": f"{sm_width}vw", "width": f"{sm_width}vw"}
    lg_margins = {
        "maxWidth": f"{lg_width}vw",
        "width": f"{lg_width}vw",
        "marginLeft": f"{margin_left}vw",
        "marginRight": "0vw",
    }
    return sm_margins, lg_margins


@timeit
def get_records() -> pd.DataFrame:
    return _get_records_cached().copy()


@lru_cache(maxsize=1)
def _get_records_cached() -> pd.DataFrame:
    path = Path(__file__).parents[2] / "data" / "gold" / "jpmdb"
    df = (
        pl.read_delta(str(path))
        .with_columns(
            [
                pl.col("genres").list.join(", ").alias("genres"),
                pl.when(pl.col("season").is_not_null())
                .then(
                    pl.concat_str(
                        pl.col("primaryTitle"), pl.lit(" Season "), pl.col("season")
                    )
                )
                .otherwise(pl.col("primaryTitle"))
                .alias("title_with_season"),
                pl.when(pl.col("titleType").str.starts_with("tv"))
                .then(pl.lit("tv"))
                .otherwise(pl.col("titleType"))
                .alias("type"),
            ]
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("["),
                    pl.col("title_with_season"),
                    pl.lit("]"),
                    pl.lit("("),
                    pl.lit("https://www.imdb.com/title/"),
                    pl.col("tconst"),
                    pl.lit(")"),
                ]
            ).alias("title"),

        )
        .select(
            [
                pl.col("watched_id").alias("id"),
                "watched_year",
                "title",
                "type",
                pl.col("startYear").alias("year"),
                "genres",
                pl.col("rating").alias("juan"),
                pl.col("averageRating").alias("imdb"),
                (pl.col("rating") - pl.col("averageRating")).alias("diff"),
                pl.col("numVotes").alias("votes"),
            ]
        )
    )
    return df.to_pandas()


@timeit
def style_timeseries_fig(
    fig: Figure,
    watched_id_list: list[str],
    font_color: str,
    screen_width: str,
    point_size: int,
) -> Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(categoryorder="array", categoryarray=watched_id_list)
    fig.update_yaxes(title="rating", showline=False, showgrid=False, zeroline=False, title_standoff=5)

    trace_color = font_color.replace("rgb", "rgba").replace(")", ", 0.5)")
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=1, color=trace_color))
    )

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
                x0=0.005,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=font_color, width=2),
            )
        ],
        legend_title_text="",
        legend=dict(x=1.005, xanchor="left", itemsizing="constant"),
    )

    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=70, r=50, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig


@lru_cache(maxsize=1)
def _compute_trendline():
    df = _get_records_cached()
    watched_id_list = df["id"].to_list()
    smoothed = lowess(df["juan"], range(len(watched_id_list)), frac=0.025)
    x_trend = tuple(watched_id_list[i] for i, _ in enumerate(smoothed[:, 0]))
    y_trend = tuple(smoothed[:, 1])
    years = sorted(set(i.split("_")[0] for i in watched_id_list))
    return x_trend, y_trend, years, watched_id_list


@timeit
def add_timeseries_trendline(
    df: pd.DataFrame, fig: Figure, watched_id_list: list[str], font_color: str
):
    x_trend, y_trend, years, _ = _compute_trendline()
    for year in years[1:]:
        boundary_id = watched_id_list.index(year + "_001")
        fig.add_vline(x=boundary_id, line_color=font_color, layer="below")

    outer_line_width = 6
    inner_line_width = 3

    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(shape="spline", color=font_color, width=outer_line_width),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(
                shape="spline", color="rgb(247, 111, 83)", width=inner_line_width
            ),
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
        Output("summary-fade-bottom", "is_in"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
@timeit
def get_timeseries_viz(dark_mode: bool, screen_width: str):
    fig, style = _timeseries_viz(dark_mode, screen_width, False)
    return fig, style, True


@callback(
    [
        Output("timeseries-viz-latest", "figure"),
        Output("timeseries-viz-latest", "style"),
        Output("timeseries-latest-ready", "data"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
@timeit
def get_timeseries_viz_latest(dark_mode: bool, screen_width: str):
    fig, style = _timeseries_viz(dark_mode, screen_width, True)
    return fig, style, True


def _timeseries_viz(dark_mode: bool, screen_width: str, filter_latest_year: bool):
    df = get_records()
    if filter_latest_year:
        latest_year = datetime.datetime.now().year - 1
        df = df.query(f"watched_year >= {latest_year}")

    df["watched_year"] = df["watched_year"].astype(str)
    df["title_hover"] = df["id"] + " - " + df["title"]

    _, font_color = get_site_colors(dark_mode, contrast=False)

    watched_id_list: list[str] = df["id"].to_list()  # pyright: ignore[reportAssignmentType]
    colors = qualitative.Plotly

    types = list(df["type"].unique())
    colors = get_color_palette(types)
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(types)}

    template = "plotly_dark" if dark_mode else "plotly_white"
    fig = px.scatter(
        df,
        x="id",
        y="juan",
        color="type",
        color_discrete_map=color_map,
        hover_name="title_hover",
        hover_data={
            "juan": ":.1f",
            "imdb": ":.1f",
            "diff": ":.1f",
            "votes": ":,",
            "id": False,
        },
        template=template,
        opacity=0.8,
    )
    point_size = 12 if filter_latest_year else 6
    fig = style_timeseries_fig(
        fig, watched_id_list, font_color, screen_width, point_size
    )
    if not filter_latest_year:
        fig = add_timeseries_trendline(df, fig, watched_id_list, font_color)

    return fig, {}


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
@timeit
def get_rating_diff_viz(dark_mode: bool, screen_width: str):
    df = get_records()[
        [
            "id",
            "title",
            "juan",
            "imdb",
            "diff",
            "votes",
        ]
    ]

    df = df.dropna(subset=["juan", "imdb"])

    df["rated_higher"] = df["diff"] > 0
    df["rated_higher"] = np.where(df["diff"] > 0, "higher than imdb", "lower than imdb")
    df["hover_title"] = df["id"] + " - " + df["title"]

    jitter = 0.05
    df["rating_jitter"] = df["juan"] + np.random.uniform(-jitter, jitter, size=len(df))
    df["imdb rating_jitter"] = df["imdb"] + np.random.uniform(
        -jitter, jitter, size=len(df)
    )

    min_marker = 5
    max_marker = 30
    min_votes = df["votes"].min()
    max_votes = df["votes"].max()
    df["scaled_votes"] = min_marker + (
        (df["votes"] - min_votes) / (max_votes - min_votes)
    ) * (max_marker - min_marker)

    _, font_color = get_site_colors(dark_mode, contrast=False)
    fig = px.scatter(
        df,
        x="imdb rating_jitter",
        y="rating_jitter",
        color="rated_higher",
        hover_name="hover_title",
        hover_data={
            "juan": ":.1f",
            "imdb": ":.1f",
            "diff": ":.1f",
            "votes": ":,",
            "rated_higher": False,
            "rating_jitter": False,
            "imdb rating_jitter": False,
            "scaled_votes": False,
        },
        template="plotly_dark" if dark_mode else "plotly_white",
        size="scaled_votes",
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color=font_color)))
    fig.update_xaxes(
        title="imdb rating", showline=False, showgrid=False, zeroline=False
    )
    fig.update_yaxes(
        title="rating", showline=False, showgrid=False, zeroline=False, title_standoff=5
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.005,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=font_color, width=2),
            )
        ],
        legend_title_text="rating compared to imdb",
        legend=dict(x=1.005, xanchor="left"),
    )
    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=70, r=50, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig, {}


@timeit
def get_genre_df():
    return _get_genre_df_cached().copy()


@lru_cache(maxsize=1)
def _get_genre_df_cached():
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
@timeit
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
    fig.update_xaxes(title="", showline=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        range=[-0.5, 11.5],
        title="rating",
        showline=False,
        showgrid=False,
        zeroline=False,
        title_standoff=5,
    )

    for i, genre in enumerate(df["genre"].unique()):
        n_titles = df[df["genre"] == genre]["n_titles"].iloc[0]
        avg_rating = df[df["genre"] == genre]["rating_avg"].iloc[0]
        fig.add_annotation(
            x=i,
            y=10.75,
            text=f"n={n_titles}<br>{avg_rating:.1f}",
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
                x0=0.005,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=font_color, width=2),
            )
        ],
        legend_title_text="",
        legend=dict(x=1.005, xanchor="left"),
    )
    if screen_width != ScreenWidth.xs:
        fig.update_layout(
            margin=dict(l=70, r=50, t=50, b=0),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

    return fig, {}


@callback(
    [
        Output("ratings-histogram", "figure"),
        Output("ratings-histogram", "style"),
        Output("histogram-ready", "data"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_ratings_histogram(dark_mode: bool, screen_width: str):
    df = get_records()
    df["year"] = df["watched_year"].astype(str)
    bg_color, font_color = get_site_colors(dark_mode, contrast=False)
    fig = px.histogram(
        df,
        x="juan",
        nbins=10,
        facet_col="year",
        facet_col_wrap=2,
        facet_row_spacing=0.02,
        template="plotly_dark" if dark_mode else "plotly_white",
    )

    fig.update_yaxes(matches=None, rangemode="tozero")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    num_years = len(df["year"].unique())
    num_rows = (num_years + 1) // 2
    row_height = 1.0 / num_rows
    label_space = 0.03
    plot_height = row_height - label_space

    for i in range(num_years):
        row_num = i // 2
        row_bottom = 1.0 - ((row_num + 1) * row_height)
        row_top = row_bottom + plot_height

        yaxis_name = "yaxis" if i == 0 else f"yaxis{i + 1}"
        if hasattr(fig.layout, yaxis_name):
            fig.layout[yaxis_name].domain = [row_bottom, row_top]

    for i, annotation in enumerate(fig.layout.annotations):
        row_num = i // 2
        row_top = 1.0 - (row_num * row_height)
        annotation.y = row_top - 0.005
        annotation.yanchor = "top"
        annotation.font = dict(size=14, weight="bold")

    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color=font_color)))
    fig.update_xaxes(
        title=None, showline=False, showgrid=False, zeroline=False, showticklabels=False
    )
    fig.update_yaxes(
        title=None, showline=False, showgrid=False, zeroline=False, showticklabels=False
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    num_cols = 2

    for i in range(1, num_cols):
        fig.add_shape(
            type="line",
            x0=i / num_cols,
            y0=0,
            x1=i / num_cols,
            y1=1,
            yref="paper",
            xref="paper",
            line=dict(color=font_color, width=2),
        )

    for i in range(1, num_rows):
        fig.add_shape(
            type="line",
            x0=0,
            y0=i / num_rows,
            x1=1,
            y1=i / num_rows,
            yref="paper",
            xref="paper",
            line=dict(color=font_color, width=2),
        )

    return fig, {}, True


@callback(
    Output("summary-fade-top", "is_in"),
    [
        Input("table-ready", "data"),
        Input("histogram-ready", "data"),
        Input("timeseries-latest-ready", "data"),
    ],
)
def trigger_top_fade(table_ready, histogram_ready, timeseries_latest_ready):
    return table_ready and histogram_ready and timeseries_latest_ready


@callback(
    [
        Output("summary-table", "children"),
        Output("summary-table", "style"),
        Output("table-ready", "data"),
    ],
    [
        Input("color-mode-switch", "value"),
        Input("breakpoints", "widthBreakpoint"),
    ],
)
def get_styled_summary_table(dark_mode: bool, breakpoint_name: str):
    df = get_records().drop(columns=["watched_year", "genres"])

    summary_style = get_dt_style(dark_mode)
    summary_style["style_table"]["height"] = "85vh"
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
        "juan": float_style,
        "imdb": float_style,
        "diff": float_style,
        "votes": int_style,
    }

    width_mapping = {
        "id": 75,
        "title": 200,
        "type": 65,
        "year": 65,
        "rating": 65,
        "imdb rating": 65,
        "diff": 65,
        "votes": 65,
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

    rating_diff_styles = [
        {
            "if": {
                "filter_query": "{diff} < 0",
                "column_id": "diff",
            },
            "color": "#EF553B",
        },
        {
            "if": {
                "filter_query": "{diff} > 0",
                "column_id": "diff",
            },
            "color": "#636EFA",
        },
    ]

    # Color scale for rating column (salmon low, blue high)
    summary_style["style_data_conditional"].extend(rating_diff_styles)

    for col in df.columns:
        if col in col_mapping:
            tbl_cols.append({**col_mapping[col], "id": col, "name": col})
        else:
            tbl_cols.append({"id": col, "name": col})

    tbl = dash_table.DataTable(
        df.to_dict("records"),
        columns=tbl_cols,
        sort_by=[{"column_id": "id", "direction": "desc"}],
        **summary_style,
    )

    return tbl, {}, True


def layout():
    return [
        dcc.Store(id="table-ready", data=False),
        dcc.Store(id="histogram-ready", data=False),
        dcc.Store(id="timeseries-latest-ready", data=False),
        dbc.Fade(
            id="summary-fade-top",
            children=[
                dbc.Row(
                    children=[
                        dbc.Col(
                            id="summary-table",
                            width=7,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="ratings-histogram",
                                config={"displayModeBar": False},
                                style={"height": "85vh"},
                            ),
                            width=5,
                        ),
                    ],
                ),
                dcc.Graph(
                    id="timeseries-viz-latest",
                    config={"displayModeBar": False},
                ),
            ],
            style={"transition": "opacity 200ms ease-in"},
            is_in=False,
        ),
        dbc.Fade(
            id="summary-fade-bottom",
            children=[
                dcc.Graph(
                    id="timeseries-viz",
                    config={"displayModeBar": False},
                ),
                dcc.Graph(
                    id="rating-diff-viz",
                    config={"displayModeBar": False},
                ),
                dcc.Graph(
                    id="box-viz",
                    config={"displayModeBar": False},
                ),
            ],
            style={"transition": "opacity 200ms ease-in"},
            is_in=False,
        ),
    ]
