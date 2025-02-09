from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, callback, dash_table, dcc, html
from plotly.colors import qualitative
from plotly.graph_objs import Figure
from statsmodels.nonparametric.smoothers_lowess import lowess

from jpmdb.common import timeit
from jpmdb.styling import ScreenWidth, get_dt_style, get_site_colors


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
    fig: Figure, watched_id_list: list[str], font_color: str
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

    return fig


@timeit
def add_timeseries_trendline(
    df: pd.DataFrame, fig: Figure, watched_id_list: list[str], font_color: str
):
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

    years = sorted(set(i.split("_")[0] for i in watched_id_list))
    for year in years[1:]:
        boundary_id = watched_id_list.index(year + "_001")
        fig.add_vline(x=boundary_id, line_color=font_color)

    return fig


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
def get_timeseries_viz(dark_mode: bool, _: str):
    df = get_records()
    df["watched_year"] = df["watched_year"].astype(str)
    df["jp_title_hover"] = df["id"] + " - " + df["jp_title"]

    _, font_color = get_site_colors(dark_mode, contrast=False)

    watched_id_list = df["id"].to_list()
    colors = qualitative.Plotly
    unique_years = sorted(df["watched_year"].unique())
    color_map = {year: colors[i % len(colors)] for i, year in enumerate(unique_years)}

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
        opacity=0.7,
    )
    fig = style_timeseries_fig(fig, watched_id_list, font_color)
    fig = add_timeseries_trendline(df, fig, watched_id_list, font_color)
    return fig, {}, True


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
                html.Div(id="summary-table", style={"visibility": "hidden"}),
            ],
            style={"transition": "opacity 200ms ease-in", "minHeight": "100vh"},
            is_in=False,
        )
    ]
