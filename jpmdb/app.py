import argparse
import os

import dash
import dash_bootstrap_components as dbc
import dash_breakpoints
from dash import Input, Output, State, callback, dcc, html

from jpmdb.pages import summary
from jpmdb.styling import ScreenWidth, SitePalette, get_site_colors


@callback(
    [
        Output("navbar-brand", "style"),
    ],
    [
        Input("current-url", "pathname"),
        Input("color-mode-switch", "value"),
    ],
)
def update_downloads_link_color(pathname: str, dark_mode: bool):
    _, color = get_site_colors(dark_mode, contrast=False)
    highlighted_background_color, highlighted_text_color = get_site_colors(
        dark_mode, contrast=True
    )

    pages = [""]
    output_styles = [{"color": color} for _ in range(len(pages))]

    if pathname in ["network"]:
        return output_styles

    current_page = pathname.removeprefix("/").split("-")[0]

    output_styles[pages.index(current_page)] = {
        "color": highlighted_text_color,
        "backgroundColor": highlighted_background_color,
        "borderRadius": "20px",
    }

    return output_styles


@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("navbar-collapse", "is_open", allow_duplicate=True),
    Input("current-url", "pathname"),
    prevent_initial_call=True,
)
def close_navbar_on_navigate(_):
    return False


@callback(
    [
        Output("color-mode-switch", "children"),
        Output("color-mode-switch", "value"),
    ],
    Input("color-mode-switch", "n_clicks"),
    State("color-mode-switch", "children"),
)
def toggle_color_mode(n_clicks, _):
    is_dark = n_clicks % 2 == 1
    if is_dark:
        return html.I(
            className="fas fa-sun",
            style={"color": SitePalette.PAGE_BACKGROUND_COLOR_LIGHT},
        ), True

    return html.I(
        className="fas fa-moon",
        style={"color": SitePalette.PAGE_BACKGROUND_COLOR_DARK},
    ), False


@callback(
    [
        Output("jpmdb-page", "className"),
        Output("navbar", "className"),
    ],
    Input("color-mode-switch", "value"),
)
def toggle_page_color(dark_mode: bool):
    class_name = "dark-mode" if dark_mode else "light-mode"
    return class_name, class_name


def layout():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    "jpmdb", href="/", class_name="navbar-brand", id="navbar-brand"
                ),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dcc.Location("current-url", refresh=False),
                        ]
                    )
                ),
                dbc.NavItem(
                    dbc.Button(
                        id="color-mode-switch",
                        n_clicks=0,
                        children=html.I(
                            className="fas fa-moon",
                            style={
                                "color": SitePalette.PAGE_BACKGROUND_COLOR_DARK,
                            },
                        ),
                        color="link",
                    )
                ),
            ],
            fluid=True,
            id="navbar-container",
        ),
        id="navbar",
    )

    return dbc.Container(
        id="jpmdb-page",
        children=[
            navbar,
            html.Br(),
            dash_breakpoints.WindowBreakpoints(
                id="breakpoints",
                widthBreakpointThresholdsPx=[
                    768,
                    1200,
                    1920,
                    2560,
                ],
                widthBreakpointNames=[i.value for i in ScreenWidth],
            ),
            dash.page_container,
        ],
        fluid=True,
    )


def init_app(env: str = "prod"):
    serve_locally = {"dev": True, "prod": False}[env]

    app = dash.Dash(
        use_pages=True,
        external_stylesheets=["assets/css/bootstrap.min.css", dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True,
        compress=True,
        serve_locally=serve_locally,
    )

    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                body, #navbar {{
                    background-color: {SitePalette.PAGE_BACKGROUND_COLOR_LIGHT} !important;
                    margin: 0;
                }}
                
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """
    server = app.server

    app.layout = layout()

    dash.register_page(
        summary.__name__, name="summary", path="/", layout=summary.layout
    )

    return app, server


def run_app(app):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment to run the app in. Default is prod",
    )
    envs = {
        # "prod": {"host": "0.0.0.0", "debug": False},
        "dev": {"host": "127.0.0.1", "debug": True},
    }

    env = parser.parse_args().env
    os.environ["DASHBOARD_ENV"] = env

    app.run(host=envs[env]["host"], debug=envs[env]["debug"])


app, server = init_app()

if __name__ == "__main__":
    run_app(app)
