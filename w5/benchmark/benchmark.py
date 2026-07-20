import colorsys
import os
import polars as pl
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, dcc, html

CSV_PATH = f"{os.getcwd() if str(os.getcwd()).endswith('/benchmark') else os.getcwd() + '/benchmark'}/benchmark_results.csv"
TIME_COL = "total_us"
BACKEND_COL = "backend"

TEXT_SECONDARY = "#52514e"


def series_colors(n: int) -> list[str]:
    """Return n colors with hues equally spaced around the color wheel."""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.80)
        colors.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return colors

def load_data() -> tuple[pl.DataFrame, list[str], list[str]]:
    df = pl.read_csv(CSV_PATH)
    backends = df[BACKEND_COL].unique().sort().to_list()
    dependent_vars = [c for c in df.columns if c not in (BACKEND_COL, TIME_COL)]
    return df, backends, dependent_vars


df, backends, dependent_vars = load_data()
unique_values = {var: sorted(df[var].unique().to_list()) for var in dependent_vars}

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Backend Benchmark Explorer"),
        html.Div(
            [
                html.Label("Horizontal axis variable", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="x-axis-var",
                    options=[{"label": v, "value": v} for v in dependent_vars],
                    value=dependent_vars[0],
                    clearable=False,
                    style={"width": "320px"},
                ),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div(
            [
                dcc.Graph(id="bar-chart", style={"flex": "0 1 720px", "minWidth": 0}),
                html.Div(
                    id="sliders-container",
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "alignItems": "flex-start",
                        "gap": "36px",
                        "flex": "1 1 auto",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "24px"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "40px auto", "fontFamily": "sans-serif"},
)


@app.callback(
    Output("sliders-container", "children"),
    Input("x-axis-var", "value"),
)
def render_sliders(x_var: str):
    vertical_label_style = {
        "fontWeight": 600,
        "writingMode": "vertical-rl",
        "transform": "rotate(180deg)",
        "textAlign": "center",
        "marginTop": "8px",
    }
    sliders = []
    for var in dependent_vars:
        if var == x_var:
            continue
        values = unique_values[var]
        if len(values) == 1:
            # Only one value in the data — no slider needed, show it as a fixed label.
            sliders.append(
                html.Div(
                    [
                        html.Div(
                            f"{values[0]} (fixed)",
                            style={
                                "color": TEXT_SECONDARY,
                                "writingMode": "vertical-rl",
                                "transform": "rotate(180deg)",
                                "height": "260px",
                                "textAlign": "center",
                            },
                        ),
                        html.Label(var, style=vertical_label_style),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                    },
                )
            )
            continue
        marks = {i: str(v) for i, v in enumerate(values)}
        sliders.append(
            html.Div(
                [
                    dcc.Slider(
                        id={"type": "dep-slider", "var": var},
                        min=0,
                        max=len(values) - 1,
                        step=1,
                        value=0,
                        marks=marks,
                        vertical=True,
                        verticalHeight=260,
                        updatemode="drag",
                    ),
                    html.Label(var, style=vertical_label_style),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                },
            )
        )
    return sliders


@app.callback(
    Output("bar-chart", "figure"),
    Input("x-axis-var", "value"),
    Input({"type": "dep-slider", "var": ALL}, "value"),
    Input({"type": "dep-slider", "var": ALL}, "id"),
)
def update_chart(x_var: str, slider_values: list[int], slider_ids: list[dict]):
    filtered = df
    for slider_id, idx in zip(slider_ids, slider_values):
        var = slider_id["var"]
        value = unique_values[var][idx]
        filtered = filtered.filter(pl.col(var) == value)

    x_values = unique_values[x_var]

    fig = go.Figure()
    for backend, color in zip(backends, series_colors(len(backends))):
        subset = filtered.filter(pl.col(BACKEND_COL) == backend)
        means = (
            subset.group_by(x_var)
            .agg((pl.col(TIME_COL).mean() / 1000).alias("mean_ms"))
            .sort(x_var)
        )
        lookup = dict(zip(means[x_var].to_list(), means["mean_ms"].to_list()))
        y_values = [lookup.get(v) for v in x_values]
        fig.add_trace(
            go.Bar(
                name=backend,
                x=[str(v) for v in x_values],
                y=y_values,
                marker_color=color,
                marker_line_width=0,
            )
        )

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        xaxis_title=x_var,
        yaxis_title="total time (milliseconds)",
        legend_title_text=BACKEND_COL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        font=dict(color=TEXT_SECONDARY),
        margin=dict(t=60, r=20, b=40, l=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e7e2")

    return fig


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
