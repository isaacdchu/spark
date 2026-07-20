import atexit
import colorsys
import os
import re
import signal
import subprocess
import sys
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

CSV_PATH = f"{os.getcwd() if str(os.getcwd()).endswith('/benchmark') else os.getcwd() + '/benchmark'}/benchmark_results.csv"
TIME_COL = "total_us"
BACKEND_COL = "backend"
# The blocked backend whose register-tile pairing gets its own per-tile chart, and
# the two CSV columns that hold that tile's (Q, KV) dims.
BLOCK_BACKEND = "conv2d_block_rsck"
BLOCK_Q_COL = "block_q"
BLOCK_K_COL = "block_k"

# The sweep launcher and the status/log files live alongside the master CSV.
BENCHMARK_DIR = os.path.dirname(CSV_PATH)
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
STATUS_PATH = os.path.join(BENCHMARK_DIR, ".sweep_status")
LOG_PATH = os.path.join(BENCHMARK_DIR, ".sweep_log")

# All backends known to benchmark.cpp (the BACKENDS registry). Offered in the UI.
KNOWN_BACKENDS = [
    "conv2d_implicit_gemm_krsc",
    "conv2d_explicit_gemm_crsk",
    "conv2d_pipeline",
    "conv2d_padding_separate_krsc",
    "conv2d_simd_rsck",
    "conv2d_simd_krsc",
    "conv2d_block_rsck",
    "conv2d_ind_rsck",
    "conv2d_xnnpack",
]
# The subset that main() runs by default today (the contract's default `use` list).
DEFAULT_BACKENDS = [
    "conv2d_ind_rsck",
    "conv2d_block_rsck",
    "conv2d_simd_rsck",
    "conv2d_xnnpack",
]

# Defaults for the problem-shape / sweep inputs, matching the contract defaults.
DEFAULT_THREAD_COUNTS = "1,8"
# conv2d_block_rsck is the only blocked backend, so its register tile is the sole
# thing that varies across compiled binaries. Q and KV are independent, comma-separated
# lists (like every other sweep dimension); their cartesian product is the set of "Q KV"
# tile pairings turned into SCONE_BLOCK_CONFIGS entries below.
DEFAULT_BLOCK_Q = "6"
DEFAULT_BLOCK_KV = "2"
DEFAULT_BATCH_SIZES = "16,32"
DEFAULT_INPUT_SIZES = "64"
DEFAULT_KERNEL_SIZES = "3"
DEFAULT_INPUT_CHANNELS = "1,16,32,64"
DEFAULT_OUTPUT_CHANNELS = "1,16,32,64"
DEFAULT_STRIDES = "1"
DEFAULT_PADDINGS = "0"

TEXT_SECONDARY = "#52514e"

# Handle to the most recently launched sweep process (for the concurrency guard).
current_proc: "subprocess.Popen | None" = None


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


# Module-level view of the current data. These are repopulated by reload_data()
# both at import and after a sweep finishes, so callbacks always read fresh data.
df: pl.DataFrame
backends: list[str]
dependent_vars: list[str]
unique_values: dict[str, list]
sparse_vars: set[str]


def reload_data() -> None:
    """(Re)load the master CSV into the module-level globals the callbacks read.

    Called once at import and again whenever a sweep completes, so the chart,
    dropdown, and sliders reflect fresh data without restarting the server. If the
    CSV cannot be read, the existing data is left untouched rather than crashing.
    """
    global df, backends, dependent_vars, unique_values, sparse_vars
    try:
        new_df, new_backends, new_deps = load_data()
    except Exception:
        # No readable CSV yet (e.g. first launch before any run). Fall back to a
        # safe empty state so the layout and callbacks still build -- the user can
        # then launch a sweep from the control panel. Only overwrite existing data
        # on the first (import-time) load; a failed reload after a good load keeps
        # the current data untouched.
        if "df" not in globals() or df is None:
            df = pl.DataFrame()
            backends = []
            dependent_vars = []
            unique_values = {}
            sparse_vars = set()
        return
    df = new_df
    backends = new_backends
    dependent_vars = new_deps
    # Some columns are sparse: they only apply to certain backends (e.g. block_q /
    # block_k are populated for the blocked conv backends and null everywhere else).
    # Drop nulls before building the sorted value list so None never reaches sort()
    # (None and int are not comparable) and so a null value can never be selected.
    unique_values = {
        var: df[var].drop_nulls().unique().sort().to_list() for var in dependent_vars
    }
    # A column that contains any null is "sparse": the rows where it is null belong
    # to backends the variable does not apply to, and those rows must stay visible
    # no matter which value of the sparse variable is selected.
    sparse_vars = {var for var in dependent_vars if df[var].null_count() > 0}


reload_data()


def read_status() -> "dict[str, str] | None":
    """Parse the key=value status file the sweep writes. None if not present.

    Tolerates a partially written / stale file: blank and malformed lines are
    skipped rather than raising.
    """
    if not os.path.exists(STATUS_PATH):
        return None
    result: dict[str, str] = {}
    try:
        with open(STATUS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    except OSError:
        return None
    return result


def read_log_tail() -> str:
    """Return the latest rendered progress line from the sweep log.

    The benchmark binary draws its per-shape progress bar with carriage returns,
    so a single log line accumulates many '\\r'-separated frames. We split on both
    '\\r' and '\\n' and take the last non-empty frame, giving the live "backend
    [####----] 42% (54/128)" the user can watch advance between combo counter ticks.
    """
    if not os.path.exists(LOG_PATH):
        return ""
    try:
        with open(LOG_PATH, "rb") as fh:
            # Only the tail matters; avoid reading a large log fully.
            try:
                fh.seek(-8192, os.SEEK_END)
            except OSError:
                fh.seek(0)
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    frames = [f.strip() for f in re.split(r"[\r\n]", data) if f.strip()]
    return frames[-1] if frames else ""


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone we cannot signal.
        return True
    except OSError:
        return False
    return True


def is_run_active() -> bool:
    """True only if a sweep is genuinely in flight right now.

    We do NOT trust the status file's state alone: a run that was killed leaves a
    stale state=running behind, which would otherwise deadlock the Run button
    forever. Liveness is decided by an actual process: the one we launched this
    session (current_proc), or the pid the script recorded in the status file
    (which survives a dashboard restart). A non-terminal state with no live
    process means the file is stale.
    """
    if current_proc is not None and current_proc.poll() is None:
        return True
    status = read_status()
    if not status or status.get("state") not in ("starting", "compiling", "running"):
        return False
    pid_str = status.get("pid")
    if pid_str and pid_str.isdigit() and _pid_alive(int(pid_str)):
        return True
    return False


def terminate_stray_sweeps() -> None:
    """Best-effort reap of leftover sweep processes from a crashed/prior session.

    Sweeps run detached so a single poll cannot kill them, which means a
    force-quit dashboard can leave run_sweep.sh / benchmark_sweep churning in the
    background -- competing for CPU and clobbering the shared status/log/CSV
    files. Called before launching a fresh run (only once no run is active), so
    orphans cannot starve or corrupt it.
    """
    for pattern in ("benchmark/run_sweep.sh", "build/benchmark_sweep"):
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (OSError, ValueError):
            pass


def stop_current_run() -> None:
    """Terminate the sweep this dashboard launched, and its whole process group.

    Registered on interpreter exit / SIGTERM so closing the dashboard does not
    leave a detached sweep (and its benchmark_sweep child) running forever.
    """
    proc = current_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def combine_block_configs(block_q: str, block_kv: str) -> str:
    """Turn the independent Q and KV lists into a SCONE_BLOCK_CONFIGS string.

    Q and KV are each a comma-separated list; the compiled conv2d_block_rsck binaries
    are their cartesian product, one "Q KV" tile per (Q, KV) pairing. run_sweep.sh parses
    each config as the two ints "RSCK_Q RSCK_KV".
    """
    q_list = [p.strip() for p in block_q.split(",") if p.strip()] or DEFAULT_BLOCK_Q.split(",")
    kv_list = [p.strip() for p in block_kv.split(",") if p.strip()] or DEFAULT_BLOCK_KV.split(",")
    return ";".join(f"{q} {kv}" for q in q_list for kv in kv_list)


def build_sweep_env(
    backends_sel: list[str],
    thread_counts: str,
    block_q: str,
    block_kv: str,
    batch_sizes: str,
    input_sizes: str,
    kernel_sizes: str,
    input_channels: str,
    output_channels: str,
    strides: str,
    paddings: str,
) -> dict[str, str]:
    """Assemble the SCONE_* environment overrides for a sweep from UI inputs.

    Returned separately from os.environ so it can be asserted on directly; the
    launcher merges it over a copy of the current environment.
    """
    return {
        "SCONE_BENCH_BACKENDS": ",".join(backends_sel or []),
        "SCONE_THREAD_COUNTS": thread_counts,
        "SCONE_BLOCK_CONFIGS": combine_block_configs(block_q, block_kv),
        "SCONE_BENCH_BATCH_SIZES": batch_sizes,
        "SCONE_BENCH_INPUT_SIZES": input_sizes,
        "SCONE_BENCH_KERNEL_SIZES": kernel_sizes,
        "SCONE_BENCH_INPUT_CHANNELS": input_channels,
        "SCONE_BENCH_OUTPUT_CHANNELS": output_channels,
        "SCONE_BENCH_STRIDES": strides,
        "SCONE_BENCH_PADDINGS": paddings,
        "SCONE_BENCH_OUTPUT": CSV_PATH,
        "SCONE_BENCH_STATUS": STATUS_PATH,
        "SCONE_BENCH_LOG": LOG_PATH,
    }


def render_progress(status: "dict[str, str] | None"):
    """Build the progress display (bar + phase + detail) from a status dict."""
    if status is None:
        return html.Div("No run yet.", style={"color": TEXT_SECONDARY})

    state = status.get("state", "?")
    completed = status.get("completed", "0")
    total = status.get("total", "?")

    try:
        pct = 100.0 * int(completed) / int(total) if int(total) > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        pct = 0.0

    bar_color = {"done": "#4a9d6a", "error": "#c05a5a"}.get(state, "#5a7fc0")
    lines = [
        html.Div(
            f"state: {state}   ({completed}/{total} combos done)",
            style={"fontWeight": 600, "marginBottom": "6px"},
        ),
        html.Div(
            html.Div(
                style={
                    "width": f"{pct:.0f}%",
                    "height": "14px",
                    "backgroundColor": bar_color,
                    "borderRadius": "3px",
                    "transition": "width 0.3s",
                }
            ),
            style={
                "width": "100%",
                "height": "14px",
                "backgroundColor": "#e8e7e2",
                "borderRadius": "3px",
                "overflow": "hidden",
            },
        ),
    ]

    detail_bits = []
    if status.get("config_index") and status.get("config_total"):
        detail_bits.append(
            f"block config {status['config_index']}/{status['config_total']}"
        )
    if status.get("block_config"):
        detail_bits.append(f"[{status['block_config']}]")
    if status.get("threads"):
        detail_bits.append(f"threads={status['threads']}")
    if detail_bits:
        lines.append(
            html.Div(
                "  ".join(detail_bits),
                style={"color": TEXT_SECONDARY, "marginTop": "6px"},
            )
        )
    if status.get("message"):
        lines.append(
            html.Div(status["message"], style={"color": TEXT_SECONDARY, "marginTop": "4px"})
        )
    # A single combo is a full backend x shape sweep that can take minutes, so the
    # combo counter above barely moves. Show the binary's own live progress line so
    # the user can see it is actually working.
    if state in ("compiling", "running"):
        tail = read_log_tail()
        detail = tail if tail else "compiling benchmark binary (first build can take a minute)..."
        lines.append(
            html.Div(
                detail,
                style={
                    "color": TEXT_SECONDARY,
                    "marginTop": "8px",
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "whiteSpace": "pre",
                    "overflowX": "auto",
                },
            )
        )
    if status.get("error"):
        lines.append(
            html.Div(
                f"error: {status['error']}",
                style={"color": "#c05a5a", "fontWeight": 600, "marginTop": "4px"},
            )
        )
    return html.Div(lines)


app = Dash(__name__)

def _text_field(label: str, field_id: str, default: str, width: str = "180px"):
    return html.Div(
        [
            html.Label(label, style={"fontWeight": 600, "fontSize": "13px"}),
            dcc.Input(
                id=field_id,
                type="text",
                value=default,
                debounce=True,
                style={"width": width, "display": "block", "marginTop": "4px"},
            ),
        ]
    )


control_panel = html.Div(
    [
        html.H3("Run benchmark", style={"marginTop": 0}),
        html.Div(
            [
                html.Label("Backends", style={"fontWeight": 600, "fontSize": "13px"}),
                dcc.Dropdown(
                    id="in-backends",
                    options=[{"label": b, "value": b} for b in KNOWN_BACKENDS],
                    value=DEFAULT_BACKENDS,
                    multi=True,
                    style={"width": "560px", "marginTop": "4px"},
                ),
            ],
            style={"marginBottom": "12px"},
        ),
        html.Div(
            [
                _text_field("Thread counts", "in-threads", DEFAULT_THREAD_COUNTS),
                _text_field("Block tile Q", "in-block-q", DEFAULT_BLOCK_Q, width="120px"),
                _text_field("Block tile KV", "in-block-kv", DEFAULT_BLOCK_KV, width="120px"),
            ],
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        html.Div(
            [
                _text_field("Batch sizes", "in-batch", DEFAULT_BATCH_SIZES),
                _text_field("Input sizes", "in-input-size", DEFAULT_INPUT_SIZES),
                _text_field("Kernel sizes", "in-kernel", DEFAULT_KERNEL_SIZES),
                _text_field("Input channels", "in-in-ch", DEFAULT_INPUT_CHANNELS),
                _text_field("Output channels", "in-out-ch", DEFAULT_OUTPUT_CHANNELS),
                _text_field("Strides", "in-stride", DEFAULT_STRIDES),
                _text_field("Paddings", "in-padding", DEFAULT_PADDINGS),
            ],
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginBottom": "16px"},
        ),
        html.Div(
            [
                html.Button(
                    "Run",
                    id="run-button",
                    n_clicks=0,
                    style={
                        "padding": "8px 20px",
                        "fontWeight": 600,
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="run-status",
                    children=render_progress(read_status()),
                    style={"flex": "1 1 auto"},
                ),
            ],
            style={"display": "flex", "gap": "24px", "alignItems": "center"},
        ),
    ],
    style={
        "border": "1px solid #e8e7e2",
        "borderRadius": "6px",
        "padding": "20px",
        "marginBottom": "28px",
    },
)


app.layout = html.Div(
    [
        html.H2("Backend Benchmark Explorer"),
        control_panel,
        dcc.Interval(id="sweep-interval", interval=1500, disabled=True),
        dcc.Store(id="data-version", data=0),
        dcc.Store(id="chart-mode", data="same"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Horizontal axis variable", style={"fontWeight": 600}
                        ),
                        dcc.Dropdown(
                            id="x-axis-var",
                            options=[{"label": v, "value": v} for v in dependent_vars],
                            value=dependent_vars[0] if dependent_vars else None,
                            clearable=False,
                            style={"width": "320px"},
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Label("Chart layout", style={"fontWeight": 600}),
                        html.Button(
                            "Same chart (shared y-axis)",
                            id="layout-toggle",
                            n_clicks=0,
                            style={
                                "display": "block",
                                "marginTop": "4px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "flex-end",
                "gap": "24px",
                "marginBottom": "24px",
            },
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
        html.Div(
            [
                html.Label(
                    f"{BLOCK_BACKEND}: block-tile (Q/KV) comparison",
                    style={"fontWeight": 600},
                ),
                dcc.Graph(id="block-chart", style={"width": "100%"}),
            ],
            style={"marginTop": "32px"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "40px auto", "fontFamily": "sans-serif"},
)


@app.callback(
    Output("x-axis-var", "options"),
    Input("data-version", "data"),
)
def refresh_x_axis_options(_version):
    # Keep the x-axis choices in sync with the (possibly reloaded) columns.
    return [{"label": v, "value": v} for v in dependent_vars]


@app.callback(
    Output("sliders-container", "children"),
    Input("x-axis-var", "value"),
    Input("data-version", "data"),
)
def render_sliders(x_var: str, _version):
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
        if len(values) <= 1:
            # Zero non-null values means the column is entirely null (e.g. an old CSV
            # with no block sweeping) -- treat it as absent. One value means a single
            # fixed value. Either way there is nothing to slide, so show a label.
            if len(values) == 0:
                fixed_label = "(n/a)"
            else:
                fixed_label = f"{values[0]} (fixed)"
            sliders.append(
                html.Div(
                    [
                        html.Div(
                            fixed_label,
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
    Output("chart-mode", "data"),
    Output("layout-toggle", "children"),
    Input("layout-toggle", "n_clicks"),
)
def toggle_chart_mode(n_clicks: int):
    # Even clicks -> one shared chart; odd clicks -> one subplot per backend so a
    # slow configuration can't flatten the fast backends onto an unreadable scale.
    if n_clicks % 2 == 1:
        return "separate", "Separate charts (per-backend y-axis)"
    return "same", "Same chart (shared y-axis)"


@app.callback(
    Output("bar-chart", "figure"),
    Input("x-axis-var", "value"),
    Input({"type": "dep-slider", "var": ALL}, "value"),
    Input({"type": "dep-slider", "var": ALL}, "id"),
    Input("chart-mode", "data"),
    Input("data-version", "data"),
)
def update_chart(
    x_var: str,
    slider_values: list[int],
    slider_ids: list[dict],
    chart_mode: str = "same",
    _version=0,
):
    if not x_var or df.is_empty():
        # No data loaded yet (no CSV before the first run). Show an empty chart
        # with a hint rather than erroring; it refreshes once a sweep finishes.
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            font=dict(color=TEXT_SECONDARY),
            margin=dict(t=60, r=20, b=40, l=60),
            annotations=[
                dict(
                    text="No benchmark data yet -- run a sweep from the panel above.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=14, color=TEXT_SECONDARY),
                )
            ],
        )
        return fig

    filtered = df
    for slider_id, idx in zip(slider_ids, slider_values):
        var = slider_id["var"]
        value = unique_values[var][idx]
        if var in sparse_vars:
            # Keep rows matching the selected value AND rows where the variable does
            # not apply (null) -- otherwise selecting a block size would hide every
            # backend that has no block size at all.
            filtered = filtered.filter(
                (pl.col(var) == value) | pl.col(var).is_null()
            )
        else:
            filtered = filtered.filter(pl.col(var) == value)

    x_values = unique_values[x_var]
    colors = series_colors(len(backends))

    def bars_for(backend):
        # Mean time (ms) per x-value for one backend, aligned to x_values order.
        subset = filtered.filter(pl.col(BACKEND_COL) == backend)
        means = (
            subset.group_by(x_var)
            .agg((pl.col(TIME_COL).mean() / 1000).alias("mean_ms"))
            .sort(x_var)
        )
        lookup = dict(zip(means[x_var].to_list(), means["mean_ms"].to_list()))
        return [lookup.get(v) for v in x_values]

    if chart_mode == "separate":
        # One subplot per x-axis value, each with its own (independent) y-axis so a
        # slow configuration doesn't crush the faster ones onto an unreadable scale.
        # Within each subplot the bars are the backends, so they stay comparable.
        per_backend = {backend: bars_for(backend) for backend in backends}
        fig = make_subplots(
            rows=1,
            cols=len(x_values),
            subplot_titles=[str(v) for v in x_values],
            horizontal_spacing=0.04,
        )
        for col, _x in enumerate(x_values, start=1):
            j = col - 1
            fig.add_trace(
                go.Bar(
                    x=list(backends),
                    y=[per_backend[backend][j] for backend in backends],
                    marker_color=list(colors),
                    marker_line_width=0,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.update_xaxes(showgrid=False, row=1, col=col)
            fig.update_yaxes(showgrid=True, gridcolor="#e8e7e2", row=1, col=col)
        fig.update_yaxes(title_text="total time (milliseconds)", row=1, col=1)
        fig.update_layout(
            bargap=0.25,
            template="plotly_white",
            font=dict(color=TEXT_SECONDARY),
            margin=dict(t=60, r=20, b=40, l=60),
            title_text=f"per {x_var} value",
        )
        return fig

    fig = go.Figure()
    for backend, color in zip(backends, colors):
        fig.add_trace(
            go.Bar(
                name=backend,
                x=[str(v) for v in x_values],
                y=bars_for(backend),
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


def _hint_figure(text: str) -> go.Figure:
    """An empty white chart carrying a single centered hint -- used before any data."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        font=dict(color=TEXT_SECONDARY),
        margin=dict(t=60, r=20, b=40, l=60),
        annotations=[
            dict(
                text=text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=TEXT_SECONDARY),
            )
        ],
    )
    return fig


@app.callback(
    Output("block-chart", "figure"),
    Input("x-axis-var", "value"),
    Input({"type": "dep-slider", "var": ALL}, "value"),
    Input({"type": "dep-slider", "var": ALL}, "id"),
    Input("data-version", "data"),
)
def update_block_chart(
    x_var: str, slider_values: list[int], slider_ids: list[dict], _version=0
):
    """Grouped bars for conv2d_block_rsck alone: within each x-axis group, one bar
    per distinct register-tile (Q, KV) pairing. The block_q / block_k sliders are
    NOT applied here -- those dims are the series, so every swept tile stays visible.
    """
    if not x_var or df.is_empty() or BLOCK_BACKEND not in backends:
        return _hint_figure(
            f"No {BLOCK_BACKEND} data yet -- run a sweep with it selected."
        )

    filtered = df.filter(pl.col(BACKEND_COL) == BLOCK_BACKEND)
    for slider_id, idx in zip(slider_ids, slider_values):
        var = slider_id["var"]
        if var in (BLOCK_Q_COL, BLOCK_K_COL):
            # The tile dims are the series dimension of this chart -- don't pin them.
            continue
        value = unique_values[var][idx]
        if var in sparse_vars:
            filtered = filtered.filter((pl.col(var) == value) | pl.col(var).is_null())
        else:
            filtered = filtered.filter(pl.col(var) == value)

    # Distinct (Q, KV) tile pairings still present after filtering, in sorted order.
    pairings = (
        filtered.select([BLOCK_Q_COL, BLOCK_K_COL])
        .unique()
        .drop_nulls()
        .sort([BLOCK_Q_COL, BLOCK_K_COL])
        .rows()
    )

    if not pairings:
        return _hint_figure(f"No {BLOCK_BACKEND} rows match the current selection.")

    x_values = unique_values[x_var]

    fig = go.Figure()
    for (bq, bk), color in zip(pairings, series_colors(len(pairings))):
        subset = filtered.filter(
            (pl.col(BLOCK_Q_COL) == bq) & (pl.col(BLOCK_K_COL) == bk)
        )
        means = (
            subset.group_by(x_var)
            .agg((pl.col(TIME_COL).mean() / 1000).alias("mean_ms"))
            .sort(x_var)
        )
        lookup = dict(zip(means[x_var].to_list(), means["mean_ms"].to_list()))
        y_values = [lookup.get(v) for v in x_values]
        fig.add_trace(
            go.Bar(
                name=f"Q={bq}, KV={bk}",
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
        legend_title_text="block tile",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        font=dict(color=TEXT_SECONDARY),
        margin=dict(t=60, r=20, b=40, l=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e7e2")

    return fig


@app.callback(
    Output("run-status", "children"),
    Output("run-button", "disabled"),
    Output("sweep-interval", "disabled"),
    Input("run-button", "n_clicks"),
    State("in-backends", "value"),
    State("in-threads", "value"),
    State("in-block-q", "value"),
    State("in-block-kv", "value"),
    State("in-batch", "value"),
    State("in-input-size", "value"),
    State("in-kernel", "value"),
    State("in-in-ch", "value"),
    State("in-out-ch", "value"),
    State("in-stride", "value"),
    State("in-padding", "value"),
    prevent_initial_call=True,
)
def start_run(
    n_clicks,
    backends_sel,
    thread_counts,
    block_q,
    block_kv,
    batch_sizes,
    input_sizes,
    kernel_sizes,
    input_channels,
    output_channels,
    strides,
    paddings,
):
    if not n_clicks:
        raise PreventUpdate
    # Concurrency guard: never launch a second sweep on top of a live one.
    if is_run_active():
        return render_progress(read_status()), True, False

    overrides = build_sweep_env(
        backends_sel,
        thread_counts,
        block_q,
        block_kv,
        batch_sizes,
        input_sizes,
        kernel_sizes,
        input_channels,
        output_channels,
        strides,
        paddings,
    )
    env = os.environ.copy()
    env.update(overrides)

    # No run is active (guard above), so any lingering sweep process is an orphan
    # from a crashed session -- reap it before launching so it cannot compete for
    # CPU or interleave into the shared status/log/CSV files.
    terminate_stray_sweeps()

    # Reset any prior status so the poller does not immediately read a stale "done".
    try:
        with open(STATUS_PATH, "w") as fh:
            fh.write("state=starting\n")
    except OSError:
        pass

    global current_proc
    # Detached, non-blocking: the script writes progress to STATUS_PATH / LOG_PATH.
    current_proc = subprocess.Popen(
        ["bash", "benchmark/run_sweep.sh"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Disable Run, enable the polling interval.
    return render_progress(read_status()), True, False


@app.callback(
    Output("run-status", "children", allow_duplicate=True),
    Output("run-button", "disabled", allow_duplicate=True),
    Output("sweep-interval", "disabled", allow_duplicate=True),
    Output("data-version", "data"),
    Input("sweep-interval", "n_intervals"),
    State("data-version", "data"),
    prevent_initial_call=True,
)
def poll_status(_n_intervals, version):
    status = read_status()
    display = render_progress(status)
    state = status.get("state") if status else None

    if state == "done":
        # Fresh results: reload globals and bump the version to refresh chart/UI.
        reload_data()
        return display, False, True, (version or 0) + 1
    if state == "error":
        # Surface the error, stop polling, re-enable Run.
        return display, False, True, no_update
    # starting / compiling / running (or missing): keep polling, Run stays disabled.
    return display, True, False, no_update


def main() -> None:
    # Ensure a launched sweep is torn down when the dashboard exits, so runs do
    # not orphan into the background. atexit covers normal exit / Ctrl-C (which
    # Werkzeug turns into a clean shutdown); the SIGTERM handler makes `kill` and
    # editor/terminal stops also run the cleanup.
    atexit.register(stop_current_run)
    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except (ValueError, OSError):
        pass
    # use_reloader=False: the reloader runs a second process that would launch and
    # then orphan sweeps on every code change. One process keeps child ownership clear.
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
