"""
G-G Diagram / Traction Circle Analyzer
========================================
Streamlit application for telemetry data analysis from a race car.

Usage:
    pip install streamlit pandas numpy plotly scipy pyarrow
    streamlit run traction_circle.py

Note: The speed column is interpreted in m/s (converted to km/h internally
for binning and display purposes).
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="G-G Diagram Analyzer", layout="wide")

MS_TO_KMH = 3.6

# ----------------------------------------------------------------------------
# Synthetic data generator
# ----------------------------------------------------------------------------
def generate_synthetic_data(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 90, n)

    lateral = (
        6.5 * np.sin(t * 0.28 + 0.3)
        + 2.5 * np.sin(t * 0.11 + 1.7)
        + 1.2 * np.sin(t * 0.55)
    )
    longitudinal = (
        4.0 * np.cos(t * 0.19 + 0.6)
        - 2.0 * np.sin(t * 0.28 + 0.3) ** 2
    )

    lateral = lateral / np.max(np.abs(lateral)) * 2.1 * 9.81
    longitudinal = longitudinal / np.max(np.abs(longitudinal)) * 1.8 * 9.81

    noise_x = rng.normal(0, 0.35, n)
    noise_y = rng.normal(0, 0.35, n)

    acc_x = longitudinal + noise_x
    acc_y = lateral + noise_y

    # Speed in km/h for realism, then converted to m/s (target column format)
    speed_kmh = 115 + 55 * np.sin(t * 0.12) - 6.0 * np.abs(np.sin(t * 0.28 + 0.3))
    speed_kmh = np.clip(speed_kmh + rng.normal(0, 2.5, n), 40, 190)
    speed_ms = speed_kmh / MS_TO_KMH

    df = pd.DataFrame({
        "time": t,
        "speed": speed_ms,  # m/s
        "acc_x": acc_x,     # m/s^2
        "acc_y": acc_y,     # m/s^2
    })
    return df


# ----------------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------------
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw))
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    elif name.endswith(".log"):
        text = raw.decode("utf-8", errors="ignore")
        for sep in [",", ";", r"\s+"]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if df.shape[1] > 1:
                    break
            except Exception:
                continue
        else:
            raise ValueError("Failed to parse .LOG file")
    else:
        raise ValueError("Unsupported file format")

    return df


def guess_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


# ----------------------------------------------------------------------------
# Signal filters & processing
# ----------------------------------------------------------------------------
def moving_average(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def butterworth(series, cutoff, fs, order=3):
    nyq = 0.5 * fs
    norm_cutoff = min(cutoff / nyq, 0.99)
    if norm_cutoff <= 0:
        return series
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(series) <= padlen:
        return series
    return pd.Series(filtfilt(b, a, series.values), index=series.index)


def compute_envelope(x, y, n_bins=72, percentile=98):
    """Grip envelope: P-th percentile of radius in each angular bin."""
    angles = np.arctan2(y, x)
    radii = np.sqrt(x ** 2 + y ** 2)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(angles, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    env_r = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() >= 3:
            env_r[i] = np.percentile(radii[mask], percentile)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid = ~np.isnan(env_r)
    if valid.sum() < 3:
        return None, None
    if not valid.all():
        env_r[~valid] = np.interp(
            bin_centers[~valid], bin_centers[valid], env_r[valid],
            period=2 * np.pi,
        )

    env_x = env_r * np.cos(bin_centers)
    env_y = env_r * np.sin(bin_centers)
    env_x = np.append(env_x, env_x[0])
    env_y = np.append(env_y, env_y[0])
    return env_x, env_y


# ----------------------------------------------------------------------------
# Sidebar - Input Data
# ----------------------------------------------------------------------------
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader(
    "Log file (CSV / PARQUET / LOG)", type=["csv", "parquet", "log"]
)

use_synthetic = uploaded_file is None
if use_synthetic:
    df_raw = generate_synthetic_data()
    col_time, col_speed, col_accx, col_accy = "time", "speed", "acc_x", "acc_y"
else:
    try:
        df_raw = load_file(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

    st.sidebar.subheader("Column Mapping")
    cols = list(df_raw.columns)

    def idx_of(guess):
        return cols.index(guess) if guess in cols else 0

    g_time = guess_column(df_raw, ["time", "time_s", "t", "czas"])
    g_speed = guess_column(df_raw, ["speed", "speed_ms", "vcar", "velocity", "predkosc"])
    g_accx = guess_column(df_raw, ["acc_x", "accx", "ax", "long_acc", "gforcelon"])
    g_accy = guess_column(df_raw, ["acc_y", "accy", "ay", "lat_acc", "gforcelat"])

    col_time = st.sidebar.selectbox("Time", cols, index=idx_of(g_time) if g_time else 0)
    col_speed = st.sidebar.selectbox("Speed [m/s]", cols, index=idx_of(g_speed) if g_speed else 0)
    col_accx = st.sidebar.selectbox("Longitudinal Acceleration (X)", cols, index=idx_of(g_accx) if g_accx else 0)
    col_accy = st.sidebar.selectbox("Lateral Acceleration (Y)", cols, index=idx_of(g_accy) if g_accy else 0)

st.sidebar.header("Data Range")
t_min, t_max = float(df_raw[col_time].min()), float(df_raw[col_time].max())
time_range = st.sidebar.slider(
    "Time Range [s]", min_value=t_min, max_value=t_max, value=(t_min, t_max),
)

st.sidebar.header("Signal Filtering")
smooth_method = st.sidebar.radio("Smoothing Method", ["None", "Moving Average", "Butterworth"])
if smooth_method == "Moving Average":
    smooth_window = st.sidebar.slider("Window [samples]", 1, 51, 5, step=2)
elif smooth_method == "Butterworth":
    smooth_cutoff = st.sidebar.slider("Cutoff Frequency [Hz]", 0.1, 20.0, 5.0)

st.sidebar.header("Units & Noise")
convert_to_g = st.sidebar.checkbox("Convert m/s² to G (÷9.81)", value=True)
exclude_noise = st.sidebar.checkbox("Exclude noise around (0,0)", value=True)
noise_threshold = st.sidebar.slider(
    "Noise Exclusion Radius", 0.0, 1.0, 0.08, step=0.01, disabled=not exclude_noise,
)

st.sidebar.header("Speed Bins")
speed_bin_options = ["Low (<80 km/h)", "Medium (80-130 km/h)", "High (>130 km/h)"]
selected_speed_bins = st.sidebar.multiselect(
    "Include Speed Ranges", speed_bin_options, default=speed_bin_options,
)

st.sidebar.header("G-G Visualization")
color_mode = st.sidebar.selectbox(
    "Color Points By",
    ["Resultant |G|", "Speed", "Time"],
)
show_envelope = st.sidebar.checkbox("Draw Grip Envelope (98th Percentile)", value=True)
show_circle = st.sidebar.checkbox("Draw Reference Circle (99.5th Percentile)", value=False)

# ----------------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------------
df = df_raw.copy()
df = df[(df[col_time] >= time_range[0]) & (df[col_time] <= time_range[1])].reset_index(drop=True)

if df.empty:
    st.warning("No data in selected range.")
    st.stop()

dt = np.median(np.diff(df[col_time].values)) if len(df) > 1 else 0.01
fs = 1.0 / dt if dt > 0 else 100.0

acc_x = df[col_accx].astype(float).copy()
acc_y = df[col_accy].astype(float).copy()

if smooth_method == "Moving Average":
    acc_x = moving_average(acc_x, smooth_window)
    acc_y = moving_average(acc_y, smooth_window)
elif smooth_method == "Butterworth":
    acc_x = butterworth(acc_x, smooth_cutoff, fs)
    acc_y = butterworth(acc_y, smooth_cutoff, fs)

if convert_to_g:
    acc_x = acc_x / 9.81
    acc_y = acc_y / 9.81
    unit_label = "G"
else:
    unit_label = "m/s2"

df["acc_x_proc"] = acc_x
df["acc_y_proc"] = acc_y
df["magnitude"] = np.sqrt(df["acc_x_proc"] ** 2 + df["acc_y_proc"] ** 2)
df["speed_kmh"] = df[col_speed].astype(float) * MS_TO_KMH

if exclude_noise:
    df = df[df["magnitude"] > noise_threshold].reset_index(drop=True)

if df.empty:
    st.warning("All points were filtered out as noise. Lower the exclusion threshold.")
    st.stop()

bin_masks = []
if "Low (<80 km/h)" in selected_speed_bins:
    bin_masks.append(df["speed_kmh"] < 80)
if "Medium (80-130 km/h)" in selected_speed_bins:
    bin_masks.append((df["speed_kmh"] >= 80) & (df["speed_kmh"] <= 130))
if "High (>130 km/h)" in selected_speed_bins:
    bin_masks.append(df["speed_kmh"] > 130)

if bin_masks:
    combined_mask = np.logical_or.reduce(bin_masks)
    df_plot = df[combined_mask].reset_index(drop=True)
else:
    df_plot = df.iloc[0:0]

if df_plot.empty:
    st.warning("No data points found within selected speed ranges.")
    st.stop()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("G-G Diagram / Traction Circle Analyzer")
st.caption(
    f"Points: {len(df_plot)} / {len(df)} | Time Range: {time_range[0]:.1f}-{time_range[1]:.1f} s "
    f"| Smoothing: {smooth_method} | Units: {unit_label}"
)

col_left, col_right = st.columns([3, 1])

# ----------------------------------------------------------------------------
# G-G Plot
# ----------------------------------------------------------------------------
with col_left:
    fig = go.Figure()

    if color_mode == "Resultant |G|":
        color_values = df_plot["magnitude"]
        colorscale = "Viridis"
        colorbar_title = f"|G| [{unit_label}]"
    elif color_mode == "Speed":
        color_values = df_plot["speed_kmh"]
        colorscale = "Plasma"
        colorbar_title = "Speed [km/h]"
    else:
        color_values = df_plot[col_time]
        colorscale = "Cividis"
        colorbar_title = "Time [s]"

    customdata = np.stack([
        df_plot[col_time].values,
        df_plot["speed_kmh"].values,
        df_plot["magnitude"].values,
    ], axis=-1)

    fig.add_trace(go.Scatter(
        x=df_plot["acc_y_proc"],
        y=df_plot["acc_x_proc"],
        mode="markers",
        marker=dict(
            size=6,
            color=color_values,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            showscale=True,
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=(
            "Lateral: %{x:.3f} " + unit_label + "<br>"
            "Longitudinal: %{y:.3f} " + unit_label + "<br>"
            "Time: %{customdata[0]:.2f} s<br>"
            "Speed: %{customdata[1]:.1f} km/h<br>"
            "|G|: %{customdata[2]:.3f} " + unit_label + "<extra></extra>"
        ),
        name="Data Points",
    ))

    max_g = float(np.percentile(df_plot["magnitude"], 99.5))

    if show_envelope:
        env_x, env_y = compute_envelope(df_plot["acc_y_proc"].values, df_plot["acc_x_proc"].values)
        if env_x is not None:
            fig.add_trace(go.Scatter(
                x=env_x, y=env_y, mode="lines",
                line=dict(color="firebrick", width=2),
                name="Grip Envelope (P98)",
            ))
            max_g = max(max_g, float(np.nanmax(np.sqrt(env_x ** 2 + env_y ** 2))))

    if show_circle:
        theta = np.linspace(0, 2 * np.pi, 200)
        ref_r = float(np.percentile(df_plot["magnitude"], 99.5))
        fig.add_trace(go.Scatter(
            x=ref_r * np.cos(theta), y=ref_r * np.sin(theta),
            mode="lines", line=dict(color="gray", width=1, dash="dot"),
            name=f"Reference Circle ({ref_r:.2f} {unit_label})",
        ))

    axis_lim = max_g * 1.25 if max_g > 0 else 1.0
    fig.add_shape(type="line", x0=-axis_lim, x1=axis_lim, y0=0, y1=0, line=dict(color="lightgray", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-axis_lim, y1=axis_lim, line=dict(color="lightgray", width=1))

    fig.update_layout(
        xaxis=dict(title=f"Lateral [{unit_label}]", range=[-axis_lim, axis_lim],
                    zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title=f"Longitudinal [{unit_label}]", range=[-axis_lim, axis_lim], zeroline=False),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        margin=dict(t=60, b=40, l=60, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Speed profile over time
    speed_fig = go.Figure()
    speed_fig.add_trace(go.Scatter(
        x=df_plot[col_time], y=df_plot["speed_kmh"],
        mode="lines", line=dict(color="steelblue", width=1.5),
        name="Speed",
    ))
    speed_fig.update_layout(
        xaxis=dict(title="Time [s]"),
        yaxis=dict(title="Speed [km/h]"),
        height=250,
        template="plotly_white",
        margin=dict(t=20, b=40, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(speed_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
with col_right:
    st.subheader("Statistics")

    max_braking = df_plot.loc[df_plot["acc_x_proc"].idxmin(), "acc_x_proc"]
    max_accel = df_plot.loc[df_plot["acc_x_proc"].idxmax(), "acc_x_proc"]
    max_lat_right = df_plot["acc_y_proc"].max()
    max_lat_left = df_plot["acc_y_proc"].min()
    avg_magnitude = df_plot["magnitude"].mean()
    max_speed = df_plot["speed_kmh"].max()

    st.metric("Max Braking G", f"{max_braking:.2f} {unit_label}")
    st.metric("Max Acceleration G", f"{max_accel:.2f} {unit_label}")
    st.metric("Max Right G", f"{max_lat_right:.2f} {unit_label}")
    st.metric("Max Left G", f"{max_lat_left:.2f} {unit_label}")
    st.metric("Mean Resultant |G|", f"{avg_magnitude:.2f} {unit_label}")
    st.metric("Top Speed", f"{max_speed:.1f} km/h")
    st.metric("Filtered Points", f"{len(df_plot)} / {len(df)}")

    st.divider()
    st.subheader("Speed Distribution")
    for label, mask in zip(
        speed_bin_options,
        [df_plot["speed_kmh"] < 80,
         (df_plot["speed_kmh"] >= 80) & (df_plot["speed_kmh"] <= 130),
         df_plot["speed_kmh"] > 130],
    ):
        st.text(f"{label}: {int(mask.sum())} pts")

    with st.expander("Processed Data"):
        st.dataframe(
            df_plot[[col_time, "speed_kmh", col_accx, col_accy, "acc_x_proc", "acc_y_proc", "magnitude"]]
            .rename(columns={col_accx: "acc_x_raw", col_accy: "acc_y_raw"})
            .head(200)
        )
