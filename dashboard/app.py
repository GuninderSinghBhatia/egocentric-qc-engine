# QC dashboard: reads the tables written by `python -m egoqc check <dataset>`
#
#   streamlit run dashboard/app.py
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))  # so the dashboard can reuse egoqc helpers
from egoqc.quiet import quiet_stderr  # noqa: E402

# Check name -> (title, what a failure means in plain words)
CHECKS = {
    "spec": ("Spec", "resolution or frame rate below the delivery spec"),
    "corrupt": ("Corrupt file", "part of the file cannot be decoded"),
    "frame_drops": ("Frame drops", "frames missing from the recording"),
    "frozen": ("Frozen video", "the picture stops changing"),
    "blur": ("Blur", "frames too soft to use"),
    "brightness": ("Exposure", "frames too dark or blown out"),
    "hands": ("Hands in view", "too few frames show a hand"),
}
# Timelines: score column -> (title, how to read it)
TIMELINES = {
    "blur": ("Sharpness (blur score)", "higher is sharper"),
    "brightness": ("Brightness (mean gray, 0-255)", "must stay inside the band"),
    "frozen": ("Motion (difference to next frame)", "near 0 means frozen"),
    "hands": ("Hands (hands found by MediaPipe)", "0 means no hand in view"),
}

SERIES = "#2a78d6"  # one blue for every single-series chart
GOOD, CRITICAL = "#0ca30c", "#d03b3b"  # status colours, always shown with an icon and a word
MUTED = "#8a8985"

st.set_page_config(page_title="Egocentric QC", page_icon="🎥", layout="wide")


# --- data -----------------------------------------------------------------------------------

@st.cache_data
def load_report(name: str):
    report = pd.read_csv(OUTPUTS / f"report_{name}.csv")
    scores_path = OUTPUTS / f"scores_{name}.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    return report, scores


@st.cache_data
def load_thresholds():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["checks"]


@st.cache_data
def episode_labels(clip_paths: tuple[str, ...]) -> dict[str, str]:
    """Clip file name -> task label from the dataset's meta/episodes.jsonl, if there is one."""
    labels = {}
    roots = set()
    for p in clip_paths:
        parts = Path(p).parts
        if "videos" in parts:
            roots.add(Path(*parts[: parts.index("videos")]))
    for root in roots:
        meta = (ROOT / root / "meta" / "episodes.jsonl") if not root.is_absolute() else root / "meta" / "episodes.jsonl"
        if not meta.exists():
            continue
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = json.loads(line)
                tasks = ep.get("tasks") or [""]
                labels[f"episode_{ep['episode_index']:06d}.mp4"] = tasks[0]
    return labels


def video_exists(path: str) -> bool:
    full = Path(path.replace("\\", "/"))
    return (full if full.is_absolute() else ROOT / full).exists()


@st.cache_data
def frame_at(path: str, seconds: float, width: int = 720):
    """The video frame at `seconds`, as RGB, or None if it cannot be decoded."""
    full = Path(path.replace("\\", "/"))  # reports written on Windows use backslashes
    full = full if full.is_absolute() else ROOT / full
    with quiet_stderr():  # damaged clips make FFmpeg print hundreds of lines
        cap = cv2.VideoCapture(str(full))
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


reports = sorted(p.stem.removeprefix("report_") for p in OUTPUTS.glob("report_*.csv"))
if not reports:
    st.error("No reports yet. Run `python -m egoqc check <dataset>` first.")
    st.stop()

# --- sidebar -------------------------------------------------------------------------------

with st.sidebar:
    st.header("Dataset")
    default = reports.index("demo") if "demo" in reports else 0
    dataset = st.selectbox("Report", reports, index=default, label_visibility="collapsed")
    thresholds = load_thresholds()
    with st.expander("Thresholds (config.yaml)"):
        st.json(thresholds, expanded=False)
    st.caption("Generate a report: `python -m egoqc check data/<dataset>`")

report, scores = load_report(dataset)
labels = episode_labels(tuple(report["path"].astype(str))) if "path" in report else {}
report["label"] = report["clip"].map(labels).astype(object).fillna("")
# Only the checks this report contains (older reports were made before some checks existed)
CHECKS = {c: v for c, v in CHECKS.items() if f"{c}_passed" in report}

# --- headline ------------------------------------------------------------------------------

st.title("Egocentric video QC")
st.caption(f"Dataset **{dataset}** · {len(CHECKS)} automatic checks per clip · one decoding pass per clip")

n = len(report)
passed = int(report["passed"].sum())
minutes = report["duration_s"].fillna(0).sum() / 60 if "duration_s" in report else float("nan")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clips checked", n)
k2.metric("Passed", f"{passed}  ({passed / n:.0%})")
k3.metric("Failed", f"{n - passed}  ({(n - passed) / n:.0%})")
k4.metric("Footage checked", f"{minutes:.1f} min")

# --- failures by check + clip table ----------------------------------------------------------

left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("Failures by check")
    counts = [int((~report[f"{c}_passed"]).sum()) for c in CHECKS]
    titles = [CHECKS[c][0] for c in CHECKS]
    fig = go.Figure(go.Bar(
        x=counts, y=titles, orientation="h", marker_color=SERIES,
        text=counts, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} clips failed<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=30, t=10, b=0), bargap=0.45,
        xaxis=dict(title="clips failed", showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                   rangemode="tozero", dtick=1 if max(counts, default=0) <= 10 else None),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit")

with right:
    st.subheader("Clips")
    table = pd.DataFrame({
        "Clip": report["clip"],
        "What it is": report["label"],
        "Verdict": report["passed"].map({True: "✅ PASS", False: "❌ FAIL"}),
        "Failed checks": report["failed_checks"].astype(object).fillna("").map(
            lambda s: ", ".join(CHECKS.get(c.strip(), (c.strip(),))[0] for c in s.split(",") if c.strip())),
        "Length (s)": report.get("duration_s"),
    })
    if not table["What it is"].any():
        table = table.drop(columns="What it is")
    st.dataframe(table, hide_index=True, width="stretch", height=300)

# --- clip detail -----------------------------------------------------------------------------

st.divider()
st.subheader("Clip detail")
failing = report[~report["passed"]]["clip"].tolist()
options = report["clip"].tolist()
start = options.index(failing[0]) if failing else 0
clip_name = st.selectbox(
    "Clip", options, index=start,
    format_func=lambda c: f"{c} — {labels[c]}" if labels.get(c) else c,
)
row = report[report["clip"] == clip_name].iloc[0]

cols = st.columns(len(CHECKS))
for col, (check, (title, meaning)) in zip(cols, CHECKS.items()):
    ok = bool(row[f"{check}_passed"])
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(
                f"<span style='color:{GOOD if ok else CRITICAL};font-weight:600'>"
                f"{'✅ pass' if ok else '❌ fail'}</span>",
                unsafe_allow_html=True,
            )
            st.caption(str(row[f"{check}_note"]))

# Worst moment: the start of the first failing check's worst stretch
failed_here = [c for c in CHECKS if not row[f"{c}_passed"]]
img_col, chart_col = st.columns([2, 3], gap="large")
with img_col:
    st.markdown("**What the problem looks like**")
    moment = None
    for check in failed_here:
        start_s = row.get(f"{check}_start_s")
        if pd.notna(start_s):
            moment = (check, float(start_s))
            break
    if moment is None:
        frame = frame_at(str(row["path"]), 1.0) if "path" in row else None
        if frame is not None:
            st.image(frame, caption="Frame at 1 s (no problem found)", width="stretch")
        else:
            st.info("No frame to show.")
    else:
        check, t = moment
        frame = frame_at(str(row["path"]), t + 0.5) if "path" in row else None
        if "path" not in row:
            st.info("This report has no file paths; re-run `python -m egoqc check` to see frames.")
        elif not video_exists(str(row["path"])):
            st.info("The video is not on this machine (data/ is not in the repo), so no frame to show.")
        elif frame is not None:
            st.image(frame, caption=f"{CHECKS[check][0]}: frame at {t + 0.5:.1f} s, where the problem starts",
                     width="stretch")
        else:
            st.error(f"{CHECKS[check][0]}: the frame at {t:.0f} s cannot be decoded at all.")

with chart_col:
    st.markdown("**Per-second scores**")
    clip_scores = scores[scores["clip"] == clip_name] if not scores.empty else pd.DataFrame()
    if clip_scores.empty:
        st.info("No per-second scores for this clip (it could not be read).")
    else:
        tabs = st.tabs([TIMELINES[k][0].split(" (")[0] for k in TIMELINES])
        for tab, (column, (title, hint)) in zip(tabs, TIMELINES.items()):
            with tab:
                if column not in clip_scores:
                    st.info("No scores for this check in this report; re-run `python -m egoqc check`.")
                    continue
                data = clip_scores[["time", column]].dropna()
                t = thresholds[column]
                if column == "blur":
                    bad = data[column] < t["score_threshold"]
                    lines = [(t["score_threshold"], "threshold")]
                elif column == "brightness":
                    bad = (data[column] < t["minimum_brightness"]) | (data[column] > t["maximum_brightness"])
                    lines = [(t["minimum_brightness"], "min"), (t["maximum_brightness"], "max")]
                elif column == "frozen":
                    bad = data[column] < t["difference_threshold"]
                    lines = [(t["difference_threshold"], "threshold")]
                else:  # hands: a sample is bad when no hand is found
                    bad = data[column] == 0
                    lines = []

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data["time"], y=data[column], mode="lines", name=title.split(" (")[0],
                    line=dict(color=SERIES, width=2),
                    hovertemplate="%{x:.0f} s: %{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=data["time"][bad], y=data[column][bad], mode="markers", name="fails the check",
                    marker=dict(color=CRITICAL, size=9, symbol="x"),
                    hovertemplate="%{x:.0f} s: %{y:.2f} (fails)<extra></extra>",
                ))
                for value, name in lines:
                    fig.add_hline(y=value, line=dict(color=MUTED, dash="dash", width=1),
                                  annotation_text=f"{name} {value}", annotation_position="top left",
                                  annotation_font_color=MUTED)
                fig.update_layout(
                    height=300, margin=dict(l=0, r=10, t=10, b=0), hovermode="x unified",
                    legend=dict(orientation="h", y=-0.25),
                    xaxis=dict(title="time in clip (s)", gridcolor="rgba(128,128,128,0.15)"),
                    yaxis=dict(title=None, gridcolor="rgba(128,128,128,0.15)", rangemode="tozero"),
                )
                st.plotly_chart(fig, width="stretch", theme="streamlit", key=f"{clip_name}-{column}")
                st.caption(hint)

with st.expander("Full results table"):
    st.dataframe(report, hide_index=True, width="stretch")
