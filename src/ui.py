# src/ui.py
"""
Football Analytics Dashboard – Streamlit UI (final)

Works with your repo structure (no fa_dash/).
Fixes:
- Replaces deprecated `use_container_width` with `width="stretch"`
- Avoids Plotly list-style line widths (uses scalar + binning)
- Adds caching for data/model
- Robust filters + friendly empty states
- Shareable state via query params

Expected (gracefully optional) files:
- data/processed_shots.csv
- data/passing_data.csv
- models/xgb_xg_calibrated.joblib (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib  # optional
except Exception:
    joblib = None  # model handling becomes optional

# ---------------- Paths (relative to project root) ----------------
# streamlit run from repo root keeps this correct
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SHOTS_CSV = DATA_DIR / "processed_shots.csv"
PASSING_CSV = DATA_DIR / "passing_data.csv"
CAL_MODEL = MODEL_DIR / "xgb_xg_calibrated.joblib"


# ---------------- Caching ----------------
@st.cache_data(ttl=3600)
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_shots() -> pd.DataFrame:
    return _read_csv(SHOTS_CSV)


@st.cache_data(ttl=3600)
def load_passing() -> pd.DataFrame:
    return _read_csv(PASSING_CSV)


@st.cache_resource
def load_model(path: Path = CAL_MODEL):
    if joblib is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# ---------------- Small utils ----------------
def _safe_unique(series: pd.Series) -> Iterable:
    try:
        values = [
            v
            for v in series.dropna().unique().tolist()
            if str(v).strip() != ""
        ]
        return sorted(values)
    except Exception:
        return []


def _apply_common_filters(
    df: pd.DataFrame,
    season: Optional[str],
    competition: Optional[str],
    team: Optional[str],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if season and "season" in out.columns:
        out = out[out["season"].astype(str) == str(season)]
    if competition and "competition" in out.columns:
        out = out[out["competition"].astype(str) == str(competition)]
    if team:
        # prefer 'team', fallback to 'possession_team'
        team_cols = [
            c for c in out.columns if c.lower() in {"team", "possession_team"}
        ]
        if team_cols:
            out = out[out[team_cols[0]].astype(str) == str(team)]
    if date_range and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        start, end = date_range
        mask = (
            (out["date"] >= pd.to_datetime(start)) &
            (out["date"] <= pd.to_datetime(end))
        )
        out = out[mask]
    return out


def _metric(value, default="–"):
    if (
        value is None or
        (isinstance(value, float) and (np.isnan(value) or np.isinf(value)))
    ):
        return default
    return value


# ---------------- Plot helpers ----------------
def make_pitch(width: int = 120, height: int = 80) -> go.Figure:
    """Minimalist horizontal pitch."""
    fig = go.Figure()
    # Outer
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=width,
        y1=height,
        line=dict(color="black", width=1),
        fillcolor="white",
    )
    # Midline
    fig.add_shape(
        type="line",
        x0=width / 2,
        y0=0,
        x1=width / 2,
        y1=height,
        line=dict(color="black", width=1),
    )
    # Center circle (r ~ 9.15m)
    fig.add_shape(
        type="circle",
        x0=width / 2 - 9.15,
        y0=height / 2 - 9.15,
        x1=width / 2 + 9.15,
        y1=height / 2 + 9.15,
        line=dict(color="black", width=1),
    )

    fig.update_xaxes(
        range=[-2, width + 2], showgrid=False, zeroline=False, visible=False
    )
    fig.update_yaxes(
        range=[-2, height + 2],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(align="left"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    return fig


def plot_shotmap(
    shots: pd.DataFrame,
    title: Optional[str] = None
) -> go.Figure:
    fig = make_pitch()
    if shots.empty:
        fig.update_layout(title=title or "Shot Map (no data)")
        return fig

    is_goal = shots.get("is_goal")
    if is_goal is None:
        is_goal = pd.Series(False, index=shots.index)

    goals = (
        shots[is_goal == 1]
        if np.issubdtype(is_goal.dtype, np.number)
        else shots[is_goal.astype(bool)]
    )
    non_goals = shots.drop(goals.index, errors="ignore")

    def _customdata(df: pd.DataFrame) -> np.ndarray | None:
        cols = []
        if "player" in df.columns:
            cols.append(df["player"].astype(str))
        if "xg" in df.columns:
            cols.append(df["xg"].astype(float))
        if "minute" in df.columns:
            cols.append(df["minute"])
        if not cols:
            return None
        return np.vstack(cols).T

    hover = []
    if "player" in shots.columns:
        hover.append("Player: %{customdata[0]}")
    if "xg" in shots.columns:
        hover.append("xG: %{customdata[1]:.2f}")
    if "minute" in shots.columns:
        hover.append("Min: %{customdata[2]}")
    hover_tmpl = "<br>" + "<br>".join(hover) if hover else ""

    if not non_goals.empty:
        fig.add_trace(
            go.Scatter(
                x=non_goals.get("x", []),
                y=non_goals.get("y", []),
                mode="markers",
                name="Shots",
                marker=dict(size=8, opacity=0.65, symbol="circle"),
                customdata=_customdata(non_goals),
                hovertemplate="%{x}, %{y}" + hover_tmpl,
            )
        )
    if not goals.empty:
        fig.add_trace(
            go.Scatter(
                x=goals.get("x", []),
                y=goals.get("y", []),
                mode="markers",
                name="Goals",
                marker=dict(size=11, opacity=0.95, symbol="star"),
                customdata=_customdata(goals),
                hovertemplate="%{x}, %{y}" + hover_tmpl,
            )
        )

    fig.update_layout(title=title or "Shot Map")
    return fig


def plot_passing_network(
    df: pd.DataFrame,
    title: Optional[str] = None
) -> go.Figure:
    """
    Passing network with scalar line widths (no lists).
    Columns expected at minimum: passer, receiver, passes
    Optional: passer_x, passer_y, receiver_x, receiver_y
    """
    fig = make_pitch()
    if df.empty or not {"passer", "receiver", "passes"}.issubset(df.columns):
        fig.update_layout(title=title or "Passing Network (no data)")
        return fig

    players = sorted(set(df["passer"]).union(set(df["receiver"])))
    pos: Dict[str, tuple[float, float]] = {}

    has_coords = {"passer_x", "passer_y", "receiver_x", "receiver_y"}.issubset(
        df.columns
    )
    if has_coords:
        px = (
            df.groupby("passer")[["passer_x", "passer_y"]]
            .mean(numeric_only=True)
            .rename(columns={"passer_x": "x", "passer_y": "y"})
        )
        rx = (
            df.groupby("receiver")[["receiver_x", "receiver_y"]]
            .mean(numeric_only=True)
            .rename(columns={"receiver_x": "x", "receiver_y": "y"})
        )
        avg = pd.concat([px, rx]).groupby(level=0).mean(numeric_only=True)
        for p in players:
            x = (
                float(avg.loc[p, "x"])
                if p in avg.index and pd.notna(avg.loc[p, "x"])
                else 60.0
            )
            y = (
                float(avg.loc[p, "y"])
                if p in avg.index and pd.notna(avg.loc[p, "y"])
                else 40.0
            )
            pos[p] = (x, y)
    else:
        # Circle fallback
        cx, cy, r = 60.0, 40.0, 28.0
        angles = np.linspace(0, 2 * np.pi, num=len(players), endpoint=False)
        for p, a in zip(players, angles):
            pos[p] = (cx + r * np.cos(a), cy + r * np.sin(a))

    # Node size by involvement
    touches = (
        df.groupby("passer")["passes"].sum()
        .add(df.groupby("receiver")["passes"].sum(), fill_value=0)
    )
    max_t = max(float(touches.max()), 1.0)
    node_sizes = {
        p: 12 + 18 * float(touches.get(p, 0.0)) / max_t
        for p in players
    }

    # Edge binning for scalar widths
    bins = [(0, 4, 1), (4, 8, 2), (8, 1e9, 3)]
    for low, high, lw in bins:
        sub = df[(df["passes"] >= low) & (df["passes"] < high)]
        if sub.empty:
            continue
        xs, ys = [], []
        for _, row in sub.iterrows():
            a, b = str(row["passer"]), str(row["receiver"])
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            xs += [x1, x2, None]
            ys += [y1, y2, None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=lw),
                opacity=0.35 + 0.2 * (lw - 1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Nodes
    fig.add_trace(
        go.Scatter(
            x=[pos[p][0] for p in players],
            y=[pos[p][1] for p in players],
            mode="markers+text",
            text=players,
            textposition="top center",
            marker=dict(size=[node_sizes[p] for p in players], opacity=0.9),
            name="Players",
            hovertemplate="Player: %{text}<extra></extra>",
        )
    )

    fig.update_layout(title=title or "Passing Network")
    return fig


# ---------------- Pages ----------------
def page_match_center(shots: pd.DataFrame):
    st.subheader("Match Center")
    if shots.empty:
        st.info("No shots data available for the selected filters.")
        return
    cols = [
        c for c in [
            "date", "competition", "season", "team", "opponent",
            "player", "minute", "xg", "is_goal"
        ]
        if c in shots.columns
    ]
    st.dataframe(
        shots[cols].sort_values(
            [c for c in ["date", "minute"] if c in cols],
            ascending=[True, True],
        ),
        use_container_width=True,
    )


def page_xg_view(shots: pd.DataFrame):
    st.subheader("xG & Shot Quality")
    if shots.empty:
        st.info("No shots data available for the selected filters.")
        return

    total_xg = (
        float(shots.get("xg", pd.Series(dtype=float)).sum())
        if "xg" in shots.columns
        else np.nan
    )
    goals = (
        int(shots.get("is_goal", pd.Series(dtype=int)).sum())
        if "is_goal" in shots.columns
        else 0
    )
    n_shots = int(len(shots))
    xg_ps = (total_xg / n_shots) if n_shots > 0 else np.nan
    xg_diff = (goals - total_xg) if not np.isnan(total_xg) else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total xG", f"{_metric(round(total_xg, 2))}")
    k2.metric("xG/Shot", f"{_metric(round(xg_ps, 2))}")
    k3.metric("Goals", f"{goals}")
    k4.metric("Shots", f"{n_shots}")
    k5.metric("xG Diff (G - xG)", f"{_metric(round(xg_diff, 2))}")

    fig = plot_shotmap(shots, title="Shot Map")
    st.plotly_chart(fig, width="stretch")


def page_passing_view(passes: pd.DataFrame):
    st.subheader("Passing Network")
    if passes.empty:
        st.info("No passing data available for the selected filters.")
        return

    fig = plot_passing_network(passes, title="Team Passing Network")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Underlying pass edges"):
        cols = [
            c for c in ["passer", "receiver", "passes", "team", "match_id"]
            if c in passes.columns
        ]
        st.dataframe(
            passes[cols].sort_values("passes", ascending=False),
            use_container_width=True,
        )


# ---------------- Sidebar / Query params ----------------
def _init_query_state():
    qp = st.experimental_get_query_params()
    return {k: v[0] for k, v in qp.items() if isinstance(v, list) and v}


def _set_query_state(**kwargs):
    st.experimental_set_query_params(
        **{k: v for k, v in kwargs.items() if v is not None}
    )


def _filters_panel(shots: pd.DataFrame, passes: pd.DataFrame):
    with st.sidebar:
        st.header("Filters")

        seasons = (
            _safe_unique(shots.get("season", pd.Series(dtype=str))) or
            _safe_unique(passes.get("season", pd.Series(dtype=str)))
        )
        competitions = (
            _safe_unique(shots.get("competition", pd.Series(dtype=str))) or
            _safe_unique(passes.get("competition", pd.Series(dtype=str)))
        )
        teams = sorted(
            set(_safe_unique(shots.get("team", pd.Series(dtype=str)))) |
            set(_safe_unique(passes.get("team", pd.Series(dtype=str))))
        )

        q = _init_query_state()
        season = st.selectbox(
            "Season",
            seasons,
            index=(
                seasons.index(q.get("season"))
                if q.get("season") in seasons
                else (len(seasons) - 1 if seasons else 0)
            )
            if seasons
            else None,
        )
        comp = st.selectbox(
            "Competition",
            competitions,
            index=(
                competitions.index(q.get("competition"))
                if q.get("competition") in competitions
                else 0
            )
            if competitions
            else None,
        )
        team = st.selectbox(
            "Team",
            teams,
            index=(
                teams.index(q.get("team"))
                if q.get("team") in teams
                else 0
            ) if teams else None,
        )

        # Date range from shots
        min_date = (
            pd.to_datetime(shots["date"]).min()
            if "date" in shots.columns and not shots.empty
            else None
        )
        max_date = (
            pd.to_datetime(shots["date"]).max()
            if "date" in shots.columns and not shots.empty
            else None
        )
        date_range = None
        if min_date is not None and max_date is not None:
            date_range = st.date_input(
                "Date range", value=(min_date.date(), max_date.date())
            )

        st.button("Apply", type="primary", use_container_width=True)
        _set_query_state(
            season=str(season) if season else None,
            competition=str(comp) if comp else None,
            team=str(team) if team else None,
        )
    return season, comp, team, date_range


# ---------------- Entry ----------------
def run():
    st.set_page_config(
        page_title="Football Analytics Dashboard",
        layout="wide"
    )
    st.title("Football Analytics Dashboard")
    st.caption("xG model • Passing networks • Match insights")

    shots_df = load_shots()
    passes_df = load_passing()
    _ = load_model()  # optional for future calibrated scoring in UI

    season, comp, team, date_range = _filters_panel(shots_df, passes_df)

    shots_f = _apply_common_filters(shots_df, season, comp, team, date_range)
    passes_f = _apply_common_filters(passes_df, season, comp, team, date_range)

    tab1, tab2, tab3 = st.tabs([
        "Match Center",
        "xG & Shot Quality",
        "Passing Network"
    ])
    with tab1:
        page_match_center(shots_f)
    with tab2:
        page_xg_view(shots_f)
    with tab3:
        page_passing_view(passes_f)

    statuses = []
    for name, exists in [
        ("processed_shots.csv", SHOTS_CSV.exists()),
        ("passing_data.csv", PASSING_CSV.exists()),
        ("xgb_xg_calibrated.joblib", CAL_MODEL.exists()),
    ]:
        statuses.append(f"{name}: {'present' if exists else 'missing'}")

    st.caption("Data paths — " + "  •  ".join(statuses))


if __name__ == "__main__":
    run()
