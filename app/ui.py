"""Shared Streamlit presentation helpers for the four website pages.

The helpers here keep the visual language consistent across pages and avoid the
deprecated legacy full-width argument on modern Streamlit versions while still
working with the project's ``streamlit>=1.30`` floor.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import streamlit as st

from app.data import (
    HONESTY_NOTE,
    ReplayRun,
    compute_benchmark,
    discover_replays,
    experiment_manifest,
    figure_catalog,
    load_bootstrap,
    load_tables,
)

T = TypeVar("T")

DEPLOYABLE_KINDS = frozenset({"observable_baseline", "learned_observable"})
IDEALIZED_POLICIES = frozenset({"allslow", "allfast"})

# Inline CSS keeps the dashboard self-contained: no theme plugin or CDN needed.
APP_CSS = """
<style>
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 340px);
  }
  [data-testid="stSidebar"] {
    background: #101c2c;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
  [data-testid="stSidebar"] .stCaption {
    color: #d9e4ef;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #ffffff;
  }
  .ct-hero {
    border-radius: 18px;
    padding: 28px 32px 24px;
    background: linear-gradient(120deg, #102b45 0%, #1c5680 62%, #2e7ba6 100%);
    color: #f6fbff;
    box-shadow: 0 14px 34px rgba(16, 43, 69, 0.22);
    margin-bottom: 22px;
  }
  .ct-hero h1 { color: #ffffff; font-size: 2.25rem; margin: 0 0 6px; }
  .ct-hero p { color: #d7e9f7; margin: 0; font-size: 1.02rem; max-width: 78ch; }
  .ct-eyebrow {
    display: inline-block; font-size: 0.72rem; letter-spacing: 0.16em;
    text-transform: uppercase; font-weight: 700; color: #9fd1f0;
    margin-bottom: 10px;
  }
  .ct-pills { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
  .ct-pill {
    border: 1px solid rgba(255,255,255,0.32); border-radius: 999px;
    padding: 4px 12px; font-size: 0.78rem; color: #eef7ff;
    background: rgba(255,255,255,0.08);
  }
  .ct-section {
    margin: 26px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid #dce6f0;
  }
  .ct-section h2 { margin: 0; color: #14324d; font-size: 1.35rem; }
  .ct-section span { color: #5e7a90; font-size: 0.88rem; }
  .ct-note {
    border-left: 3px solid #4b91c1; background: #eef5fb;
    padding: 10px 14px; border-radius: 0 10px 10px 0;
    color: #294a63; font-size: 0.9rem; margin: 12px 0;
  }
  .ct-card {
    background: #ffffff; border: 1px solid #e1e9f1; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 3px 14px rgba(20, 50, 77, 0.06);
    height: 100%;
  }
  .ct-card-label {
    font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: #61809a; font-weight: 700;
  }
  .ct-card-value {
    font-size: 1.55rem; font-weight: 750; color: #14324d;
    line-height: 1.25; margin: 6px 0 2px;
  }
  .ct-card-foot { font-size: 0.82rem; color: #5e7a90; }
  div[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e1e9f1; border-radius: 12px;
    padding: 10px 14px; box-shadow: 0 2px 10px rgba(20, 50, 77, 0.05);
  }
  div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 12px; overflow: hidden;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 10px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 16px;
  }
</style>
"""


def configure_page(title: str = "ContextTier") -> None:
    """Set the shared page config. Call before any other Streamlit command."""
    st.set_page_config(
        page_title=f"{title} · ContextTier",
        page_icon="🗂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    """Inject the shared dashboard stylesheet (safe on every page)."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def full_width(renderer: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Render at full container width across supported Streamlit versions.

    Streamlit 1.50+ accepts ``width="stretch"``; older releases only understand
    the legacy full-width flag. We try the modern argument first and fall back
    only when the installed version rejects it.
    """
    try:
        return renderer(*args, width="stretch", **kwargs)
    except TypeError as error:
        if "width" not in str(error):
            raise
        legacy_flag = "use_container_" + "width"
        return renderer(*args, **{legacy_flag: True}, **kwargs)


def hero(eyebrow: str, title: str, description: str, pills: list[str] | None = None) -> None:
    """Render the consistent hero banner used at the top of every page."""
    pill_html = ""
    if pills:
        items = "".join(
            f'<span class="ct-pill">{html_escape(str(item))}</span>' for item in pills
        )
        pill_html = f'<div class="ct-pills">{items}</div>'
    st.markdown(
        '<div class="ct-hero">'
        f'<span class="ct-eyebrow">{html_escape(eyebrow)}</span>'
        f'<h1>{html_escape(title)}</h1>'
        f'<p>{html_escape(description)}</p>'
        f"{pill_html}</div>",
        unsafe_allow_html=True,
    )


def section(title: str, description: str = "") -> None:
    """Render a compact section heading with an optional subtitle."""
    subtitle = f" <span>{html_escape(description)}</span>" if description else ""
    st.markdown(
        f'<div class="ct-section"><h2>{html_escape(title)}</h2>{subtitle}</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Render a small explanatory callout."""
    st.markdown(f'<div class="ct-note">{html_escape(text)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, foot: str = "") -> None:
    """Render a compact KPI card using plain HTML (no widget state)."""
    foot_html = f'<div class="ct-card-foot">{html_escape(foot)}</div>' if foot else ""
    st.markdown(
        '<div class="ct-card">'
        f'<div class="ct-card-label">{html_escape(label)}</div>'
        f'<div class="ct-card-value">{html_escape(value)}</div>'
        f"{foot_html}</div>",
        unsafe_allow_html=True,
    )


def html_escape(text: str) -> str:
    """Escape text inserted into the inline HTML helpers."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_sidebar() -> None:
    """Render the persistent sidebar; page navigation is auto-generated."""
    st.sidebar.title("ContextTier")
    st.sidebar.caption("Context-aware data tiering · research dashboard")
    st.sidebar.markdown(
        "**Navigate**\n\n"
        "- Overview\n"
        "- Results Explorer\n"
        "- Figures\n"
        "- Replay Mode (reconstructed playback)\n\n"
        "All numbers are read from the result files in `results/`."
    )
    selected_model = st.sidebar.selectbox(
        "Model view",
        options=("cnn", "mlp"),
        key="dashboard_model",
        help=(
            "Model-specific runs are stored under results/models/. The current "
            "matrix remains the preserved CNN baseline until a model-specific "
            "matrix is generated."
        ),
    )
    st.sidebar.caption(f"Selected model: `{selected_model}`")
    st.sidebar.info(
        "Replay Mode is **reconstructed playback**: migrations are rebuilt from "
        "the logged migration events, and `tier_served` is authoritative.",
        icon="▶️",
    )
    with st.sidebar.expander("Honesty note"):
        st.markdown(HONESTY_NOTE)
    st.sidebar.caption(f"Streamlit {st.__version__}")


def show_honesty_note() -> None:
    """Show the standing emulation/NoPFS-style disclaimer."""
    st.info(HONESTY_NOTE, icon="ℹ️")


@st.cache_data(show_spinner=False)
def cached_tables() -> dict[str, pd.DataFrame]:
    """Cache the four Phase 8 CSV tables for the session."""
    return load_tables()


@st.cache_data(show_spinner=False)
def cached_experiment_manifest() -> dict[str, Any]:
    """Cache the Phase 8 provenance manifest."""
    return experiment_manifest()


@st.cache_data(show_spinner=False)
def cached_bootstrap() -> pd.DataFrame:
    """Cache the optional paired-bootstrap comparison table."""
    return load_bootstrap()


@st.cache_data(show_spinner=False)
def cached_compute_benchmark() -> dict[str, Any]:
    """Cache the measured CPU step-time benchmark."""
    return compute_benchmark()


@st.cache_data(show_spinner=False)
def cached_figures() -> tuple[dict[str, Any], ...]:
    """Cache the Phase 9 figure catalog."""
    return tuple(figure_catalog())


@st.cache_data(show_spinner=False)
def cached_replays() -> tuple[ReplayRun, ...]:
    """Cache the list of replayable saved traces."""
    return tuple(discover_replays())


def data_error(error: Exception) -> None:
    """Show a consistent message when a required result artefact is missing."""
    st.error(f"Result artefacts unavailable: {error}")
    st.code("python experiments.py && python plots.py", language="bash")


def download_dataframe(frame: pd.DataFrame, filename: str, label: str) -> None:
    """Offer a filtered table as a UTF-8 CSV download."""
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def download_file(path: Path | None, label: str) -> None:
    """Offer a single artefact (e.g. a PDF figure) as a download."""
    if path is None or not path.is_file():
        st.caption(f"{label}: artefact not available")
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/pdf" if path.suffix == ".pdf" else "image/png",
    )
