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

# Inline CSS keeps the dashboard self-contained: cohesive, professional Blue & Slate engineering theme
APP_CSS = """
<style>
  /* App Root & Main Background: Clean, consistent slate canvas */
  .stApp, [data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  }
  .main .block-container {
    max-width: 1200px;
    padding-top: 1.8rem;
    padding-bottom: 4rem;
  }

  /* Universal High-Contrast Typography */
  h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
  }
  p, span, div, li {
    color: #334155;
  }
  
  /* Widget & Slider Labels */
  label,
  label[data-testid="stWidgetLabel"],
  .stSlider label,
  .stSelectbox label,
  .stRadio label,
  .stMultiSelect label {
    color: #0f172a !important;
    font-weight: 650 !important;
    font-size: 0.90rem !important;
    margin-bottom: 6px !important;
  }
  
  /* Sliders: Theme Blue & Slate */
  div[data-testid="stSlider"] [data-testid="stTickBar"] div,
  div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.80rem !important;
  }
  div[data-testid="stSlider"] [role="slider"] {
    background-color: #2563eb !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
    width: 18px !important;
    height: 18px !important;
  }
  
  /* Dropdowns, Inputs, and Selectboxes: 4-Sided Uniform Border */
  div[data-baseweb="select"] {
    border-radius: 8px !important;
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: border-color 0.15s ease !important;
  }
  div[data-baseweb="select"]:hover,
  div[data-baseweb="select"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
  }
  div[data-baseweb="select"] * {
    color: #0f172a !important;
    font-size: 0.90rem !important;
  }
  div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
  }

  /* Buttons: Consistent Theme Blue */
  button[kind="primary"],
  button[data-testid="baseButton-primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid #2563eb !important;
    padding: 9px 22px !important;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25) !important;
    font-size: 0.90rem !important;
    transition: all 0.15s ease !important;
  }
  button[kind="primary"]:hover,
  button[data-testid="baseButton-primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
  }
  button[kind="secondary"],
  button[data-testid="baseButton-secondary"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    font-weight: 550 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 0.90rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.15s ease !important;
  }
  button[kind="secondary"]:hover,
  button[data-testid="baseButton-secondary"]:hover {
    background: #f8fafc !important;
    border-color: #2563eb !important;
    color: #2563eb !important;
  }

  /* Sidebar: Consistent Slate Panel */
  section[data-testid="stSidebar"],
  [data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
  }
  section[data-testid="stSidebar"] * {
    color: #334155 !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
  }

  /* Hero Card: 4-Sided Uniform Border, Consistent Theme */
  .ct-hero {
    border-radius: 10px;
    padding: 24px 28px;
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    margin-bottom: 24px;
  }
  .ct-hero h1 {
    color: #0f172a !important;
    font-size: 1.85rem;
    margin: 4px 0 8px;
    font-weight: 750;
    letter-spacing: -0.02em;
  }
  .ct-hero p {
    color: #475569 !important;
    margin: 0;
    font-size: 0.95rem;
    max-width: 80ch;
    line-height: 1.55;
  }
  .ct-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    color: #2563eb !important;
    margin-bottom: 6px;
  }
  .ct-pills {
    margin-top: 14px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .ct-pill {
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #1d4ed8 !important;
    background: #eff6ff;
    font-weight: 600;
  }

  /* Section Headings */
  .ct-section {
    margin: 26px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #e2e8f0;
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  .ct-section h2 {
    margin: 0;
    color: #0f172a !important;
    font-size: 1.20rem;
    font-weight: 700;
    letter-spacing: -0.015em;
  }
  .ct-section span {
    color: #64748b !important;
    font-size: 0.85rem;
    font-weight: 450;
  }

  /* Callouts & Notes */
  .ct-note {
    border: 1.5px solid #bfdbfe;
    background: #eff6ff;
    padding: 10px 14px;
    border-radius: 8px;
    color: #1e3a8a !important;
    font-size: 0.86rem;
    margin: 14px 0;
    font-weight: 450;
  }

  /* Metric KPI Cards: 4-Sided Uniform Border */
  .ct-card {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    height: 100%;
    transition: border-color 0.15s ease;
  }
  .ct-card:hover {
    border-color: #2563eb;
  }
  .ct-card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b !important;
    font-weight: 600;
  }
  .ct-card-value {
    font-size: 1.6rem;
    font-weight: 750;
    color: #0f172a !important;
    line-height: 1.25;
    margin: 4px 0 2px;
    letter-spacing: -0.02em;
  }
  .ct-card-foot {
    font-size: 0.82rem;
    color: #2563eb !important;
    font-weight: 500;
  }

  /* Data Tables: 4-Sided Uniform Border */
  div[data-testid="stTable"] table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1.5px solid #cbd5e1;
    background: #ffffff !important;
  }
  div[data-testid="stTable"] th {
    background: #f8fafc !important;
    color: #1e293b !important;
    font-weight: 650 !important;
    padding: 10px 14px !important;
    font-size: 0.84rem !important;
    border-bottom: 1.5px solid #e2e8f0;
  }
  div[data-testid="stTable"] td {
    padding: 9px 14px !important;
    color: #0f172a !important;
    border-top: 1px solid #f1f5f9;
    font-size: 0.84rem !important;
  }

  /* Expanders: 4-Sided Uniform Border */
  div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
  }
  div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #0f172a !important;
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
    st.sidebar.caption("Context-Aware Data Tiering · Research Testbed")
    selected_model = st.sidebar.selectbox(
        "Model Architecture",
        options=("cnn", "mlp"),
        key="dashboard_model",
        help=(
            "Model-specific runs are stored under results/models/. The current "
            "matrix remains the preserved CNN baseline."
        ),
    )
    st.sidebar.caption(f"Active Model: `{selected_model.upper()}`")
    st.sidebar.info(
        "Replay Mode is **reconstructed playback**: migrations are rebuilt from "
        "the logged migration events, and `tier_served` is authoritative.",
        icon="ℹ️",
    )
    with st.sidebar.expander("Research Contract & Honesty"):
        st.markdown(HONESTY_NOTE)


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
