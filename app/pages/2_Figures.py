"""Figures page: display the publication figures and their PDFs."""

from __future__ import annotations

import streamlit as st

from app.ui import (
    apply_theme,
    cached_experiment_manifest,
    cached_figures,
    configure_page,
    data_error,
    download_file,
    full_width,
    hero,
    note,
    render_sidebar,
    section,
    show_honesty_note,
)

FIGURE_DESCRIPTIONS: dict[str, str] = {
    "data_wait_all": "All policies on all scenarios; symlog x-axis because idealized and tied cases span orders of magnitude.",
    "data_wait_deployable": "Readable linear-scale comparison of deployable policies (oracle, declared-type and idealized rows omitted).",
    "scenario_c_focus": "Scenario C data wait and throughput side by side, including the oracle bounds.",
    "capacity_sweep": "Data wait against fast-tier capacity on Scenario C, with seed error bars.",
    "compute_sweep": "Data wait against compute time per step on Scenario C, with seed error bars.",
    "context_ablation": "Paired per-seed data wait saved by ContextRules' declared-type channel.",
    "latency_percentiles": "Mean / p95 / p99 read latency for representative Scenario C policies.",
}


def main() -> None:
    """Streamlit entry point."""
    configure_page("Figures")
    apply_theme()
    render_sidebar()
    hero(
        eyebrow="Visual Artifacts",
        title="Publication Figures",
        description=(
            "Browse every high-resolution plot and export vector PDFs directly. "
            "All figures are plotted from the identical benchmark result tables."
        ),
        pills=["High-Res Preview", "Vector PDF Download", "Exact Plot Manifest"],
    )
    try:
        figures = cached_figures()
        manifest = cached_experiment_manifest()
    except (FileNotFoundError, ValueError) as error:
        data_error(error)
        return

    section("Featured Plot", f"select from {len(figures)} plots listed in the manifest")
    options = [figure["stem"] for figure in figures]
    if not options:
        note("No figures are listed in the plot manifest yet.")
        return
    selected = st.selectbox(
        "Select Figure to Inspect",
        options=options,
        format_func=lambda stem: next(
            figure["title"] for figure in figures if figure["stem"] == stem
        ),
    )
    figure = next(item for item in figures if item["stem"] == selected)
    st.subheader(figure["title"])
    st.caption(FIGURE_DESCRIPTIONS.get(figure["stem"], ""))

    if figure["png"] is not None and figure["png"].is_file():
        full_width(st.image, str(figure["png"]))
    else:
        st.warning("PNG artefact missing; regenerate with `python plots.py`.")
    download_file(figure["pdf"], "⬇️ Download Vector PDF")

    section("Figure Gallery", "all 8 benchmark charts")
    cols = st.columns(2)
    for idx, item in enumerate(figures):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(FIGURE_DESCRIPTIONS.get(item["stem"], ""))
                if item["png"] is not None and item["png"].is_file():
                    full_width(st.image, str(item["png"]))
                download_file(item["pdf"], f"⬇️ Download PDF ({item['stem']})")

    with st.expander("📖 Experimental Provenance & Source Tables"):
        st.markdown("**Source Result Files**")
        for path in manifest.get("source_tables", []):
            st.code(str(path), language="text")
        show_honesty_note()
        st.caption(
            "Replay Mode shows reconstructed playback of saved traces, with "
            "log-derived migrations and the trace's tier_served as authoritative."
        )


if __name__ == "__main__":
    main()
