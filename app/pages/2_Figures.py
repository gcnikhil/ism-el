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
        eyebrow="Publication artefacts",
        title="Figures",
        description=(
            "The exact plots used in the paper, generated from the same result "
            "tables you can explore in the Results Explorer."
        ),
        pills=["PNG preview", "Vector PDF download", "Manifest-driven"],
    )
    show_honesty_note()
    try:
        figures = cached_figures()
        manifest = cached_experiment_manifest()
    except (FileNotFoundError, ValueError) as error:
        data_error(error)
        return

    section("Figure viewer", f"{len(figures)} figures listed by results/plots/plot_manifest.json")
    options = [figure["stem"] for figure in figures]
    if not options:
        note("No figures are listed in the plot manifest yet.")
        return
    selected = st.selectbox(
        "Figure",
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
    download_file(figure["pdf"], "Download vector PDF")

    with st.expander("Provenance and honesty notes"):
        st.markdown("**Source tables**")
        for path in manifest.get("source_tables", []):
            st.code(str(path), language="text")
        for item in manifest.get("honesty_notes", []):
            st.caption(f"• {item}")

    section("All figures", "quick gallery")
    for item in figures:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.caption(FIGURE_DESCRIPTIONS.get(item["stem"], ""))
            if item["png"] is not None and item["png"].is_file():
                full_width(st.image, str(item["png"]))
            download_file(item["pdf"], f"PDF · {item['stem']}")
    note(
        "Replay Mode shows reconstructed playback of saved traces, with "
        "log-derived migrations and the trace's tier_served as authoritative."
    )


if __name__ == "__main__":
    main()
