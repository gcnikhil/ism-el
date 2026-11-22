# ContextTier demo website

Two ways to view the same results. Both read every experimental number from
the files in `results/`; no experimental value is hard-coded in the UI.

## 1. Streamlit dashboard (interactive)

```bash
python -m pip install -r requirements.txt   # includes streamlit>=1.30
python -m streamlit run app/main.py --server.port 8513
```

Open <http://localhost:8513>. Leave the terminal running while you browse.

| Page | What it shows |
|---|---|
| **Overview** (`app/main.py`) | Project pitch, headline mean ± std read from `results/full/summary.csv`, scenario definitions and run provenance. |
| **Results Explorer** (`app/pages/1_Results_Explorer.py`) | Filter the main matrix by scenario, policy and information class; mean ± std charts, per-seed dots, sweeps and CSV download. |
| **Figures** (`app/pages/2_Figures.py`) | The publication PNG figures and vector PDF downloads described by `results/plots/plot_manifest.json`. |
| **Replay Mode — reconstructed playback** (`app/pages/3_Replay_Mode.py`) | Step through any saved `*_trace.csv`: access details, reconstructed tier residency, wait timeline and recent migrations. The evaluation-only `true_type` column is hidden by default. |

**Why a Streamlit tab can look completely blank.** The HTML shell loads without
the Python script if the `/_stcore/stream` WebSocket never connects. Check the
server terminal: with `uvicorn==0.45.0` and `websockets==17.x` Streamlit 1.6x
prints a handshake warning and the stream can abort, leaving an empty page. Fix
that environment with `python -m pip install "uvicorn>=0.52"` (or pin
`websockets>=16.1.1,<17`) and restart the server. Also confirm you opened the
URL printed in the terminal (port matters) and that the process is still alive.
The static dashboard below is unaffected by this issue.

## 2. Static dashboard (no server, no CDN)

```bash
python build_site.py
```

Open `site/index.html` directly in any modern browser (double-click it, or use
`file:///.../site/index.html`). The generated file embeds all result tables,
the manifest-listed figures and saved traces, so it works offline and even when
Streamlit is not installed. Re-run `python build_site.py` after regenerating
results to refresh it. It has the same four sections: Overview, Results
Explorer, Figures and Replay Mode (explicitly labelled *reconstructed playback*).

## Reconstructed playback semantics

Replay Mode / reconstructed playback is **not** a live simulation. It starts
every file in the initial tier derived from the run, then applies completed
migrations from the saved `*_migrations.csv` in timestamp order to rebuild an
*approximate* tier residency. The trace's own `tier_served` column is the
**authoritative** served tier for each access and is shown as such. In-flight
prefetches are absent from the migration log, so reconstructed occupancy is
illustrative. No value is invented.

Regenerate the data shown by both dashboards with:

```bash
python experiments.py   # results/full/*
python plots.py         # results/plots/*
```

`app/data.py`, `app/replay.py`, `app/charts.py` and `build_site.py` contain no
Streamlit imports, so their behaviour is covered by headless tests in
`tests/test_app.py`.
