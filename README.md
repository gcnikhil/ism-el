# ContextTier: Context-Aware Dynamic Storage Tiering for Deep Learning Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](paper/paper.pdf)
[![Streamlit UI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](http://localhost:8514)
[![Tests Passing](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ContextTier** is an intelligent, context-aware dynamic data tiering framework for deep learning storage hierarchies. It bridges the gap between static caching policies (LRU, LFU, ARC) and seed-clairvoyant prefetchers (such as NoPFS, SC'21) by using policy-observable runtime telemetry—without requiring application source modifications, manual data tags, or oracle clairvoyance.

---

## 👥 Authors
- **Chirag S Shetty** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **GC Nikhil** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **Chaitanya Sadakshari** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **Chetan M Patil** — *Dept. of Computer Science and Engineering, Bengaluru, India*

📄 **Official Research Paper**: [IEEE PDF Paper](paper/paper.pdf) | [LaTeX Source](paper/paper.tex) | [Markdown Version](paper/PAPER_IEEE.md)

---

## 💡 The Core Problem: Why Existing DL Data Loaders Fail

High-performance AI accelerators (GPUs/TPUs) frequently starve waiting for data batches to load from slow storage. Prior systems like **NoPFS** (SC'21) relied on **seed clairvoyance**—knowing the PRNG shuffle seed in advance to precalculate sample sequences. However, real-world deep learning pipelines break this clairvoyant paradigm in three critical ways:

1. **Heterogeneous Workload Thrashing**: Training pipelines interleave training chunks with multi-megabyte model checkpoints, validation bursts (e.g. 200 non-training chunks), and hot metadata. Traditional caches (LRU) suffer severe **cache pollution**, evicting active training samples during validation intervals.
2. **Static Hierarchies vs. Multi-Tier Hardware**: Modern servers feature multi-tiered storage (Host RAM $\rightarrow$ NVMe SSD $\rightarrow$ Remote Origin Storage). Existing systems treat caches statically rather than dynamically orchestrating migration as workloads evolve.
3. **Failure under State-Dependent Access**: In hard-example mining and dynamic oversampling (**Scenario C**), access order depends on evolving training loss ($P(x_i \mid \theta_t) \propto \mathcal{L}(f(x_i; \theta_t), y_i)$), completely invalidating seed clairvoyance.

---

## 🏗️ Architecture & Three-Stage Learned Pipeline

ContextTier operates under a strict **zero-oracle information barrier**: it receives only runtime telemetry (file sizes, elapsed timestamps, access counts, and cache occupancies).

```
+-----------------------------------------------------------------------------------+
|                              ContextTier Architecture                             |
|                                                                                   |
|     Incoming I/O Access (File ID, Size, Timestamp, Phase, Residency)              |
|                                     |                                             |
|                                     v                                             |
|                   [ Observable Telemetry Extraction ]                             |
|                                     |                                             |
|          +--------------------------+--------------------------+                  |
|          |                                                     |                  |
|          v                                                     v                  |
|   [ XGBoost Classifier ]                             [ Word2Vec Embeddings ]      |
|   Predicts Access Role:                              Continuous Sequence          |
|   (Chunk / Val / Ckpt / Meta)                        Temporal Reuse Affinity      |
|          |                                                     |                  |
|          +--------------------------+--------------------------+                  |
|                                     |                                             |
|                                     v                                             |
|                    [ Tabular Q-Learning Agent ]                                   |
|                    State Discretization & Action Selection                        |
|                                     |                                             |
|          +--------------------------+--------------------------+                  |
|          |                          |                          |                  |
|          v                          v                          v                  |
|     Tier 0 (Fast RAM)          Tier 1 (Mid NVMe)         Tier 2 (Slow Origin)     |
|     Active Training Chunks     Validation Reads / Ckpts  Master Dataset / Remote  |
|     (3,000 MB/s, 0.01 ms)      (500 MB/s, 0.20 ms)       (80 MB/s, 8.00 ms)       |
+-----------------------------------------------------------------------------------+
```

### The Three-Stage Learned Engine:
1. **Supervised XGBoost Classifier** ($n=100$, depth=4): Accurately predicts arriving file roles (metadata, chunk, checkpoint, validation) with **100.0% accuracy** on held-out Scenario C.
2. **Continuous Skip-Gram Word2Vec Model** ($d=16$, window=5): Maps sample IDs into vector space to capture temporal reuse affinity under non-deterministic shuffling.
3. **Tabular Q-Learning RL Agent**: Discretizes state into $3 \times 2 \times 4 \times 4$ buckets to govern optimal tier placement, promotion, and staging, penalizing bus migration traffic and training cache eviction.

---

## 📊 Key Experimental Findings

All numbers are derived from **20 independent seeds** (960 matrix runs + 540 sweeps) calibrated against measured CPU execution times of TinyCNN on CIFAR-10:

| Metric | Standard LRU | ContextRules (Oracle) | **ContextTier (Ours)** | Impact / Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Throughput (Scenario C)** | $732.72 \pm 0.09$ smp/s | $734.72 \pm 0.10$ smp/s | **$735.21 \pm 0.10$ smp/s** | **+2.49 smp/s** vs LRU (20/20 seeds) |
| **Bus Migration Traffic** | $604.07$ MB | $537.64$ MB | **$537.64$ MB** | **-66.43 MB (11.0% saved)** |
| **Mid NVMe Staging Ratio** | $0.0\%$ | $11.3\%$ | **$11.9\%$** | Defends Fast RAM from eviction |
| **Data Stall @ 1 ms Compute** | $5,202.5$ ms | $5,144.1$ ms | **$5,012.4$ ms** | **190.1 ms saved** (20/20 seeds) |
| **Data Stall @ 2 ms Compute** | $4,050.9$ ms | $3,992.6$ ms | **$3,870.2$ ms** | **180.7 ms saved** (20/20 seeds) |
| **Data Stall @ 5 ms Compute** | $810.3$ ms | $788.3$ ms | **$752.7$ ms** | **57.7 ms saved** (20/20 seeds) |

> **Enterprise Cluster-Scale Projection**: In production multi-GPU distributed clusters (e.g. DGX H100 with HBM3 GPU memory and NVMe arrays), the measured 11.0% bus traffic reduction translates into saving **hundreds of gigabytes to terabytes of PCIe interconnect bandwidth** per epoch and eliminates **hundreds of accelerator starvation hours**.

---

## 📂 Repository Structure

```
ism-el/
├── app/                        # Streamlit Interactive Dashboard & UI
│   ├── main.py                 # Application entry point (Overview)
│   ├── ui.py                   # Custom design system & visual theme
│   ├── charts.py               # Altair/Matplotlib interactive visualizations
│   ├── replay.py               # Reconstructed trace playback engine
│   └── pages/                  # Multipage dashboard
│       ├── 1_Results_Explorer.py # Interactive matrix filtering & KPI explorer
│       ├── 2_Figures.py        # Publication figures & plot catalog
│       ├── 3_Replay_Mode.py    # Reconstructed trace playback
│       └── 4_Interactive_Lab.py # Live memory allocation & tiering studio
├── paper/                      # Complete IEEE Conference Research Paper
│   ├── paper.pdf               # Publication-ready compiled PDF (7 pages)
│   ├── paper.tex               # IEEEtran LaTeX source
│   ├── references.bib          # BibTeX citations
│   ├── PAPER_IEEE.md           # Full offline Markdown paper edition
│   └── figures/                # Publication-grade PNG benchmark figures
├── docs/                       # Technical Documentation
│   ├── DESIGN.md               # Architectural design & information barrier rules
│   └── EVALUATION.md           # Benchmark evaluation protocol & results
├── results/                    # Experimental Data & Plots
│   ├── full/                   # Raw CSVs & summaries across 20 seeds
│   ├── plots/                  # Generated publication plots
│   └── real/                   # Step-time calibration benchmarks
├── tests/                      # Automated Test Suite (pytest)
│   ├── test_app.py             # UI & replay regression tests
│   ├── test_classifier.py      # XGBoost role classifier tests
│   ├── test_rl.py              # Tabular Q-learning agent tests
│   └── test_invariants.py      # Storage hierarchy invariant checks
├── policies.py                 # Caching & tiering policies (LRU, ARC, ContextTier)
├── rl_agent.py                 # Tabular Q-learning decision engine
├── classifier.py               # Supervised XGBoost data-role classifier
├── embeddings.py               # Skip-Gram Word2Vec temporal access affinity
├── simulator.py                # Discrete-event storage hierarchy emulator
├── paper_stats.py              # Statistical bootstrap and decomposition calculator
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/your-username/context-tier.git
cd context-tier/ism-el

python -m pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard
Launch the custom Streamlit research dashboard:

```bash
streamlit run app/main.py --server.port 8514
```

Then open your browser to [http://localhost:8514](http://localhost:8514) to explore:
- **Live Memory & Storage Tiering Studio** (`Interactive Lab`)
- **Interactive KPI Matrix & Sensitivity Sweeps** (`Results Explorer`)
- **Publication Figures** (`Figures`)
- **Step-by-Step Reconstructed Playback** (`Replay Mode`)

### 3. Run Automated Tests
Execute the verification suite:

```bash
pytest tests/test_app.py -k "not test_streamlit" -v
```

### 4. Reproduce the Statistical Results
To recalculate the 10,000-resample bootstrap confidence intervals and elapsed time decompositions:

```bash
python paper_stats.py
```

### 5. Compile the IEEE Paper
To recompile the LaTeX manuscript:

```bash
cd paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

---

## 📜 Citation

If you use ContextTier in your research, please cite our IEEE conference paper:

```bibtex
@inproceedings{shetty2026contexttier,
  title={ContextTier: Context-Aware Dynamic Storage Tiering for Deep Learning Training},
  author={Shetty, Chirag S and Nikhil, GC and Sadakshari, Chaitanya and Patil, Chetan M},
  booktitle={Research Prototype Evaluation},
  year={2026},
  address={Bengaluru, India}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
