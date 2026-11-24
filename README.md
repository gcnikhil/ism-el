# ContextTier: Context-Aware Dynamic Storage Tiering for Deep Learning

> **An intelligent, context-aware storage controller that stops AI training pipelines from stalling on I/O by dynamically learning which files belong in Fast RAM, Mid NVMe, or Slow Origin Storage.**

---

## 👥 The Team
- **Chirag S Shetty** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **GC Nikhil** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **Chaitanya Sadakshari** — *Dept. of Computer Science and Engineering, Bengaluru, India*
- **Chetan M Patil** — *Dept. of Computer Science and Engineering, Bengaluru, India*

📄 **Official Research Paper**: [Download IEEE PDF Paper](paper/paper.pdf) | [Read LaTeX Source](paper/paper.tex) | [Read Markdown Edition](paper/PAPER_IEEE.md)

---

## 📖 What is ContextTier?

When training deep learning models (like Vision Transformers, ResNets, or Large Language Models), the computing processor (GPU, TPU, or CPU) is only as fast as the rate at which data is fed to it. In modern AI clusters, **accelerators spend up to 40–70% of their time sitting idle**, simply waiting for data batches to be read, decoded, and transferred from slow disks or network storage. This is known as the **I/O Bottleneck**.

**ContextTier** is an autonomous storage tiering system designed specifically for deep learning pipelines. Instead of treating all data accesses identically, ContextTier **observes access patterns in real time** (file sizes, arrival intervals, and training phases) and intelligently routes files across a multi-tier memory hierarchy:
- ⚡ **Tier 0 (Fast RAM Cache)**: 3,000 MB/s, 0.01 ms latency — Kept strictly for frequently reused active training data.
- 💾 **Tier 1 (Mid NVMe Staging)**: 500 MB/s, 0.20 ms latency — High-capacity staging buffer for bursts of validation data and checkpoints.
- 🌐 **Tier 2 (Slow Origin Storage)**: 80 MB/s, 8.00 ms latency — The master disk or cloud storage holding the raw dataset.

By intelligently segregating different types of data, ContextTier **stops cache pollution, reduces internal bus traffic by 11.0%, saves 190+ ms of accelerator stall time per step, and boosts overall training throughput.**

---

## 🔍 The Problem: Why Traditional Systems Fail

### 1. The Cache Thrashing Dilemma (Why standard LRU fails)
In traditional operating systems, caches use **LRU (Least Recently Used)** replacement: whatever was accessed most recently stays in fast memory.
- During training, thousands of training chunks are cycled through Fast RAM.
- Every few hundred steps, the training job pauses for a **validation phase**, reading 200 validation chunks.
- **The disaster**: Because validation chunks were just read, standard LRU treats them as "important" and promotes all of them into Fast RAM—**kicking out the active training chunks**!
- When training resumes, every single training sample suffers a cache miss and must be re-read over the slow system bus. This churn wastes gigabytes of bus bandwidth and freezes the processor.

### 2. The Collapse of "Seed Clairvoyance" (Why NoPFS fails)
State-of-the-art systems like **NoPFS** (SC'21) tried to solve caching by looking at the random shuffle seed of PyTorch's DataLoader. If you know the seed, you know which training samples will be requested in the future. This works under simple uniform random shuffling.
- **Where it breaks**: Modern deep learning uses **dynamic hard-example oversampling, curriculum learning, and loss-driven sampling**. Samples with higher loss are sampled more frequently based on model weights:
  $$P(x_i \mid \theta_t) \propto \mathcal{L}(f(x_i; \theta_t), y_i)$$
- Because the access sequence depends on dynamic model training loss rather than a static seed, seed clairvoyance completely collapses and provides zero predictive power.

---

## 🧠 How ContextTier Works: The 3-Stage Autonomous Brain

ContextTier solves this with a **zero-oracle information barrier**: it receives **no true file labels**, no manual developer tags, and no hints from the user. It observes only raw telemetry:
$$\mathbf{x}_i = [\text{file\_size}, \Delta t_{\text{arrival}}, \text{access\_count}, \text{workload\_phase}, \text{current\_tier}, \text{RAM\_occupancy}, \text{NVMe\_occupancy}]$$

From this telemetry, ContextTier runs a coordinated three-stage decision pipeline:

```
+-----------------------------------------------------------------------------------------+
|                                ContextTier Decision Pipeline                            |
|                                                                                         |
|        Incoming File Access (File ID, Size, Arrival Timestamp, Training Phase)          |
|                                        |                                                |
|                                        v                                                |
|                      [ Telemetry Extraction & Normalization ]                           |
|                                        |                                                |
|              +-------------------------+-------------------------+                      |
|              |                                                   |                      |
|              v                                                   v                      |
|    [ 1. XGBoost Role Classifier ]                     [ 2. Word2Vec Embeddings ]        |
|    Predicts data semantic role:                       Continuous token sequence model   |
|    - Metadata: Small files -> Pin in RAM              Computes cosine temporal reuse    |
|    - Validation: Bursts -> Stage in NVMe              affinity under dynamic shuffle.   |
|    - Checkpoint: Large writes -> Drain to Origin                                        |
|              |                                                   |                      |
|              +-------------------------+-------------------------+                      |
|                                        |                                                |
|                                        v                                                |
|                      [ 3. Tabular Q-Learning RL Agent ]                                 |
|                      Discretizes continuous state space (3x2x4x4)                       |
|                      Selects optimal placement / migration action:                      |
|                      { Stay in Tier, Promote to RAM, Stage in NVMe }                    |
|                                        |                                                |
|              +-------------------------+-------------------------+                      |
|              |                         |                         |                      |
|              v                         v                         v                      |
|      Tier 0 (Fast RAM)         Tier 1 (Mid NVMe)        Tier 2 (Slow Origin)            |
|      Active Training Chunks    Validation / Checkpoints Master Dataset Files            |
+-----------------------------------------------------------------------------------------+
```

1. **Stage 1: XGBoost Data-Role Classifier** ($n=100$, max depth=4)
   - Accurately classifies whether an arriving file is a training chunk, validation sample, checkpoint, or metadata.
   - When a validation burst arrives, ContextTier immediately identifies it and **stages it into Tier 1 (Mid NVMe)** instead of letting it pollute Tier 0 (Fast RAM).
   - Achieves **100.0% accuracy** on held-out test scenarios.
2. **Stage 2: Word2Vec Sequence Embeddings** ($d=16$, window=5)
   - Treats file access traces like sentences and file IDs like words.
   - Learns vector representations of sample access affinity to detect when a file is likely to be reused soon, even when random shuffling is non-stationary.
3. **Stage 3: Tabular Q-Learning Reinforcement Learning Agent**
   - Discretizes the system state (file size, residency, training phase, RAM fullness) into a small, stable Q-table ($3 \times 2 \times 4 \times 4$ states by 3 actions).
   - Learns optimal migration policies over virtual episodes using Bellman updates, balancing transfer latency against internal bus traffic.

---

## 🔬 Evaluated Scenarios

To prove ContextTier under every real-world training condition, we evaluated 4 distinct workload scenarios across **20 configured random seeds** (Seeds 0–19):

- **Scenario A (Baseline Shuffle)**: Standard mini-batch SGD on CIFAR-10 with random sample shuffling and per-step metadata reads.
- **Scenario B (Heterogeneous DL Pipeline)**: Scenario A + periodic 4 MB model checkpoint writes every 50 steps + 200 validation reads per interval.
- **Scenario C (State-Dependent Hard-Example Oversampling)**: Scenario B + dynamic loss-driven oversampling. Hard examples with higher loss are sampled repeatedly. Access trajectories are non-deterministic, causing seed clairvoyance to collapse.
- **Scenario D (Capacity-Saturated Control)**: Scenario B with infinite Fast RAM where all files fit in cache (theoretical upper ceiling).

---

## 📈 Real Benchmark Results

Every number reported in this project comes from **960 full matrix runs** and **540 parameter sweeps** calibrated with empirically measured CPU execution times of TinyCNN on CIFAR-10 ($166.986$ ms median per step across 200 forward+backward steps):

### 1. Scenario C Headline Comparison (Means over 20 Seeds)
| Storage Policy | Information Type | Throughput (smp/s) | Fast RAM Hit Rate | Mid NVMe Served | Bus Migration Traffic (MB) | Fast RAM Protection |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AllSlow** (No Cache) | Lower Control | $605.07 \pm 0.00$ | 0.0% | 0.0% | 0.00 MB | N/A |
| **LRU** (Standard Cache) | Observable Baseline | $732.72 \pm 0.09$ | 25.8% | 0.0% | 604.07 MB | ❌ Thrashes |
| **LFU** (Frequency Cache)| Observable Baseline | $732.52 \pm 0.09$ | 25.3% | 0.0% | 608.49 MB | ❌ Thrashes |
| **ARC** (Adaptive Cache) | Observable Baseline | $732.55 \pm 0.09$ | 25.3% | 0.0% | 607.98 MB | ❌ Thrashes |
| **Clairvoyant** (NoPFS)  | Seed Oracle Bound  | $732.72 \pm 0.09$ | 25.8% | 0.0% | 604.07 MB | ❌ Collapses in C |
| **ContextRules** (Oracle)| True Type Bound    | $734.72 \pm 0.10$ | 26.5% | 11.3% | 537.64 MB | 100% Protected |
| **FullContextTier (Ours)**| **Learned Observable** | **$735.21 \pm 0.10$** | **26.5%** | **11.9%** | **537.64 MB** | **100% Protected** |

### 2. Key Discoveries
- 🛡️ **Zero Cache Eviction**: ContextTier routes 11.9% of accesses (all validation reads) to Mid NVMe, completely shielding Fast RAM from cache pollution.
- 📉 **66.43 MB (11.0%) Bus Traffic Saved**: Over 3 epochs, ContextTier reduces unnecessary inter-tier promotions from 604.07 MB down to 537.64 MB.
- 🚀 **Faster Throughput (+2.49 samples/s)**: By avoiding demand promotions, ContextTier saves **958.5 ms of synchronous promotion delay**, achieving higher end-to-end throughput than LRU across all 20 seeds.
- ⚡ **190.1 ms Stall Elimination on Fast Accelerators**: When batch compute times drop into fast accelerator regimes (1–5 ms per step on GPUs/TPUs), ContextTier cuts data stall time by **190.09 ms at 1 ms compute** (winning 20/20 seeds, 95% CI: [184.68, 195.66 ms]).

---

## 🖥️ The Interactive Streamlit Studio

The project includes an interactive web application to explore the data, inspect reconstructed playback, and run live simulations:

```bash
streamlit run app/main.py --server.port 8514
```

| Application Page | What You Can Do |
| :--- | :--- |
| **🏠 Overview** (`app/main.py`) | High-level summary of the research, headline KPI metrics, storage hierarchy cards, and run provenance. |
| **📊 Results Explorer** (`pages/1_Results_Explorer.py`) | Interactive matrix filtering across all 11 policies, 4 scenarios, and 20 seeds; view distribution plots, compute/capacity sweeps, and download CSV data. |
| **📈 Figures Gallery** (`pages/2_Figures.py`) | Browse all publication figures (PNG/PDF) with detailed captions and research explanations. |
| **⏪ Replay Mode** (`pages/3_Replay_Mode.py`) | Step-by-step reconstructed playback of saved execution traces; inspect exact file transfers, access timestamps, and tier occupancy. |
| **🧪 Interactive Lab** (`pages/4_Interactive_Lab.py`) | Custom allocation studio: adjust Fast RAM & Mid NVMe capacities, choose a workload scenario, and watch dynamic tiering execute live! |

---

## 📁 Repository Structure

```
ism-el/
├── app/                        # Streamlit Interactive Web Application
│   ├── main.py                 # Dashboard overview & entry point
│   ├── ui.py                   # Modern clean styling, card components & palettes
│   ├── charts.py               # Interactive chart generators (Altair/Matplotlib)
│   ├── replay.py               # Trace reconstruction and migration playback
│   └── pages/                  # Multipage dashboard sections
│       ├── 1_Results_Explorer.py # Interactive benchmark matrix & sweeps
│       ├── 2_Figures.py        # Publication figures viewer
│       ├── 3_Replay_Mode.py    # Step-by-step trace playback
│       └── 4_Interactive_Lab.py # Live memory simulation studio
├── paper/                      # Full Academic Publication (IEEE Conference Format)
│   ├── paper.pdf               # Compiled 7-page IEEE PDF paper
│   ├── paper.tex               # LaTeX source with IEEEtran format & Algorithm 1
│   ├── references.bib          # Full BibTeX citations
│   ├── PAPER_IEEE.md           # Complete Markdown edition of the paper
│   └── figures/                # 8 high-resolution publication PNG plots
├── docs/                       # Technical Documentation
│   ├── DESIGN.md               # Architecture specification & information barrier rules
│   └── EVALUATION.md           # Benchmark evaluation results & statistical analysis
├── results/                    # Benchmark Outputs
│   ├── full/                   # 960 simulation runs + 540 sweeps across 20 seeds
│   ├── plots/                  # Generated publication plots
│   └── real/                   # Measured TinyCNN CPU execution benchmarks
├── tests/                      # Pytest Automated Test Suite
│   ├── test_app.py             # UI, replay, and headless tests
│   ├── test_classifier.py      # XGBoost role classifier unit tests
│   ├── test_rl.py              # Tabular Q-learning agent tests
│   └── test_invariants.py      # Storage hierarchy invariant tests
├── policies.py                 # Storage policies (LRU, LFU, ARC, ContextTier)
├── rl_agent.py                 # Tabular Q-learning decision engine
├── classifier.py               # XGBoost data-role classifier
├── embeddings.py               # Skip-Gram Word2Vec temporal access affinity
├── simulator.py                # Discrete-event multi-tier storage emulator
├── paper_stats.py              # 10,000-resample bootstrap & decomposition calculator
└── requirements.txt            # Python dependencies
```

---

## ⚡ Quickstart

### 1. Setup Environment
Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/your-username/context-tier.git
cd context-tier/ism-el

python -m pip install -r requirements.txt
```

### 2. Launch the Web Studio
```bash
streamlit run app/main.py --server.port 8514
```
Open **`http://localhost:8514`** in your browser.

### 3. Run Automated Tests
```bash
pytest tests/test_app.py -k "not test_streamlit" -v
```

### 4. Re-calculate Research Statistics
```bash
python paper_stats.py
```

### 5. Compile the IEEE Paper
```bash
cd paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

---

## 📝 Citation

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

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
