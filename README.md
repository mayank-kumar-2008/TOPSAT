# TOPSAT
### Topology-Guided Novelty Detection for Bandwidth-Constrained Satellite Downlinks

**FAR AWAY 2026 · Space & Aerospace Track**

---

## The Problem

A satellite gets a ~97-minute orbit window and a fixed downlink budget. Today, that
budget is spent on whatever tiles happen to fall in the imaging path — roughly
**90% of transmitted imagery is terrain already seen before**. Onboard hardware has
no understanding of *what's actually new* in a scene, so the rare, scientifically
valuable observation (a flood, a wildfire, an unmapped structure) competes for
bandwidth with routine farmland and ocean tiles — and usually loses.

## The Idea

TOPSAT scores every tile for **topological novelty** before it's queued for
downlink, and uses that score to decide what gets transmitted first.

```
Tile → RemoteCLIP embedding → Ripser H₁ persistence diagram
     → Bottleneck distance vs. archive classes (novelty score)
     → Knapsack scheduler → downlink decision
```

- **RemoteCLIP** (ViT-L-14, satellite-tuned) embeds each tile into a shared
  semantic space.
- **Ripser** computes the H₁ persistence diagram of each tile's local
  neighbourhood — i.e. whether it sits inside a "hole" in the known-terrain
  manifold.
- **Bottleneck distance** between a tile's diagram and the archive classes'
  diagrams gives a stable, noise-robust novelty score.
- A **knapsack scheduler** fills the fixed downlink budget with the
  highest-novelty tiles first.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.873 [0.832, 0.909] |
| Cohen's d | 1.529 (large effect, p = 4.63e-31) |
| OOD tiles detected | 148/150 (98.7%) |
| Scheduler lift | 1.62× more novel tiles at fixed 12 MB budget |
| Ripser H₁ latency | 7.4 ms / window |
| RemoteCLIP latency (CPU) | 3,250 ms / tile — current bottleneck |

Full evaluation, per-class breakdown, compute-cost analysis, and a baseline
comparison are in [`notebooks/topsat_mvp.ipynb`](notebooks/topsat_mvp.ipynb).

## Repository Structure

```
topsat/
├── README.md
├── requirements.txt               # pip dependencies (Python 3.9+)
├── presentation.html              # pitch deck interactive webpage (open in browser)
├── topsat_demo.html               # interactive mission control demo
├── notebooks/
│   ├── topsat_mvp.ipynb           # main pipeline: RemoteCLIP + Ripser + knapsack
│   └── topoencoder_future_work.ipynb   # v2: topology-native learned encoder
└── docs/
    └── ARCHITECTURE.md            # design notes & theoretical grounding
```

## Setup & Running

The notebooks are designed for **Google Colab (T4/A100)** .
Local Python 3.9+ also works — see the local setup path below.

### Option A — Google Colab (recommended)

1. Upload or open `notebooks/topsat_mvp.ipynb` in Colab.
2. **Run Cell 1** — it installs all dependencies automatically:
   ```
   ripser, persim, open_clip_torch, torch (cu118), torchvision
   ```
3. Run all remaining cells top to bottom.
4. Dataset: **EuroSAT** is downloaded automatically in the dataset cell via `torchvision.datasets.EuroSAT`.

> **Note:** do **not** install `giotto-tda` — it downgrades numpy and breaks the TDA stack.

### Option B — Local / venv

```bash
git clone <repo-url>
cd topsat

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# PyTorch with CUDA 11.8 (skip if you want CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Launch Jupyter
jupyter notebook notebooks/topsat_mvp.ipynb
```

### Running the TopoEncoder notebook

```bash
# Same environment — one extra install for UMAP visualisation
pip install umap-learn

jupyter notebook notebooks/topoencoder_future_work.ipynb
```

> The TopoEncoder notebook mounts Google Drive in Cell 1 (for Colab checkpoint saving).
> If running locally, comment out the `drive.mount(...)` call — checkpoints will save to `./` instead.

### Viewing the demo

Open `presentation.html` or `topsat_demo.html` directly in any browser — no server needed.
Or visit the https://nilay999.github.io/TOPSAT/ .

## Future: TopoEncoder

`notebooks/topoencoder_future_work.ipynb` replaces RemoteCLIP (307M params,
3.25s/tile on CPU) with **TopoEncoder** — a lightweight EfficientNet-B0
backbone trained with a combined contrastive + topological loss:

```
L_total = L_contrastive + λ₁ · L_topo + λ₂ · L_uniformity
```

Target: ~5M params, <10ms inference on ARM-class onboard hardware, fully
on-board pipeline with no ground-station round trip.

## Why This Works

1. **Coverage-hole framing** (Ghrist) — novelty detection is reframed as
   detecting topological holes in the archive's coverage manifold.
2. **Ripser H₁** measures exactly these holes around each tile, robust to
   noise.
3. **Bottleneck distance** gives a stable metric between persistence
   diagrams — small perturbations in the image don't cause large score jumps.
4. **Knapsack scheduling** turns novelty scores into an optimal downlink
   selection under a hard bandwidth constraint.

## Team

Built for FAR AWAY 2026. Maintained by Nilay Pal (Leader), Mayank Kumar, Mayank Kumar, Nainish, Naman Yadav.
