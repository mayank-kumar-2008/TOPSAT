# Architecture & Theoretical Grounding

This document covers the design rationale behind both pipeline versions.

---

## V1 — Production Pipeline (`notebooks/topsat_mvp.ipynb`)

```
Tile → RemoteCLIP (ViT-L-14) → 768-d embedding
     → Ripser H₁ persistence diagram (local neighbourhood)
     → Bottleneck distance vs. archive-class diagrams → novelty score
     → Knapsack scheduler → downlink decision
```

### Why bottleneck distance

Bottleneck distance is the max-cost matching between two persistence
diagrams. It is provably stable under small perturbations of the input —
sensor noise or minor scene changes shift the score smoothly rather than
discontinuously, which matters for a metric driving real downlink decisions.

### Why a knapsack scheduler

The downlink window is a hard capacity constraint (fixed MB per pass).
Framing tile selection as a knapsack problem — value = novelty score,
weight = tile size — gives the provably optimal selection for that budget,
not just a greedy heuristic.

### Compute profile

| Stage | Latency |
|---|---|
| RemoteCLIP (CPU) | 3,250 ms / tile — current bottleneck |
| Ripser H₁ | 7.4 ms / window |

RemoteCLIP's cost is the motivation for V2 below.

---

## V2 — TopoEncoder (`notebooks/topoencoder_future_work.ipynb`)

Replaces RemoteCLIP with a lightweight, topology-aware encoder trained
end-to-end:

```
L_total = L_contrastive + λ₁ · L_topo + λ₂ · L_uniformity
```

Backbone: EfficientNet-B0 → 128-d unit-norm hyperspherical embedding
(~5M params, target <10ms on ARM-class onboard hardware).

### Theoretical grounding

**1. L2-normalised embeddings → unit hypersphere → cosine distance as TDA metric**

Projecting embeddings onto the 128-d unit hypersphere via z/‖z‖₂ guarantees
every pairwise cosine distance lies in [0, 2] and satisfies the triangle
inequality — making it an admissible metric for Ripser's Vietoris–Rips
filtration.

**2. Topology proxy loss → non-degenerate H₁ structure**

Ripser itself isn't differentiable, so training uses a proxy on the cosine
distance matrix D:

```
L_topo = -Var(D) + α(D̄ - 0.3)²
```

- `-Var(D)` maximises pairwise-distance spread — a necessary condition for
  non-trivial 1-cycles (H₁ features). A collapsed embedding (Var(D) ≈ 0)
  produces degenerate persistence diagrams.
- `α(D̄ - 0.3)²` anchors the mean distance near 0.3, preventing both
  collapse and explosion of the embedding, either of which destroys the
  filtration structure.

**3. Cross-sensor invariance on the hypersphere**

Unit-norm projections are invariant to isotropic scale changes in the input
feature space (HYPO, ICLR 2024). Since satellite sensors vary in radiometric
gain, mapping to the hypersphere gives a domain-invariant representation
without per-sensor normalisation.

**4. 128-d vs 768-d — persistent homology reliability**

Vietoris–Rips persistence is more statistically reliable when ambient
dimension is proportional to the data's intrinsic dimension (NeurIPS 2024).
EuroSAT land-use categories have estimated intrinsic dimension ≈ 10–30, so
128-d is far more appropriate than RemoteCLIP's 768-d — reducing
curse-of-dimensionality artefacts that inflate false-positive novelty scores.

---

## Conceptual Framing

Novelty detection is reframed as **coverage-hole detection** on the archive's
manifold (Ghrist): known terrain occupies compact regions of the embedding
space, and a genuinely novel observation falls in a topological "hole" —
detectable as a non-trivial H₁ feature relative to the archive's persistence
diagram.
