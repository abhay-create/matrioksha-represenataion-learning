# Matryoshka Representation Learning: Geometric & Bias Analysis of Multi-Resolution Word Embeddings

> **An investigation into how Matryoshka Representation Learning (MRL) with Relational Distillation affects the geometric structure and social bias encoding of Word2Vec embeddings.**

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
  - [Standard Skip-Gram (Baseline)](#standard-skip-gram-baseline)
  - [MRL-v3: Naive Multi-Scale Loss](#mrl-v3-naive-multi-scale-loss)
  - [MRL-v4: Relational Distillation](#mrl-v4-relational-distillation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [Geometric Analysis (10 Experiments)](#geometric-analysis-10-experiments)
  - [Bias Evaluation (7 Standard + 15 Deep-Dive)](#bias-evaluation)
- [Results](#results)
  - [Geometric Results](#geometric-results)
  - [Bias Results](#bias-results)
- [How to Run](#how-to-run)
- [Reports](#reports)

---

## Overview

[Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) trains a single embedding model that produces useful representations at multiple granularities — a 768d vector can be truncated to 64d, 128d, etc. and still remain semantically meaningful. This project investigates:

1. **How well does MRL preserve word embedding geometry at lower dimensions?**
2. **Does the MRL training objective amplify, attenuate, or redistribute social bias?**

We train both a Standard Word2Vec (Skip-Gram with Negative Sampling) and an MRL-v4 model (with Relational Distillation) on the **same** corpus and vocabulary, then run **25+ experiments** comparing their geometric and bias properties across all nesting dimensions.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **MRL-v4 prevents L2 norm inflation** | v3's naive per-prefix NS loss allowed magnitude exploitation; v4's RD loss eliminates this (avg norm at d=50: Std=1.83 vs MRL=1.16) |
| **MRL improves low-dim neighborhood preservation (at 300d scale)** | Jaccard@10 at d=50: Standard=0.109, MRL-v4=0.173 (+58.8% relative) |
| **MRL does NOT debias embeddings** | WEAT converges at full dimension (Std=1.65, MRL=1.69 at d=300) |
| **MRL amplifies bias at low dimensions** | By preserving full-dim topology, MRL also preserves bias signal into truncated prefixes (WEAT at d=50: Std=0.29, MRL=1.33) |
| **MRL reorganizes which words carry bias** | Per-word gender projection correlation between models: r=−0.12 (p=0.45, not significant) |
| **MRL allocates more variance to the gender direction** | Gender variance as % of total: Std=0.30%, MRL=0.47% at d=300 |
| **MRL eliminates local neighborhood gender asymmetry** | k-NN asymmetry: Std=0.045, MRL=0.000 — bias moves from local to global encoding |

---

## Architecture

### Standard Skip-Gram (Baseline)

Standard Word2Vec with Negative Sampling. For each (center, context) pair within a window of 5:

```
L_NS = −log σ(w · c) − Σᵢ log σ(−w · nᵢ)     (K=10 negatives)
```

Produces a single embedding matrix `W ∈ ℝ^(V×D)`.

### MRL-v3: Naive Multi-Scale Loss

Applied independent NS loss at **each** prefix dimension and summed:

```
L_MRL-v3 = Σ_{d ∈ {50,100,...,300}} L_NS^(d)
```

**Problem:** The model could minimize prefix losses by inflating L2 norms (dot product grows with magnitude) rather than learning correct angular geometry. This caused **topological collapse** at low dimensions.

### MRL-v4: Relational Distillation (Current)

Applies NS loss **only at full dimension** and adds a **Relational Distillation** auxiliary loss for each prefix:

```
L_MRL-v4 = L_NS^(300) + 1/(|M|−1) Σ_{d<300} [α·KL(P_T || P_S) + β·MSE(cos_d, cos_300)]
```

**How it works per training step:**

1. **NS Loss (d=300):** Standard skip-gram on full-dimensional embeddings
2. **RD Loss (for each prefix d):**
   - Sample B=2048 unique center words from the batch
   - **Teacher:** Full 300d embeddings → projection head → L2-normalize → compute B×B cosine similarity matrix → softmax with temperature τ
   - **Student:** First d dimensions → separate projection head → L2-normalize → compute B×B similarity matrix → softmax
   - **KL Divergence:** Force student's similarity distribution to match teacher's
   - **Cosine MSE:** Direct angular constraint between prefix and full cosine similarities
3. **Gradient flow:** Teacher path is `.detach()`ed — gradients only flow to student prefix dims + projection heads. The full-dimensional embedding trains purely via NS loss.

```python
# Core of the RD loss (from mrl_v4_model.py)
t = self.teacher_proj[str(m)](vc_sub.detach())  # ← .detach() stops teacher gradients
t = F.normalize(t, dim=1)
t_sim = t @ t.T / self.tau                      # B×B teacher similarity
t_dist = F.log_softmax(t_sim, dim=1)

s = self.student_proj[str(m)](vc_sub[:, :m])     # student uses prefix
s = F.normalize(s, dim=1)
s_sim = s @ s.T / self.tau
s_dist = F.log_softmax(s_sim, dim=1)

kl = F.kl_div(s_dist, t_dist.detach(), log_target=True, reduction='batchmean')
```

**Projection heads** (2-layer MLP) are discarded after training — inference uses raw prefix embeddings directly.

---

## Dataset

- **Source:** [The Pile](https://pile.eleuther.ai/) — a large-scale, diverse, open-source language modeling corpus
- **Subsample:** 50 million tokens (deduplicated)
- **Vocabulary:** 175,523 unique words (minimum frequency threshold applied)
- **Coverage:** Academic papers, literature, web text, technical documentation
- **Shared:** Both Standard and MRL models train on the identical corpus and vocabulary

---

## Project Structure

```
MRL TESTS/
├── Scripts/
│   ├── mrl_v4_model.py              # MRL-v4 model with Relational Distillation
│   ├── geometric_analysis.py         # Exp 1-7: Core geometry (v3 embeddings)
│   ├── geometric_analysis_v2.py      # Exp 8-10: WCSS, Jaccard, SVD
│   ├── eval_300d.py                  # Full geometry + bias suite (300d)
│   ├── eval_768d.py                  # Full geometry + bias suite (768d)
│   ├── deep_bias_300d.py             # 15 deep-dive bias experiments
│   ├── evaluate_bias.py              # 7 standard bias metrics
│   ├── intrinsic_eval.py             # WordSim-353 evaluation
│   ├── eval_simlex999.py             # SimLex-999 evaluation
│   └── v4_jaccard_test.py            # Focused Jaccard test for v4
│
├── Notebooks/
│   ├── mrl-v2.ipynb                  # v2 training notebook
│   ├── mrl-v3.ipynb                  # v3/v4 training notebook
│   └── mrl-v3-cells/                 # Individual cells for assembly
│
├── Results_and_Reports/
│   ├── geometry_plots/               # Geometry experiment plots
│   ├── bias_plots_v4/                # Standard bias metric plots
│   ├── bias_results_v4.json          # Structured bias results
│   ├── v3_detailed_results.txt       # V3 geometry text output
│   └── v4_detailed_results.txt       # V4 geometry text output
│
├── Data_and_Embeddings/
│   └── _v3_embeddings_W2V/
│       └── Embeddings/
│           └── mrl_bias_v4_embeddings/
│               ├── plots/            # 300d geometry plots
│               ├── deep_bias_plots/  # 15 deep-dive bias plots
│               ├── report_300d.tex   # LaTeX report (300d)
│               ├── report_bias_deep.tex  # Deep bias analysis report
│               ├── bias_results.json
│               └── full_results.txt
│
├── latest/                            # 768d embeddings & results
│   ├── plots/                        # 768d geometry + bias plots
│   ├── report.tex                    # LaTeX report (768d)
│   └── bias_results.json
│
└── .gitignore
```

> **Note:** Embedding files (`.npy`, `.pkl`, `.zip`) are excluded from the repository due to size. Contact the author for access.

---

## Experiments

### Geometric Analysis (10 Experiments)

| # | Experiment | What It Measures |
|---|-----------|-----------------|
| 1 | **Mean Embedding Cosine Sim** | Angular alignment between model centroids at each prefix |
| 2 | **Cross Cosine Similarity** | How individual words relate to the other model's centroid |
| 3 | **PCA Explained Variance (Full Vocab)** | Information concentration in top principal components |
| 4 | **PCA by Frequency Bucket** | Information structure for high/mid/low frequency words |
| 5 | **Consecutive Chunk Angles** | Angular independence between adjacent dimension blocks |
| 6a | **Narrow Cone (Mimno & Thompson)** | Distribution of cosine similarities to mean vector |
| 6b | **Non-Negativity** | Fraction of positive values per dimension |
| 7 | **L2 Norm by Frequency Bucket** | Magnitude evolution across prefix dimensions |
| 8 | **WCSS (K-Means Inertia)** | Cluster tightness across prefixes |
| 9 | **Jaccard Neighborhood Preservation** | Top-K neighbor overlap between prefix and full dimension |
| 10 | **SVD Spectrum Decay** | Singular value distribution (low-rank structure) |

### Bias Evaluation

#### 7 Standard Metrics

| # | Metric | Source | What It Captures |
|---|--------|--------|-----------------|
| 1 | **WEAT** | Caliskan 2017 | Group-level associative bias (effect size) |
| 2 | **DirectBias** | Bolukbasi 2016 | Mean |cos(neutral word, gender direction)| |
| 3 | **RIPA** (mean/max) | Ethayarajh 2019 | Unnormalized projection (preserves magnitude) |
| 4 | **ECT** | Dev & Phillips 2019 | Rank correlation between gender centroid distances |
| 5 | **NBM** | Garg 2018 | Gender composition of k-nearest neighborhoods |
| 6 | **Cluster Purity** | Gonen & Goldberg 2019 | K-Means gender separability |
| 7 | **Bias Analogy** | Bolukbasi 2016 | Stereotypical analogy completions (he:she :: X:?) |

#### 15 Deep-Dive Experiments

| # | Experiment | Key Finding |
|---|-----------|-------------|
| A | Multi-category WEAT (Career/Family, Math/Arts, Science/Arts) | MRL shows consistently high WEAT even at d=50 |
| B | Gender direction stability across dims | MRL stabilizes gender direction from d=150 onward |
| C | Per-word gender projection profile | Completely uncorrelated between models (r=−0.12) |
| D | Bias projection heatmap (occupations × dims) | Visual confirmation of divergent bias layouts |
| E | Bias variance (spread of projections) | MRL has lower spread at d=50, converges at d=300 |
| F | Gender subspace dimensionality (PCA) | Identical between models (~21.5% on PC1) |
| G | k-NN gender asymmetry per occupation | Sparse signal — inconclusive with current word lists |
| H | Definitional pair cosine distances | MRL pairs are slightly more similar at mid-dims |
| I | Stereotype alignment score | Different stereotype gap trajectories across dims |
| J | Bias preservation correlation (d vs d=300) | MRL reaches r=0.75 at d=150; Std needs d=200 |
| K | Gender information content (% variance) | MRL allocates 56% more variance to gender direction |
| L | Inter-model bias correlation | Near-zero at all dims (|r| < 0.23) |
| M | Occupation clustering (K-Means) | No clear gender-based clustering in either model |
| N | Individual WEAT s(w) distributions | MRL has slightly more negative mean s(w) |
| O | Male vs female centroid distance differential | Moderate correlation between models (r=0.71) |

---

## Results

### Geometric Results

#### Neighborhood Preservation (300d)

| Dim | Standard | MRL-v4 | Δ |
|-----|----------|--------|---|
| 50  | 0.109 | **0.173** | +58.8% |
| 100 | 0.235 | **0.282** | +19.7% |
| 150 | 0.344 | **0.357** | +3.8% |
| 200 | **0.463** | 0.452 | −2.2% |
| 300 | 1.000 | 1.000 | — |

#### L2 Norm (Full Vocab, 300d)

| Dim | Standard | MRL-v4 |
|-----|----------|--------|
| 50  | 1.828 | 1.155 |
| 150 | 3.304 | 2.695 |
| 300 | 4.450 | 3.659 |

### Bias Results

#### WEAT Effect Sizes (Career/Family × Male/Female Names)

| Dim | Standard | MRL-v4 |
|-----|----------|--------|
| 50  | 0.287 | **1.328** |
| 100 | 1.244 | 1.555 |
| 150 | 1.281 | 1.472 |
| 200 | 1.390 | 1.664 |
| 300 | 1.653 | 1.694 |

> MRL-v4's WEAT is already large at d=50 because Relational Distillation preserves the full-dimensional association structure — including bias — into low-dimensional prefixes.

#### DirectBias & RIPA

| Dim | Std DirectBias | MRL DirectBias | Std RIPA Mean | MRL RIPA Mean |
|-----|---------------|---------------|--------------|--------------|
| 50  | 0.107 | 0.110 | 0.200 | 0.121 |
| 300 | 0.052 | 0.051 | 0.243 | 0.178 |

#### ECT (Embedding Coherence Test)

| Dim | Standard | MRL-v4 |
|-----|----------|--------|
| 50  | 0.648 | **0.776** |
| 150 | 0.679 | **0.907** |
| 300 | 0.732 | **0.794** |

> MRL-v4 shows consistently higher ECT, indicating more symmetric treatment of occupations w.r.t. gender centroids.

---

## How to Run

### Prerequisites

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Running Geometry + Bias Suite (300d)

```bash
python Scripts/eval_300d.py
```

Outputs: `plots/` directory with all geometry plots, `bias_results.json`, `full_results.txt`

### Running Deep Bias Investigation (300d)

```bash
python Scripts/deep_bias_300d.py
```

Outputs: `deep_bias_plots/` directory with 15 experiment plots, `deep_bias_results.txt`

### Running 768d Suite

```bash
python Scripts/eval_768d.py
```

### Intrinsic Evaluation

```bash
python Scripts/eval_simlex999.py
python Scripts/intrinsic_eval.py
```

> **Note:** Embedding `.npy` files must be placed in their expected directories. See the scripts for path configuration.

---

## Reports

LaTeX reports are included for compilation on Overleaf:

| Report | Location | Description |
|--------|----------|-------------|
| `report.tex` | `latest/` | Full 768d geometric + bias analysis |
| `report_300d.tex` | `mrl_bias_v4_embeddings/` | 300d analysis with cross-scale comparison |
| `report_bias_deep.tex` | `mrl_bias_v4_embeddings/` | Deep-dive bias investigation (7 key findings) |

---

## References

- Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS.
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases.* NeurIPS.
- Caliskan et al. (2017). *Semantics derived automatically from language corpora contain human-like biases.* Science.
- Bolukbasi et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker?* NeurIPS.
- Gonen & Goldberg (2019). *Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases.* NAACL.
- Dev & Phillips (2019). *Attenuating Bias in Word Vectors.* AISTATS.
- Ethayarajh et al. (2019). *Understanding Undesirable Word Embedding Associations.* ACL.
- Mimno & Thompson (2017). *The strange geometry of skip-gram with negative sampling.* EMNLP.

---

## License

This project is for academic research purposes.
