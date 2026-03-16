# Deep Learning for Time Series Classification — LSST

Multivariate time series classification of astronomical transients from the
**LSST** (Legacy Survey of Space and Time) dataset using classical baselines,
time series foundation models, and embedding-based feature engineering.

## Dataset

| Property       | Value                          |
|----------------|--------------------------------|
| Source         | UCR/UEA archive (via tslearn)  |
| Samples        | 2 459 train / 2 466 test       |
| Channels       | 6 (photometric bands: u,g,r,i,z,y) |
| Timesteps      | 36                             |
| Classes        | 14 astronomical transient types |

The dataset is **heavily imbalanced** (class 90 has 777 test samples, class 53
has 7). All trainable models use class-weighted cross-entropy loss and balanced
class weights.

## Results

### Baselines & Foundation Models (notebook 02, 03)

| Model                       | Accuracy | Weighted F1 | Macro F1 |
|-----------------------------|----------|-------------|----------|
| Catch22 + RF (B1)           | 0.6000   | 0.5400      | 0.3100   |
| MultiROCKET (B2)            | 0.6375   | 0.6039      | 0.4615   |
| 1D-CNN (B3)                 | 0.2737   | 0.3014      | 0.3743   |
| MOMENT ZS + LogReg (M1)    | 0.4757   | 0.4880      | 0.3460   |
| MOMENT Head-FT (M2)        | 0.4862   | 0.4917      | 0.3443   |
| MOMENT Full-FT (M3)        | 0.4692   | 0.4876      | 0.3867   |
| TiReX ZS + LogReg (M4)     | 0.5580   | 0.5464      | 0.3788   |
| TiReX Head-FT (M5)         | 0.5373   | 0.5262      | 0.3968   |
| Chronos-2 ZS + LogReg (M7) | 0.5998   | 0.5984      | 0.4605   |
| Chronos-2 Head-FT (M8)     | 0.5766   | 0.5765      | 0.4632   |
| Chronos-2 LP-FT LoRA (M9)  | 0.6002   | 0.5946      | 0.4775   |

### Chronos-2 Embedding Maximization (notebook 05)

| Model                            | Accuracy | Weighted F1 | Macro F1 |
|----------------------------------|----------|-------------|----------|
| Chronos mean + LogReg            | 0.6018   | 0.5933      | 0.4247   |
| Chronos mean+max + LogReg        | 0.6172   | 0.6019      | 0.4266   |
| **Chronos mean+hc + XGBoost**    | **0.7097** | **0.6796** | **0.5480** |
| Chronos stats+hc + XGBoost       | 0.7011   | 0.6664      | 0.5330   |
| Tuned XGBoost (Optuna)           | 0.7024   | 0.6748      | 0.5442   |
| Fused 3-FM + XGBoost             | 0.7056   | 0.6730      | 0.5502   |
| Stacking Ensemble                | 0.6148   | 0.6399      | 0.5559   |
| Grand Ensemble (5 models)        | 0.7076   | 0.6814      | 0.5565   |

**Best single model: Chronos-2 mean+hc + XGBoost (W-F1 = 0.680, +12.5% over MultiROCKET)**

## Key Findings

### Why Chronos-2 + handcrafted features + XGBoost wins

The best model concatenates two complementary feature sources:

1. **Chronos-2 mean-pooled embeddings (4608-dim)**: Deep temporal representations
   extracted via `pipeline.embed()`. Chronos-2's cross-channel attention captures
   inter-band correlations natively — unlike MOMENT and TiReX which treat channels
   independently.

2. **Handcrafted statistical features (90-dim)**: Per-channel statistics (mean, std,
   min, max, range, skew, kurtosis, slope, first/last value) + pairwise cross-channel
   Pearson correlations (15) + pairwise amplitude ratios (15). These directly encode
   what EDA found most discriminative — amplitude spread (200x across classes) and
   inter-band correlation patterns.

3. **XGBoost with balanced sample weights**: 500 trees, max_depth=6. Tree-based
   models can split directly on handcrafted features ("is amplitude > X?") while
   also leveraging high-dimensional embeddings via `colsample_bytree`. Sample weights
   from `compute_class_weight("balanced")` handle 111:1 class imbalance.

The embeddings alone (W-F1=0.593) or handcrafted features alone wouldn't achieve
this — the combination (4608+90=4698 dim) gives XGBoost the best of both worlds.

### Why foundation models struggle on LSST

1. **MOMENT (W-F1 ~0.49)**: Pads 36 → 512 timesteps (93% padding). RevIN
   normalization destroys class-discriminative amplitude information.
2. **TiReX (W-F1 ~0.55)**: Univariate only — misses cross-channel correlations.
   Better than MOMENT due to native short-series handling.
3. **Chronos-2 (W-F1 ~0.60)**: Best FM. Native multivariate via cross-channel
   attention, no RevIN, flexible length. Still below MultiROCKET as a standalone
   classifier, but its embeddings are powerful feature extractors for XGBoost.

### AdaPTS adapters (notebook 03, M10)

[AdaPTS](https://github.com/abenechehab/AdaPTS) (ICML 2025) provides learned
adapters that project multivariate inputs into a latent space for univariate FMs.
We tested PCA and LinearAE adapters (6 → 2 channels) with MOMENT and TiReX.
Results were mixed — the adapters help MOMENT see cross-channel info but don't
overcome its fundamental padding/RevIN limitations.

### LP-FT with LoRA (notebook 03, M9)

Linear Probing then Fine-Tuning with Low-Rank Adaptation on Chronos-2:
- Stage 1 (LP): Freeze backbone, train linear head only (64K params, lr=1e-3)
- Stage 2 (FT): Unfreeze LoRA rank-4 adapters on q/v projections, fine-tune
  LoRA+head jointly (360K params, lr=1e-4)
- Result: W-F1=0.595, Macro-F1=0.478 (best macro across all single FM models)
- LoRA improves minority class recall at slight cost to majority class accuracy

## Project Structure

```
DL-for-TS/
├── src/
│   ├── data_loader.py        # Data loading, padding, standardization, splits
│   ├── baselines.py          # B2 (MultiROCKET) and B3 (1D-CNN)
│   ├── moment_pipeline.py    # M1-M3: MOMENT embedding, head training, fine-tuning
│   ├── tirex_pipeline.py     # M4-M6: TiReX embedding, head training, fine-tuning
│   ├── chronos_pipeline.py   # M7-M9: Chronos-2 embedding, LP-FT with LoRA
│   └── metrics.py            # Evaluation utilities, confusion matrices, results table
├── 01_eda.ipynb              # Exploratory data analysis
├── 02_baselines.ipynb        # B1-B3 baseline experiments
├── 03_foundation_model.ipynb # M1-M10 foundation model experiments
├── 04_analysis.ipynb         # UMAP, results comparison (WIP)
├── 05_chronos_maxout.ipynb   # Chronos-2 embedding maximization
├── figures/
│   ├── moment_confusion_matrices.pdf
│   ├── chronos_maxout_confmat.pdf
│   └── ...
└── requirements.txt
```

## Models

### Baselines

- **B1 — Catch22 + Random Forest**: 22 canonical time series features per channel
  (132 total), fed to a balanced RF. Simple, interpretable baseline.
- **B2 — MultiROCKET**: 10 000 random convolutional kernels with MiniRocket + ROCKET
  features. State-of-the-art for time series classification without deep learning.
- **B3 — 1D-CNN**: 3-layer Conv1D (64 → 128 → 256 filters) + global average pooling.
  Overfits on this small dataset.

### Foundation Models

- **MOMENT** (ICML 2024): Masked time-series modeling, d_model=512, patch_len=8.
- **TiReX** (NeurIPS 2025 Workshop): xLSTM-based, hidden_dim=512, patch_size=32. Univariate.
- **Chronos-2** (Amazon, 2025): Patch-based with native multivariate support via
  alternating time-attention and group-attention. d_model=768, 12 layers, patch_size=16.

### Embedding Maximization Pipeline (notebook 05)

1. **Rich embedding extraction**: Mean-pool, max-pool, std-pool from Chronos-2
2. **Handcrafted features**: 90 statistical + cross-channel features from raw data
3. **Classifier shootout**: 5 feature sets × 3 classifiers (LogReg, XGBoost, RF)
4. **Stacking ensemble**: 5 diverse base models → LogReg meta-learner
5. **Optuna tuning**: 100-trial Bayesian optimization on XGBoost hyperparameters
6. **Multi-FM fusion**: Chronos-2 + MOMENT + TiReX embeddings concatenated (10842-dim)
7. **Grand ensemble**: Weighted soft-vote of 5 best models

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install chronos-forecasting peft  # Chronos-2 + LoRA
pip install tirex-ts                  # TiReX
pip install xgboost optuna            # Embedding maximization
pip install adapts                    # AdaPTS adapters
```
