# Methodology — Adapting Time Series Foundation Models for LSST Classification

This document describes our approach for classifying astronomical transients from
the LSST dataset (14 classes, 6 photometric bands, 36 timesteps, 2459 training
samples) using time series foundation models.

## Overview

Our method follows a two-stage pipeline:

1. **Embedding extraction**: Use a pre-trained time series foundation model as a
   frozen feature extractor to produce a fixed-dimensional representation of each
   multivariate light curve.
2. **Classical ML classification**: Train a gradient-boosted tree classifier
   (XGBoost) on the concatenation of these learned embeddings and hand-engineered
   statistical features.

This approach sidesteps the overfitting problem that plagues end-to-end
fine-tuning on small datasets: the foundation model's weights are never updated,
and the classifier operates on a tabular feature matrix where tree-based methods
are known to outperform neural networks.

## Choosing the embedding backbone

### The multivariate challenge

Most time series foundation models are designed for **univariate** data and
**mid-to-long** sequences. LSST presents two mismatches: its light curves are
**multivariate** (6 photometric bands observed simultaneously) and **short**
(36 timesteps). We evaluated three foundation models to find one that handles
both constraints.

### Models evaluated

**MOMENT** (Goswami et al., ICML 2024) is a general-purpose time series
foundation model pre-trained via masked time-series modeling. It requires a fixed
input length of 512 timesteps, so our 36-timestep light curves must be
zero-padded — producing inputs that are 93% padding. Moreover, MOMENT applies
Reversible Instance Normalization (RevIN), which subtracts the per-instance mean
and divides by the per-instance standard deviation. Our exploratory data analysis
revealed that amplitude is the single most discriminative feature for LSST
(varying by 200x across classes), and RevIN erases this information. Finally,
MOMENT treats each channel independently, missing cross-band correlations.
Best result: **W-F1 = 0.49**.

**TiReX** (Ansari et al., NeurIPS 2025 Workshop) is an xLSTM-based forecasting
model that handles short sequences more naturally (patch size of 32, no forced
512-padding). However, it is strictly **univariate**: each of the 6 bands is
embedded independently, producing 6 separate 512-dimensional vectors that are
concatenated. This means TiReX cannot capture inter-band temporal correlations
(e.g., band g peaking before band r), which are class-specific signatures in
LSST. Best result: **W-F1 = 0.55**.

We also explored using **AdaPTS** (Benechehab et al., ICML 2025) to adapt these
univariate models to multivariate data. AdaPTS learns lightweight adapters
(PCA or linear autoencoders) that project the 6 input channels into a lower-dimensional
latent space before feeding them to the foundation model. While conceptually
appealing, the adapted MOMENT and TiReX models did not substantially improve over
their unadapted counterparts, as the adapters cannot overcome the fundamental
limitations of these models.

**Chronos-2** (Ansari et al., 2025) is a patch-based forecasting foundation model
with **native multivariate support**. Its architecture alternates time-attention
layers (modeling temporal patterns within each channel) and group-attention layers
(modeling dependencies across channels). This means all 6 photometric bands are
processed jointly, and inter-band correlations are captured by the model's
attention mechanism. Unlike MOMENT, Chronos-2 preserves amplitude information
in its tokenization. Its patch size of 16 produces only
~5 tokens from 36 timesteps, avoiding the excessive padding problem. Best result
with simple logistic regression on embeddings: **W-F1 = 0.60**, nearly matching
MultiROCKET (W-F1 = 0.604), the strongest classical baseline.

We therefore selected Chronos-2 as our embedding backbone.

### Embedding extraction procedure

For each input sample of shape (6 channels, 36 timesteps), we run the frozen
Chronos-2 encoder, which produces a tensor of shape
(6 channels, $\sim$5 tokens, 768 hidden dimensions). We apply mean-pooling over
the token dimension to obtain a (6, 768) matrix, which is flattened into a
**4608-dimensional embedding vector**. This embedding captures both temporal
dynamics (via time-attention) and cross-channel relationships (via
group-attention) learned during Chronos-2's large-scale pre-training.

## Complementary handcrafted features

While foundation model embeddings capture rich temporal representations, they may
not explicitly encode simple statistical properties that are highly discriminative
for LSST. Our exploratory data analysis identified two key findings:

- **Amplitude** (peak-to-trough range) varies by over 200x across transient
  classes and is the single strongest discriminator.
- **Inter-band correlations** are class-specific: different transient types
  exhibit distinct patterns of correlation between photometric bands.

To make these properties directly available to the classifier, we compute 90
handcrafted features from the raw time series:

- **Per-channel statistics** (6 bands $\times$ 10 features = 60): mean, standard
  deviation, minimum, maximum, range, skewness, kurtosis, linear slope, first
  value, and last value.
- **Pairwise cross-channel correlations** ($\binom{6}{2}$ = 15): Pearson
  correlation coefficient between every pair of photometric bands.
- **Pairwise amplitude ratios** ($\binom{6}{2}$ = 15): log-scaled ratio of
  peak-to-trough amplitudes between every pair of bands, encoding relative
  brightness differences.

These 90 features are concatenated with the 4608 Chronos-2 embeddings, producing
a final feature vector of dimension **4698** per sample.

## Classification with XGBoost

We train an XGBoost gradient-boosted tree classifier on the concatenated feature
vectors. We compared logistic regression, random forest, and XGBoost on the same features
and found XGBoost to perform best. This is consistent with the well-known finding
that gradient-boosted trees outperform other classifiers on tabular data,
especially in low-data regimes.

### Handling class imbalance

The LSST dataset exhibits severe class imbalance (111:1 ratio between the largest
and smallest classes). We address this by computing balanced class weights using
scikit-learn's `compute_class_weight("balanced")`, which assigns weight
$w_c = \frac{n}{k \cdot n_c}$ to class $c$, where $n$ is the total number of
samples, $k$ is the number of classes, and $n_c$ is the number of samples in
class $c$. These weights are passed as per-sample weights to XGBoost during
training, effectively upweighting rare classes.

### Why not a neural classification head?

We experimented with MLP classification heads of various sizes (1.18M, 296K, and
64K parameters) trained on frozen Chronos-2 embeddings, as well as LP-FT
(Linear Probing then Fine-Tuning) with LoRA adapters. In all cases, the neural
heads underperformed simple logistic regression on the same embeddings. This is
consistent with the well-documented finding that on small tabular datasets,
classical ML methods outperform neural networks. With only 2459 training samples
spread across 14 classes, the effective per-class sample size is too small for
gradient-based training to generalize well.

## Hyperparameters

| Parameter         | Value |
|-------------------|-------|
| `n_estimators`    | 500   |
| `max_depth`       | 6     |
| `learning_rate`   | 0.1   |
| `subsample`       | 0.8   |
| `colsample_bytree`| 0.8   |
| `tree_method`     | hist  |

## Results

| Model                                    | Accuracy | Weighted F1 | Macro F1 |
|------------------------------------------|----------|-------------|----------|
| MultiROCKET (best classical baseline)    | 0.638    | 0.604       | 0.462    |
| Chronos-2 + LogReg (embeddings only)     | 0.600    | 0.598       | 0.461    |
| **Chronos-2 + HC features + XGBoost (ours)** | **0.710**  | **0.680**   | **0.548**  |

Our method improves over the MultiROCKET baseline by **+12.5% in Weighted F1**
and **+18.7% in Macro F1**, with the Macro F1 gain reflecting substantially
better performance on rare classes.
