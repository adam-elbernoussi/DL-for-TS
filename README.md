# Deep Learning for Time Series Classification — LSST

Multivariate time series classification of astronomical transients from the
**LSST** dataset (14 classes, 6 photometric bands, 36 timesteps, 2459 train / 2466 test samples).
The dataset is heavily imbalanced (111:1 ratio between largest and smallest class).

## Results

| Model                            | Accuracy | Weighted F1 | Macro F1 |
|----------------------------------|----------|-------------|----------|
| Catch22 + RF                     | 0.600    | 0.540       | 0.310    |
| MultiROCKET                      | 0.638    | 0.604       | 0.462    |
| **Chronos-2 + HC + XGBoost**     | **0.710**| **0.680**   | **0.548**|

Best model: frozen Chronos-2 embeddings (4608-dim) + 90 handcrafted statistical
features + XGBoost with balanced sample weights. See `methodology.md` for details.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `baselines.ipynb` | Catch22 + Random Forest, MultiROCKET |
| `ablations.ipynb` | Foundation model comparisons (MOMENT, TiReX, Chronos-2), adaptation strategies, feature/classifier ablations |
| `best_solution.ipynb` | Final pipeline: Chronos-2 embeddings + handcrafted features + XGBoost |

## Setup

```bash
pip install -r requirements.txt
```
