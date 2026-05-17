# experiments — Exploratory Models and Baselines

This folder documents the full experimental progression of the project, from early baseline algorithms to refined GRU architectures. None of these notebooks represent the final model; they exist to document the research process and justify design decisions made in the final implementation.

## Folder Structure

```
experiments/
├── GRU/                         # GRU architecture iterations (see GRU/README.md)
├── cnn_1D.ipynb                 # 1D Convolutional Neural Network baseline
├── xgboost.ipynb                # XGBoost baseline on aggregated features
└── injury_vs_no-injury.ipynb    # Initial binary classification exploration
```

## Notebooks Overview

| Notebook | Approach | Key Takeaway |
|---|---|---|
| `injury_vs_no-injury.ipynb` | Simple statistical separation | Established that injury events are rare (~5%) and highly imbalanced |
| `xgboost.ipynb` | Gradient boosting on static features | Good accuracy but poor sensitivity; ignores temporal dependencies |
| `cnn_1D.ipynb` | 1D CNN on time-series sequences | Limited by small dataset size; insufficient data for deep convolutional layers |
| `GRU/` | Recurrent architecture | Best sensitivity; handles temporal structure naturally |

## Progression Summary

The project evolved through three phases:

1. **Baseline (non-temporal):** `injury_vs_no-injury.ipynb` and `xgboost.ipynb` established a performance floor using static or aggregated features.
2. **Deep learning exploration:** `cnn_1D.ipynb` tested sequence-based convolution but revealed data-size limitations.
3. **Recurrent modeling:** The GRU family (see [`GRU/`](GRU/)) progressively improved sensitivity through architecture tuning, class-imbalance strategies, and window-length optimization.

The final model resulting from this process is in [`../src/gru_model_definitive.ipynb`](../src/gru_model_definitive.ipynb).
