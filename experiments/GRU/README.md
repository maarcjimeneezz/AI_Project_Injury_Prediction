# experiments/GRU — GRU Architecture Iterations

This folder contains all intermediate GRU experiments conducted during development. Each notebook represents a distinct experimental configuration or hypothesis tested during the research process.

## Notebooks (Chronological Order)

| Notebook | Description |
|---|---|
| `GRU_model_v1.ipynb` | First prototype. Heavy focus on data preprocessing; baseline architecture with limited tuning. |
| `GRU_model_v2.ipynb` | Second iteration with improved sequence construction and proper train/val/test split. |
| `gru_model.ipynb` | Restructured implementation. Cleaner pipeline and modular code. |
| `gru_model_balanced.ipynb` | Introduced class-balancing strategies (undersampling, oversampling). Evaluated impact on sensitivity. |
| `gru_model_weight.ipynb` | Tested weighted loss function (BCEWithLogitsLoss with pos_weight) as an alternative to resampling. |
| `gru_14day.ipynb` | Extended sequence window to 14 days. Observed improvement in capturing pre-injury build-up patterns. |

## Supporting Files

| File / Folder | Description |
|---|---|
| `gru.py` | Standalone Python module with the GRU model class and utility functions extracted from notebooks. |
| `saved_models/` | Named `.pt` checkpoint files corresponding to key experimental configurations. |
| `models/` | Hyperparameter search checkpoints: grid over sequence length (13–16) and hidden size (13–32), multiple random seeds. |
| `results/gru_results.csv` | Aggregated metrics (accuracy, sensitivity, specificity, F1) across all hyperparameter combinations. |

## Saved Model Naming Convention

Files in `saved_models/` follow this pattern:

```
GRU_{description}_{key_params}.pt
```

For example, `GRU_best_model_seq14_weighted_sampler.pt` is the best model found with a 14-day sequence using weighted random sampling.

## Key Findings

- **Sequence length 14** consistently outperformed 7-day windows for injury sensitivity.
- **Weighted random sampling** was more stable than loss-based weighting for this dataset size.
- **Hidden size 14** gave the best balance between model capacity and overfitting on the small dataset.
- Sensitivity (recall on the injury class) was used as the primary optimization criterion throughout, as missed injuries carry higher cost than false alarms.

These findings directly informed the design of the final model in [`../../src/gru_model_definitive.ipynb`](../../src/gru_model_definitive.ipynb).
