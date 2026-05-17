# src — Main Source Code

This folder contains the definitive implementation of the GRU-based injury prediction model. It is the primary deliverable of the project.

## Contents

| File | Description |
|---|---|
| `gru_model_definitive.ipynb` | **Main model notebook.** Final, production-quality GRU implementation with full training pipeline, evaluation, and analysis. |

## What the Notebook Does

1. Loads the day-approach time series dataset from `../Dataset/`.
2. Constructs 14-day sliding-window sequences for each athlete.
3. Handles class imbalance with a weighted random sampler.
4. Defines a single-layer GRU with a linear classification head.
5. Trains the model with the Adam optimizer and binary cross-entropy loss.
6. Evaluates performance with accuracy, sensitivity, specificity, F1-score, and ROC-AUC.
7. Saves the best checkpoint based on validation sensitivity.

## How to Run

```bash
jupyter notebook gru_model_definitive.ipynb
```

Ensure the `Dataset/` folder exists at the repository root with both CSV files before running. All cells should execute top to bottom without modification.
