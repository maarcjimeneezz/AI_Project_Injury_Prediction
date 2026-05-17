# data_exploration — Exploratory Data Analysis

This folder contains all EDA notebooks produced before model development. The analyses here informed key modeling decisions: feature selection, class imbalance strategy, sequence window length, and the choice of a temporal (recurrent) architecture.

## Notebooks

| Notebook | Description |
|---|---|
| `Initial_EDA.ipynb` | Overview of dataset structure. Spaghetti plots of training load per athlete. Identifies missing values and data distribution per runner. |
| `class_imbalance_analysis.ipynb` | Quantifies the injury/no-injury class ratio (~5% positive). Tests resampling strategies and their effect on class distribution. |
| `data_indiv_analysis.ipynb` | Per-athlete breakdown: training volume, injury frequency, and behavioral patterns. Reveals high inter-athlete variability. |
| `intensity_zones_analysis.ipynb` | Deep dive into heart-rate intensity zones (Z3-4, Z5-T1-T2). Examines the relationship between high-intensity load and injury events. |
| `perceived_metrics_analysis.ipynb` | Analysis of subjective variables: perceived exertion, training success, and recovery ratings. Confirms their predictive relevance. |
| `temporal_window_analysis.ipynb` | Investigates how far back training patterns predict injury. Supports the choice of a 14-day lookback window. |

## Key Findings

- The dataset is severely imbalanced: ~95% non-injury, ~5% injury events. Any model ignoring this will optimize for accuracy at the expense of sensitivity.
- High-intensity zone kilometers (Z5-T1-T2) in the days preceding injury show a detectable spike pattern.
- Perceived recovery scores drop systematically in the 3–7 days before injury.
- A 14-day lookback window captures more of the pre-injury signal than a 7-day window.
- Inter-athlete variability is high, motivating population-level modeling rather than per-athlete models.

These findings are cross-referenced in [`../src/gru_model_definitive.ipynb`](../src/gru_model_definitive.ipynb).
