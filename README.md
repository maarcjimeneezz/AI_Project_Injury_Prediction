# Sports Injury Prediction in Competitive Runners Using GRU Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Final project for the Artificial Intelligence course, 4th year — Universitat de Barcelona.

This project investigates whether temporal patterns in a competitive runner's training load can predict the onset of injury within a 14-day horizon. The core model is a **Gated Recurrent Unit (GRU)** neural network trained on sequences of daily training metrics including distance, intensity zones, perceived exertion, and perceived recovery.

> **Dataset reference:** Lövdal, S.S., Azzopardi, G., den Hartigh, R.J.R. (2021). *Injury Prediction in Competitive Runners With Machine Learning*. International Journal of Sports Physiology and Performance.

---

## Main Source File

> **`src/gru_model_definitive.ipynb`** — The definitive model. Start here.

This is the final, production-quality implementation of the GRU-based injury predictor. It contains the complete training pipeline, hyperparameter selection, class imbalance handling, and evaluation with full metrics.

---

## Repository Structure

```
.
├── src/                        # Main source code (final model)
│   └── gru_model_definitive.ipynb
├── experiments/                # Experimental notebooks and baselines
│   ├── GRU/                    # GRU architecture iterations
│   ├── cnn_1D.ipynb
│   ├── xgboost.ipynb
│   └── injury_vs_no-injury.ipynb
├── data_exploration/           # Exploratory Data Analysis notebooks
└── Dataset/                    # Original datasets (read-only)
```

---

## Quickstart

### Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter
```

Developed and tested with Python 3.9+, PyTorch 2.x.

### Run the main model

```bash
jupyter notebook src/gru_model_definitive.ipynb
```

The notebook expects the dataset files to be present at:
- `Dataset/day_approach_maskedID_timeseries.csv`
- `Dataset/week_approach_maskedID_timeseries.csv`

No additional setup is required. All preprocessing steps are self-contained within the notebook.

---

## Methodology Summary

| Stage | Description |
|---|---|
| Data source | Lövdal et al. (2021), day-approach time series, 74 competitive runners |
| Sequence length | 14 days (lookback window) |
| Features per day | 10 (km, zone breakdown, strength, cross-training, perceived metrics) |
| Target | Binary: injury within the next day (1) / no injury (0) |
| Class imbalance | Weighted random sampler (injury events ~5% of dataset) |
| Architecture | 1-layer GRU, hidden size 14, fully connected output head |
| Optimization | Adam optimizer, BCEWithLogitsLoss |
| Evaluation | Sensitivity-prioritized (minimizing missed injuries) |

---

## Results Summary

The final GRU model achieves competitive sensitivity on the held-out test set, outperforming baseline approaches (XGBoost, 1D-CNN) in recall for the minority injury class — the clinically most relevant metric. Missed injuries carry a higher cost than false alarms in this domain.

Full metrics, confusion matrices, and ROC curves are available in [`src/gru_model_definitive.ipynb`](src/gru_model_definitive.ipynb).

---

## Experimental Progression

Earlier model versions and alternative approaches are documented in [`experiments/`](experiments/). See [`experiments/README.md`](experiments/README.md) for a progression map from baseline to final model.

Exploratory data analysis that motivated design decisions is in [`data_exploration/`](data_exploration/).

---

## Authors

- Marc Jimenez ([@maarcjimeneezz](https://github.com/maarcjimeneezz))
- Roc Ferrer
- David Garcia

---

## License

This repository is for academic purposes. The dataset is the property of the original authors (Lövdal et al., 2021) and is used here under their published terms. Model code is available for educational use.
