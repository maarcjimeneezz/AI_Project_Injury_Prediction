# Dataset

This folder contains the original datasets used throughout the project. **Do not move, rename, or modify these files.** All notebooks reference them from this fixed path.

## Files

| File | Description |
|---|---|
| `day_approach_maskedID_timeseries.csv` | Main dataset. Daily training metrics for 74 competitive runners, formatted as time series. Used for all GRU experiments. |
| `week_approach_maskedID_timeseries.csv` | Weekly-aggregated version of the same dataset. Used in early baseline experiments. |
| `DATASET_EXPLANATION.md` | Authoritative feature dictionary from the original dataset authors. Read this before working with any notebook. |

## Source and Citation

> Lövdal, S.S., Azzopardi, G., den Hartigh, R.J.R. (2021). *Injury Prediction in Competitive Runners With Machine Learning*. International Journal of Sports Physiology and Performance.

Original data: [Kaggle — Injury Prediction for Competitive Runners](https://www.kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners)

## Data Format Notes

- Each row represents one event day (potential injury or no-injury) for one athlete.
- The target column is binary: `1` = injury, `0` = no injury.
- Athlete IDs are masked for privacy.
- For the day-approach dataset, feature suffixes (`.6` through `.0`) indicate days before the event (`.6` = 1 day before, `.0` = 7 days before). See `DATASET_EXPLANATION.md` for the full feature dictionary.
