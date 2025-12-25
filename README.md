# TimeSeries Forecasting — End-to-End (SARIMAX + Baselines + LSTM)

This repository is a clean, reproducible **time series forecasting pipeline** that compares:
- **Naive baselines**
- **SARIMAX (seasonal ARIMA)** via `statsmodels`
- **LSTM** via `PyTorch`

It is designed to run end-to-end out-of-the-box (includes a small synthetic dataset).

## Quickstart
```bash
pip install -r requirements.txt
python -m src.run_all
```

## Repo structure
- `data/raw/` — dataset (CSV)
- `src/` — reusable pipeline code
- `notebooks/` — step-by-step notebooks
- `figures/` — generated plots
- `results/` — metrics table

## Dataset
`data/raw/daily_energy_synthetic.csv` contains 3 years of daily values with weekly + yearly seasonality and mild trend.
To use your own dataset, replace the CSV and keep the same columns: `date`, `value`.

## Outputs
- `results/metrics.csv`
- `figures/01_series.png`
- `figures/02_forecasts.png`
- `figures/03_metrics_MAE.png`, `...RMSE.png`, `...MAPE.png`
