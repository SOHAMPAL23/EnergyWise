# EnergyWise — Energy Demand Forecasting (OPSD Germany, Prophet)

Leakage-safe forecasting pipeline for German hourly electricity load (Open Power System Data),
with daily and weekly Prophet models, Naive / Seasonal-Naive baselines, training-only
time-series cross-validation, hyperparameter tuning, prediction-interval anomaly signals,
versioned artifacts and an evidence-based audit.

> Every number below is copied from files produced by `python -m ml.pipeline --task all`
> (`reports/evaluation/final_metrics_*.json`, `ml/artifacts/metadata/tuning_*.json`).
> Re-run the pipeline to regenerate them. Do not edit them by hand.

## Data

| | |
|---|---|
| Source | OPSD *Time series* 2020-10-06, column `DE_load_actual_entsoe_transparency` (ENTSO-E) |
| Raw | 50,401 hourly rows, 2014-12-31 23:00 UTC → 2020-09-30 23:00 UTC |
| Unit | **MW** (mean load over the period). Not GWh. |
| Daily series | 2,100 complete days, 2015-01-01 → 2020-09-30 (the 1-hour partial day 2014-12-31 is excluded) |
| Weekly series | 299 complete Monday–Sunday weeks, 2015-01-05 → 2020-09-21 |
| Raw file SHA-256 | `6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8` |

The raw CSV is git-ignored (130 MB). Download it from
`https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv`
into `ml/data/opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/` and verify the hash.

## Split — 70 % train / 30 % test, chronological

| | Daily | Weekly |
|---|---|---|
| Train | 1,470 days, 2015-01-01 → 2019-01-09 | 209 weeks, 2015-01-05 → 2018-12-31 |
| Test (untouched) | 630 days, 2019-01-10 → 2020-09-30 | 90 weeks, 2019-01-07 → 2020-09-21 |

The test period contains the **2020 COVID-19 demand shock**.

## Leakage controls

1. **`HoldoutVault`** (`ml/validation_models/guard.py`) wraps the test partition. EDA, CV, tuning and final
   fitting only receive training frames; the vault opens only for `final_evaluation`, logs every access and is
   SHA-256 fingerprinted before/after.
2. **`AccessAuditor`** wraps `Prophet.fit/predict` and records the date range of every frame the model sees; it
   raises if any row at/after the test boundary arrives. Tuning observed 560 fits + 560 predictions per
   granularity, max date touched = last training day.
3. **Counterfactual test** (`ml/tests/test_cv_leakage.py`, `ml/audit.py`): all test values are replaced by
   noise, the split → CV → tuning chain is re-run, and every fold metric and the selected parameters are
   bit-identical.
4. EDA runs on the training partition only.

## Cross-validation and tuning (training partition only)

* Rolling-origin / expanding window: train window always starts at row 0; 7 folds; daily: min train 730 d,
  horizon 180 d, step 90 d; weekly: 104 / 26 / 13 weeks.
* Grid (frozen before tuning): `changepoint_prior_scale` ∈ {0.005, 0.01, 0.05, 0.1, 0.5} × `seasonality_prior_scale` ∈ {1, 10}
  × `holidays_prior_scale` ∈ {1, 10} × (`n_changepoints`, `changepoint_range`) ∈ {(25,.8), (25,.9), (10,.8), (40,.8)} = 80 configs.
* Selection: lowest mean validation MAE (tie → RMSE). Every fold of every config is logged to
  `reports/experiments/experiments.{csv,json}` with `experiment_id, model_version, granularity, parameters,
  training_range, validation_range, MAE, RMSE, timestamp` (+ run_id, config_id, fold, coverage, seed).

| Granularity | Selected parameters | CV MAE (default → tuned) | CV RMSE (tuned) |
|---|---|---:|---:|
| Daily | cps 0.005, sps 10, hps 1, 25 cp, range 0.9 | 1,830.5 → 1,398.2 | 1,916.1 |
| Weekly | cps 0.01, sps 1, hps 1, 10 cp, range 0.8 | 1,765.9 → 1,502.8 | 1,970.0 |

The daily optimum sits at the lower edge of the `changepoint_prior_scale` grid (0.005).

## Hold-out results (fixed-origin forecast of the entire test period; no refitting)

**Daily (630 days)**

| Model | MAE (MW) | RMSE (MW) |
|---|---:|---:|
| Naive | 12,228.5 | 13,965.8 |
| Seasonal-naive (7 d) | 6,152.3 | 7,508.9 |
| **Prophet** | **3,239.6** | **4,051.1** |

Weekly-aggregated daily forecasts (89 complete weeks): Prophet MAE 3,159.8 / RMSE 3,837.4; Seasonal-naive 5,879.3 / 6,981.1; Naive 12,268.5 / 12,978.2.

**Weekly (90 weeks)**

| Model | MAE (MW) | RMSE (MW) |
|---|---:|---:|
| Naive | 3,649.7 | 4,382.4 |
| **Seasonal-naive (52 w)** | **2,524.2** | **3,199.2** |
| Prophet | 3,142.7 | 3,857.8 |

**Weekly Prophet does not beat the 52-week seasonal-naive baseline.**

Interval calibration: nominal 95 % → empirical test coverage **60.8 %** (daily) and **48.9 %** (weekly). Worst months for
Prophet are April–July and September 2020 (mean error ≈ −4,300 … −7,000 MW, i.e. the model over-forecasts demand
during the COVID-19 slump). See `reports/evaluation/evaluation_report_*.md`.

## Anomaly signals

`actual < lower → DROP_ANOMALY`, `actual > upper → SPIKE_ANOMALY`, otherwise `NORMAL`, with
`lower = ŷ − s·(ŷ − ŷ_lower)`, `upper = ŷ + s·(ŷ_upper − ŷ)`; `s` = `--sensitivity` (default 1.0 = raw 95 % interval).
Each record carries timestamp, actual demand, lower/upper bound, deviation magnitude and classification.

At s = 1.0: daily 247 drops / 0 spikes (39.2 % of days); weekly 45 drops / 1 spike. These flags are
**investigation signals, not ground truth**: no labelled anomalies exist (so no precision/recall is reported) and the
alert rate is dominated by forecast drift over a 630-day fixed-origin horizon and COVID-19.

## Artifacts (`ml/artifacts/`, git-ignored)

* `models/prophet-{daily,weekly}-v2.json` + `.meta.json` (training range, params, holidays, changepoints, tuning
  reference, data hash, seed, environment, git commit — no test metrics)
* `metrics/final_metrics_*.json`, `metrics/anomaly_summary_*.json`
* `forecasts/final_predictions_*.csv`, `anomaly_detections_*.csv`, `anomalies_*.json`
* `metadata/tuning_*.json`, `manifest.json` (SHA-256 of every file; consumers should read this first)
* `*-v1*` / `prophet_daily_v1.json` are **stale** (pre-fix) — do not consume.

## Run

```powershell
pip install -r requirements.txt
python -m ml.pipeline --task all                 # ~7 min: profile → validate → preprocess → EDA → tune → train → evaluate
python -m ml.pipeline --task audit               # evidence-based audit → reports/audit/ml_pipeline_audit.md
python -m ml.pipeline --task anomalies --sensitivity 1.5
python -m pytest                                 # unit + leakage tests
```

```python
from prophet.serialize import model_from_json
m = model_from_json(open("ml/artifacts/models/prophet-daily-v2.json", encoding="utf-8").read())
future = m.make_future_dataframe(periods=30, freq="D")
m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

## Limitations

* National German holidays only; no weather regressors (planned).
* COVID-19 lies in the test window; no 2015–2019 model can anticipate it.
* Prophet intervals are under-dispersed here (see coverage); do not present them as calibrated 95 % probabilities.
* FastAPI backend / dashboard are **not implemented** in this repository yet.

## Audit

`reports/audit/ml_pipeline_audit.md` lists PASS / WARNING / FAIL for the 20 pipeline requirements, the
test-leakage proof, and the fixes required before the backend consumes the artifacts.

## References

Open Power System Data, *Time series* 2020-10-06, https://data.open-power-system-data.org/time_series/ ·
Taylor & Letham (2018), *Forecasting at scale*, The American Statistician 72(1).
