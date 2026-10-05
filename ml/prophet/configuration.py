"""
ml/prophet/configuration.py
────────────────────────────
Prophet Hyperparameter Configuration

Centralises all Prophet parameter specifications:
  - Default parameters
  - Hyperparameter search space
  - Documentation of each parameter and its rationale

IMPORTANT: Hyperparameter tuning must ONLY use the training partition.
           The test set must NEVER influence parameter selection.
"""

from typing import Dict, Any, List
from ml.configs.config import PROPHET_DEFAULT_PARAMS, PARAM_GRID, HOLIDAY_COUNTRY

# Re-export config values for convenience
DEFAULT_PARAMS: Dict[str, Any] = PROPHET_DEFAULT_PARAMS
SEARCH_SPACE: List[Dict[str, Any]] = PARAM_GRID

# Parameter documentation
PARAMETER_DOCS: Dict[str, str] = {
    "changepoint_prior_scale": (
        "Controls the flexibility of the trend. "
        "Higher values allow more rapid trend changes (more flexible, risk of overfitting). "
        "Lower values produce smoother trends. "
        "Typical range: 0.001 – 0.5. Default: 0.05."
    ),
    "seasonality_prior_scale": (
        "Controls the flexibility of the seasonality components. "
        "Higher values allow larger seasonal swings. "
        "Typical range: 0.01 – 10. Default: 10.0."
    ),
    "holidays_prior_scale": (
        "Controls the flexibility of holiday effects. "
        "Higher values allow larger holiday deviations from the baseline. "
        "Typical range: 0.01 – 10. Default: 10.0."
    ),
    "seasonality_mode": (
        "Additive: seasonality is added to the trend (appropriate when seasonal "
        "variation is roughly constant over time). "
        "Multiplicative: seasonality scales with the trend."
    ),
    "yearly_seasonality": (
        "Enable/disable yearly seasonality. Set to True if ≥2 years of training data. "
        "If enabled, Prophet uses Fourier series with order 10 by default."
    ),
    "weekly_seasonality": (
        "Enable/disable weekly seasonality (7-day pattern). "
        "Always enabled for daily energy demand forecasting. "
        "Uses Fourier series with order 3."
    ),
    "daily_seasonality": (
        "Disabled for daily/weekly granularity. Only relevant for sub-daily data."
    ),
    "interval_width": (
        "Width of the prediction uncertainty interval. "
        "0.95 = 95% prediction interval (yhat_lower, yhat_upper). "
        "Wider intervals are more conservative."
    ),
    "n_changepoints": (
        "Maximum number of potential changepoints. "
        "Prophet selects from these using an L1 penalty. "
        "Default: 25. Reduce if overfitting."
    ),
    "changepoint_range": (
        "Proportion of the training series where changepoints can be placed. "
        "Default: 0.8 (changepoints only in first 80% of training history). "
        "Prevents overfitting to the end of training."
    ),
}

# Holiday calendar documentation
HOLIDAY_CALENDAR_DOC = {
    "country": HOLIDAY_COUNTRY,
    "region": "Germany (DE) — Federal national holidays",
    "source": "Prophet built-in via holidays library (country_name='DE')",
    "includes": [
        "New Year's Day (Neujahrstag)",
        "Good Friday (Karfreitag)",
        "Easter Monday (Ostermontag)",
        "Labour Day (Tag der Arbeit)",
        "Ascension Day (Christi Himmelfahrt)",
        "Whit Monday (Pfingstmontag)",
        "German Unity Day (Tag der Deutschen Einheit)",
        "Christmas Day (1. Weihnachtstag)",
        "Boxing Day (2. Weihnachtstag)",
    ],
    "version": "via prophet.add_country_holidays(country_name='DE')",
    "note": (
        "Regional state holidays (e.g. Bavaria, NRW) are NOT included. "
        "The model uses federal-level holidays only. "
        "If the wrong region is selected, holiday effects may be inaccurate. "
        "This is a known limitation documented in the Model Card."
    ),
}


def build_prophet_params(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build a full Prophet parameter dict from defaults + optional overrides.

    Parameters
    ----------
    overrides : Optional dict of parameter overrides.

    Returns
    -------
    dict — Complete parameter dict safe to pass to Prophet(**params).
    """
    params = DEFAULT_PARAMS.copy()
    if overrides:
        params.update(overrides)
    return params
