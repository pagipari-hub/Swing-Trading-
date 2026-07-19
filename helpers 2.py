# ============================================================
# helpers.py — Small pure utility functions shared by both
# screeners. No state, no I/O.
# ============================================================

import numpy as np


def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except Exception:
        return default


def pct_ret(series, periods):
    try:
        s = series.dropna()
        if len(s) < periods + 1:
            return None
        return (s.iloc[-1] / s.iloc[-(periods+1)] - 1) * 100
    except Exception:
        return None


def yoy_growth(series):
    try:
        vals = series.dropna()
        if len(vals) < 5:
            return None
        latest = vals.iloc[0]; yr_ago = vals.iloc[4]
        if yr_ago == 0 or np.isnan(yr_ago):
            return None
        return ((latest - yr_ago) / abs(yr_ago)) * 100
    except Exception:
        return None


def fmt(val, decimals=2, fallback="—"):
    """Format a numeric value; return fallback string if None/nan/999."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or val == 999)):
            return fallback
        return round(float(val), decimals)
    except Exception:
        return fallback
