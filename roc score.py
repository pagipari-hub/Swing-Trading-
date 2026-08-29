# ============================================================
# roc_score.py — ROC (Rate of Change) scoring module.
#
# Purpose: forward-testing overlay for options trade ideas.
# Computes ROC indicators + a 0-100 conviction score per ticker,
# using the SAME weekly price_data (5y history from yfinance)
# that contra_screener.py / momentum_screener.py already load —
# no new data source, no dependency on how many weeks have been
# logged to the export sheet.
#
# IMPORTANT — additive, not gating:
# These functions only ADD new columns (roc_5, roc_score, etc.)
# to the existing row dicts. They do NOT feed into pillar_a/
# pillar_b/total_score. This keeps the already-validated Contra/
# Momentum scoring untouched while the ROC score is forward-
# tested independently. Once you're happy with hit-rate data,
# wiring roc_score into the real total_score is a one-line change
# (see bottom of this file).
# ============================================================

import numpy as np
import pandas as pd


def compute_roc(close: pd.Series, periods: int):
    """ROC over N weekly periods, using full historical close series."""
    try:
        s = close.dropna()
        if len(s) <= periods:
            return None
        past = s.iloc[-(periods + 1)]
        current = s.iloc[-1]
        if past == 0 or pd.isna(past) or pd.isna(current):
            return None
        return round((current / past - 1) * 100, 2)
    except Exception:
        return None


def compute_roc_zscore(close: pd.Series, periods: int, lookback_weeks: int = 52):
    """
    Z-score of the current ROC(periods) vs its own trailing distribution.
    Used for Contra-side exhaustion detection — "how unusual is this
    week's ROC compared to this stock's own last-year ROC behaviour."
    """
    try:
        s = close.dropna()
        if len(s) <= periods + lookback_weeks:
            return None
        roc_hist = (s.pct_change(periods) * 100).dropna().iloc[-lookback_weeks:]
        if roc_hist.empty or roc_hist.std() == 0:
            return None
        return round((roc_hist.iloc[-1] - roc_hist.mean()) / roc_hist.std(), 2)
    except Exception:
        return None


def compute_momentum_roc(close: pd.Series, vol_ratio, rsi):
    """
    Momentum-side ROC score (0-100) — bullish continuation signal.
    Mirrors momentum_screener.py's pillar style: small additive
    components, each independently capped.

    Returns dict with roc_5, roc_20, roc_score, roc_signal.
    Values are None/"—"-safe — caller applies helpers.fmt() same
    as every other metric.
    """
    roc5  = compute_roc(close, 5)
    roc20 = compute_roc(close, 20)

    if roc5 is None or roc20 is None:
        return {"roc_5": roc5, "roc_20": roc20,
                "roc_score": None, "roc_signal": "Insufficient History"}

    score = 0.0

    # 1. Acceleration spread — short-term ROC outpacing longer-term (0-35)
    score += float(np.clip((roc5 - roc20) * 2.5, 0, 35))

    # 2. Volume confirmation (0-25)
    if vol_ratio is not None:
        score += float(np.clip((vol_ratio - 1) * 15, 0, 25))

    # 3. RSI alignment — momentum zone, not blow-off (0-20)
    if rsi is not None:
        if 55 <= rsi <= 80:
            score += 20
        elif 50 <= rsi < 55:
            score += 10

    # 4. Trend consistency — both windows agree on direction (0-20)
    if roc5 > 0 and roc20 > 0:
        score += 20
    elif roc5 > 0:
        score += 8

    score = round(min(score, 100), 1)
    signal = ("Momentum-Accel" if score >= 65 else
              "Momentum-Watch" if score >= 40 else "None")

    return {"roc_5": roc5, "roc_20": roc20, "roc_score": score, "roc_signal": signal}


def compute_contra_roc(close: pd.Series, vol_ratio, drawdown):
    """
    Contra-side ROC score (0-100) — exhaustion / reversal signal.

    Returns dict with roc_3, roc_14, roc_zscore, roc_score, roc_signal.
    """
    roc3  = compute_roc(close, 3)
    roc14 = compute_roc(close, 14)
    roc20 = compute_roc(close, 20)
    z14   = compute_roc_zscore(close, 14)

    if roc14 is None or z14 is None:
        return {"roc_3": roc3, "roc_14": roc14, "roc_zscore": z14,
                "roc_score": None, "roc_signal": "Insufficient History"}

    score = 0.0

    # 1. ROC extremity — more negative z-score = more exhausted (0-35)
    if z14 < 0:
        score += float(np.clip(abs(z14) * 15, 0, 35))

    # 2. Curl confirmation — short-term turning up inside a longer downtrend (0-25)
    if roc3 is not None and roc20 is not None:
        if roc3 > 0 and roc20 < 0:
            score += 25
        elif roc3 > 0:
            score += 10

    # 3. Drawdown context bonus (0-20)
    if drawdown is not None and drawdown > 15:
        score += min(drawdown / 2, 20)

    # 4. Volume-on-bounce (0-20)
    if vol_ratio is not None:
        score += float(np.clip((vol_ratio - 1) * 12, 0, 20))

    score = round(min(score, 100), 1)
    signal = ("Contra-Exhaust" if score >= 65 else
              "Contra-Watch" if score >= 40 else "None")

    return {"roc_3": roc3, "roc_14": roc14, "roc_zscore": z14,
            "roc_score": score, "roc_signal": signal}


# ------------------------------------------------------------
# To later fold this into the real total_score (only after
# forward-test data supports it), in contra_screener.py:
#
#   total = pillar_a + pillar_b + (roc_score or 0) * 0.15   # e.g. 15% weight
#
# and equivalently in momentum_screener.py's `raw` line. Not
# done here on purpose — keep the validated scores untouched
# until ROC earns its place with real hit-rate evidence.
# ------------------------------------------------------------
