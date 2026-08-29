# ============================================================
# config.py — Shared constants, thresholds, paths.
# Nothing in here does I/O; safe to import from anywhere.
# ============================================================

import os
from datetime import datetime

# ── Week tag / run timestamp ─────────────────────────────────
_iso      = datetime.now().isocalendar()
WEEK_TAG  = f"{_iso.year}-W{_iso.week:02d}"   # e.g. 2026-W29
TIMESTAMP = datetime.now().strftime("%d %b %Y %H:%M")

# ── Telegram ──────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Google Sheets ─────────────────────────────────────────────
SHEET_NAME = "Nifty500 Screener"

# ── Cache paths ───────────────────────────────────────────────
CACHE_DIR           = "/tmp/nifty500_cache"
PRICE_FILE          = os.path.join(CACHE_DIR, "price_data.pkl")
FUND_FILE           = os.path.join(CACHE_DIR, "fund_data.pkl")
NIFTY500_LIST_FILE  = os.path.join(CACHE_DIR, "nifty500_list.pkl")
NIFTY500_SEED_FILE  = "nifty500_seed.csv"   # committed to repo — last-resort fallback
# NOTE: q_fin_data is intentionally NOT cached — always fresh each run

# price_data/fund_data are meant to refresh weekly (matches WEEK_TAG
# cadence). This is a SECOND, independent line of defense against
# stale caching — belt-and-suspenders alongside the GitHub Actions
# cache key itself being date-scoped. If a future workflow edit ever
# reintroduces a constant/broken cache key (as happened once already —
# `github.run_date` doesn't exist as a context property, so the key
# silently resolved to a fixed string and price/fund data was frozen
# for weeks with no error), this catches it here in code rather than
# depending on the YAML being correct.
CACHE_MAX_AGE_DAYS = 8   # small buffer over the 7-day refresh cycle

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Index tickers ─────────────────────────────────────────────
CONTRA_NIFTY_INDEX = "^CRSLDX"   # used by Contra's own 4-tier regime (Section 3c)
NIFTY500_INDEX      = "^CRSLDX"  # used by Momentum's Risk-On/Risk-Off regime (Section 5)

# ── Contra: fundamental gate thresholds ─────────────────────
CONTRA_GATE_DE_MAX       = 150   # debt/equity must be <= this
CONTRA_GATE_SALES_MIN    = -15   # revenue YoY must be > this
CONTRA_GATE_PROMOTER_MIN = 20    # promoter holding must be >= this

# ── Momentum: scoring / filter thresholds ────────────────────
WEEKS_VALID   = 3
RSI_UPPER     = 75
DIST_HIGH_MAX = 20
LIQ_HARD      = 10
LIQ_THIN      = 20
VOL_WINDOW    = 13
RAW_MAX       = 113    # Pillar A(50) + B(35) + C(15) + D(13) = 113
WATCH_MIN_A   = 20

FLOOR_SALES_GR_MIN = -15
FLOOR_PAT_GR_MIN   = -25
FLOOR_DE_RATIO_MAX = 150
FLOOR_PROMOTER_MIN = 20

CYCLICAL_NSE = {
    "Metals & Mining", "Oil Gas & Consumable Fuels",
    "Construction", "Construction Materials",
    "Automobile and Auto Components", "Capital Goods",
    "Power", "Chemicals"
}

# ── Sheet export — metric row order ──────────────────────────
CONTRA_METRICS = [
    "price", "total_score", "phase", "RSI", "drawdown", "de_ratio",
    "cross_week", "vol_ratio", "promoter", "sales_gr", "pat_gr",
    "pct_from_low", "low_confirmed", "pillar_a", "pillar_b",
    "quality_score", "quality_bucket", "archetype_a",
    "cross_failed", "re_confirmed",
    # ROC forward-test overlay (roc_score.py) — additive, not part of total_score
    "roc_3", "roc_14", "roc_zscore", "roc_score", "roc_signal",
]

MOMENTUM_METRICS = [
    "price", "total_score", "label", "RSI", "rs_periods", "earn_gr",
    "pct_from_high", "vol_ratio", "new_52w_high", "new_5y_high",
    "pillar_a", "pillar_b", "pillar_c", "pillar_d",
    "cross_failed", "re_confirmed",
    # ROC forward-test overlay (roc_score.py) — additive, not part of total_score
    "roc_5", "roc_20", "roc_score", "roc_signal",
]
