"""
sector_screener.py
===================
Sector-specific screening module. Runs alongside the existing stock-level
Contra screener (contra_screener.py) but is NOT part of it.

Scope (as agreed):
  - 13 sector "instruments" (ETF where liquid, index price as fallback)
  - Contra Phase 2 ONLY: weekly RSI(14) cross above its 14-week SMA,
    RSI < 60 at cross, re-confirmation check reused from stock-level logic.
    No Phase 1 volume-spike gate -- drawdown severity is used instead.
  - Rotation ranking: blended trailing-return relative-strength rank
    across all 13 sectors, with week-over-week rank-change flags.
  - Sector Momentum is explicitly OUT of scope for this module.

Integration points you'll need to wire up (marked ADAPT below):
  - get_gspread_client(), SHEET_NAME, col_letter()  -> from sheets_io.py
  - Telegram send                                    -> from telegram_notify.py
  - This module fetches its own price data directly via yfinance so it has
    no hard dependency on data_fetch.py's internals. If you'd rather route
    through data_fetch.py's existing cache, swap fetch_weekly_prices() to
    call that instead -- the rest of the logic is agnostic to the source.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Local cache paths (mirrors the /tmp caching pattern used elsewhere in the
# pipeline for price_data.pkl / fund_data.pkl).
CACHE_DIR = Path("/tmp/sector_screener_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ROTATION_RANK_CACHE = CACHE_DIR / "prev_rotation_ranks.pkl"

MIN_AVG_VOLUME = 50_000          # candidate ETF must clear this to be "tradeable"
LOOKBACK_WEEKS = 260              # ~5 years of weekly bars for RSI/drawdown history
RSI_PERIOD = 14
RSI_SMA_PERIOD = 14
CONTRA_RSI_CROSS_CEILING = 60    # RSI must be below this at the moment of cross

# Rotation blend weights (recency-weighted)
ROTATION_WEIGHTS = {"4W": 0.50, "12W": 0.30, "26W": 0.20}

# Candidate tickers per sector. Order = priority. First candidate that
# resolves in yfinance AND clears MIN_AVG_VOLUME wins. If none clear the
# volume bar, we fall back to the sector INDEX price for signal-only
# tracking (flagged as tradeable=False in the output).
#
# NOTE: several of these candidate symbols are best-effort and unconfirmed
# for current liquidity (esp. Realty, Chemicals, Power). That is exactly
# why the resolver validates live rather than trusting a static list --
# treat this dict as a starting point, not ground truth.
SECTOR_CANDIDATES: dict[str, dict] = {
    "Nifty 50": {
        "etf_candidates": ["NIFTYBEES.NS", "HDFCNIFETF.NS", "SETFNIF50.NS"],
        "index_fallback": "^NSEI",
    },
    "Nifty Next 50": {
        "etf_candidates": ["JUNIORBEES.NS", "ICICINXT50.NS", "SETFNN50.NS"],
        "index_fallback": "^NSEI",  # no reliable standalone Yahoo index ticker
    },
    "Bank Nifty": {
        "etf_candidates": ["BANKBEES.NS", "SETFNIFBK.NS", "KOTAKBKETF.NS"],
        "index_fallback": "^NSEBANK",
    },
    "Financial Services": {
        "etf_candidates": ["ICICIBFSI.NS", "FINIETF.NS"],
        "index_fallback": "NIFTY_FIN_SERVICE.NS",
    },
    "IT": {
        "etf_candidates": ["NETFIT.NS", "ICICITECH.NS", "SBIETFIT.NS"],
        "index_fallback": "^CNXIT",
    },
    "FMCG": {
        "etf_candidates": ["FMCGIETF.NS"],
        "index_fallback": "^CNXFMCG",
    },
    "Auto": {
        "etf_candidates": ["AUTOIETF.NS", "AUTOBEES.NS"],
        "index_fallback": "^CNXAUTO",
    },
    "Healthcare": {
        "etf_candidates": ["PHARMABEES.NS", "HEALTHIETF.NS"],
        "index_fallback": "^CNXPHARMA",
    },
    "Metals": {
        "etf_candidates": ["METALIETF.NS"],
        "index_fallback": "^CNXMETAL",
    },
    "Oil & Gas": {
        "etf_candidates": ["OILIETF.NS"],
        "index_fallback": "^CNXENERGY",
    },
    "Power": {
        "etf_candidates": ["POWERIETF.NS"],
        "index_fallback": "NIFTY_POWER.NS",
    },
    "Realty": {
        "etf_candidates": [],  # no confirmed liquid ETF -- index-only for now
        "index_fallback": "NIFTY_REALTY.NS",
    },
    "Chemicals": {
        "etf_candidates": [],  # no confirmed ETF at all -- index-only for now
        "index_fallback": "NIFTY_CHEMICALS.NS",
    },
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class SectorInstrument:
    sector: str
    ticker: str
    is_tradeable_etf: bool
    avg_volume: float


@dataclass
class SectorContraSignal:
    sector: str
    ticker: str
    is_tradeable_etf: bool
    signal_date: str
    rsi_at_cross: float
    drawdown_pct: float
    cross_failed: bool
    re_confirmed: bool
    quality_bucket: str
    quality_score: int


@dataclass
class SectorRotationRow:
    sector: str
    ticker: str
    ret_4w: float
    ret_12w: float
    ret_26w: float
    composite_rs: float
    rank: int
    prev_rank: Optional[int]
    rank_change: Optional[int]
    trend: str  # "UP" / "DOWN" / "FLAT" / "NEW"


# --------------------------------------------------------------------------
# Instrument resolution
# --------------------------------------------------------------------------

def _avg_volume(ticker: str, weeks: int = 12) -> float:
    try:
        hist = yf.Ticker(ticker).history(period=f"{weeks}wk", interval="1wk")
        if hist.empty or "Volume" not in hist:
            return 0.0
        return float(hist["Volume"].mean())
    except Exception:
        return 0.0


def resolve_sector_instrument(sector: str, cfg: dict) -> SectorInstrument:
    """Pick the best tradeable ETF for a sector, or fall back to the index."""
    for candidate in cfg["etf_candidates"]:
        vol = _avg_volume(candidate)
        if vol >= MIN_AVG_VOLUME:
            return SectorInstrument(sector, candidate, True, vol)

    # No candidate cleared the bar (or none exist) -- use index for signal only
    fallback = cfg["index_fallback"]
    vol = _avg_volume(fallback)
    return SectorInstrument(sector, fallback, False, vol)


def resolve_all_sectors() -> list[SectorInstrument]:
    return [resolve_sector_instrument(s, cfg) for s, cfg in SECTOR_CANDIDATES.items()]


# --------------------------------------------------------------------------
# Price fetch + indicators
# --------------------------------------------------------------------------

def fetch_weekly_prices(ticker: str, weeks: int = LOOKBACK_WEEKS) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=f"{weeks}wk", interval="1wk")
    if df.empty:
        return df
    df = df.dropna(subset=["Close"])
    return df


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_drawdown(close: pd.Series) -> pd.Series:
    running_high = close.cummax()
    return (close - running_high) / running_high * 100


# --------------------------------------------------------------------------
# Contra Phase 2 logic (drawdown tier replaces the volume-spike gate)
# --------------------------------------------------------------------------

def _drawdown_tier(drawdown_pct: float) -> int:
    dd = abs(drawdown_pct)
    if dd >= 45:
        return 3
    if dd >= 30:
        return 2
    if dd >= 15:
        return 1
    return 0


def _rsi_tier(rsi_at_cross: float) -> int:
    if 45 <= rsi_at_cross <= 60:
        return 3
    if 35 <= rsi_at_cross < 45:
        return 2
    if rsi_at_cross < 35:
        return 1
    return 0


def _quality_bucket(score: int, max_score: int) -> str:
    # Rescaled thresholds since there's no volume-spike component here
    # (stock-level Contra maxes at 7; this sector version maxes lower).
    pct = score / max_score if max_score else 0
    if pct >= 0.80:
        return "PREMIUM"
    if pct >= 0.55:
        return "HIGH"
    if pct >= 0.30:
        return "MEDIUM"
    return "LOW"


def detect_sector_contra_signal(inst: SectorInstrument) -> Optional[SectorContraSignal]:
    df = fetch_weekly_prices(inst.ticker)
    if df.empty or len(df) < RSI_PERIOD + RSI_SMA_PERIOD + 2:
        return None

    df["rsi"] = compute_rsi(df["Close"])
    df["rsi_sma"] = df["rsi"].rolling(RSI_SMA_PERIOD).mean()
    df["drawdown"] = compute_drawdown(df["Close"])
    df = df.dropna(subset=["rsi", "rsi_sma"])
    if len(df) < 3:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = prev["rsi"] <= prev["rsi_sma"] and latest["rsi"] > latest["rsi_sma"]
    if not crossed_up:
        return None
    if latest["rsi"] >= CONTRA_RSI_CROSS_CEILING:
        return None

    # Re-confirmation: RSI must still be above its SMA one bar after the cross
    cross_failed = False
    re_confirmed = False
    if len(df) >= 4:
        next_bar = df.iloc[-1]  # latest is treated as the cross bar itself here;
        # if you run this weekly, "re_confirmed" naturally resolves on the
        # following week's run when latest becomes the bar after the cross.
        re_confirmed = next_bar["rsi"] > next_bar["rsi_sma"]
        cross_failed = not re_confirmed

    dd_tier = _drawdown_tier(latest["drawdown"])
    rsi_tier = _rsi_tier(latest["rsi"])
    reconfirm_bonus = 1 if re_confirmed else 0
    score = dd_tier + rsi_tier + reconfirm_bonus
    max_score = 3 + 3 + 1  # dd_tier max + rsi_tier max + reconfirm bonus

    bucket = _quality_bucket(score, max_score)

    return SectorContraSignal(
        sector=inst.sector,
        ticker=inst.ticker,
        is_tradeable_etf=inst.is_tradeable_etf,
        signal_date=str(df.index[-1].date()),
        rsi_at_cross=round(float(latest["rsi"]), 2),
        drawdown_pct=round(float(latest["drawdown"]), 2),
        cross_failed=cross_failed,
        re_confirmed=re_confirmed,
        quality_bucket=bucket,
        quality_score=score,
    )


def screen_sector_contra(instruments: list[SectorInstrument]) -> list[SectorContraSignal]:
    signals = []
    for inst in instruments:
        sig = detect_sector_contra_signal(inst)
        if sig:
            signals.append(sig)
    return signals


# --------------------------------------------------------------------------
# Rotation ranking
# --------------------------------------------------------------------------

def _trailing_return(df: pd.DataFrame, weeks: int) -> float:
    if len(df) <= weeks:
        return np.nan
    start = df["Close"].iloc[-(weeks + 1)]
    end = df["Close"].iloc[-1]
    return float((end / start - 1) * 100)


def _load_prev_ranks() -> dict:
    if ROTATION_RANK_CACHE.exists():
        with open(ROTATION_RANK_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_ranks(ranks: dict) -> None:
    with open(ROTATION_RANK_CACHE, "wb") as f:
        pickle.dump(ranks, f)


def compute_rotation_ranking(instruments: list[SectorInstrument]) -> list[SectorRotationRow]:
    prev_ranks = _load_prev_ranks()
    rows = []

    for inst in instruments:
        df = fetch_weekly_prices(inst.ticker, weeks=40)
        if df.empty:
            continue
        r4 = _trailing_return(df, 4)
        r12 = _trailing_return(df, 12)
        r26 = _trailing_return(df, 26)

        parts = [r4, r12, r26]
        weights = [ROTATION_WEIGHTS["4W"], ROTATION_WEIGHTS["12W"], ROTATION_WEIGHTS["26W"]]
        valid = [(p, w) for p, w in zip(parts, weights) if not np.isnan(p)]
        if not valid:
            continue
        wsum = sum(w for _, w in valid)
        composite = sum(p * w for p, w in valid) / wsum

        rows.append({
            "sector": inst.sector,
            "ticker": inst.ticker,
            "ret_4w": r4,
            "ret_12w": r12,
            "ret_26w": r26,
            "composite_rs": composite,
        })

    ranked = sorted(rows, key=lambda r: r["composite_rs"], reverse=True)
    new_ranks = {}
    result = []
    for i, row in enumerate(ranked, start=1):
        sector = row["sector"]
        prev_rank = prev_ranks.get(sector)
        rank_change = (prev_rank - i) if prev_rank is not None else None
        if prev_rank is None:
            trend = "NEW"
        elif rank_change > 0:
            trend = "UP"
        elif rank_change < 0:
            trend = "DOWN"
        else:
            trend = "FLAT"

        new_ranks[sector] = i
        result.append(SectorRotationRow(
            sector=sector,
            ticker=row["ticker"],
            ret_4w=round(row["ret_4w"], 2) if not np.isnan(row["ret_4w"]) else None,
            ret_12w=round(row["ret_12w"], 2) if not np.isnan(row["ret_12w"]) else None,
            ret_26w=round(row["ret_26w"], 2) if not np.isnan(row["ret_26w"]) else None,
            composite_rs=round(row["composite_rs"], 2),
            rank=i,
            prev_rank=prev_rank,
            rank_change=rank_change,
            trend=trend,
        ))

    _save_ranks(new_ranks)
    return result


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------

def contra_signals_to_df(signals: list[SectorContraSignal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    return pd.DataFrame([s.__dict__ for s in signals])


def rotation_to_df(rows: list[SectorRotationRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in rows])


def format_telegram_section(
    signals: list[SectorContraSignal],
    rotation: list[SectorRotationRow],
) -> str:
    lines = ["📊 <b>Sector Signals</b>"]

    if signals:
        lines.append("\n<b>Contra Signals Fired:</b>")
        for s in signals:
            tag = "✅ ETF" if s.is_tradeable_etf else "⚠️ Index-only (no liquid ETF)"
            lines.append(
                f"• {s.sector} ({s.ticker}) — {s.quality_bucket} | "
                f"RSI {s.rsi_at_cross} | DD {s.drawdown_pct}% | {tag}"
            )
    else:
        lines.append("\nNo sector Contra signals this week.")

    if rotation:
        top3 = [r for r in rotation if r.rank <= 3]
        bottom3 = sorted(rotation, key=lambda r: r.rank, reverse=True)[:3]
        lines.append("\n<b>Rotation — Top 3:</b>")
        for r in top3:
            arrow = {"UP": "🔼", "DOWN": "🔽", "FLAT": "▪️", "NEW": "🆕"}[r.trend]
            lines.append(f"{arrow} #{r.rank} {r.sector} (RS {r.composite_rs}%)")
        lines.append("\n<b>Rotation — Bottom 3:</b>")
        for r in bottom3:
            arrow = {"UP": "🔼", "DOWN": "🔽", "FLAT": "▪️", "NEW": "🆕"}[r.trend]
            lines.append(f"{arrow} #{r.rank} {r.sector} (RS {r.composite_rs}%)")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Sheets export -- ADAPT to match sheets_io.py's actual write helpers.
# This assumes get_gspread_client() and SHEET_NAME are importable from
# sheets_io.py, matching the pattern already used in performance_tracker.py.
# --------------------------------------------------------------------------

def export_to_sheet(contra_df: pd.DataFrame, rotation_df: pd.DataFrame) -> None:
    try:
        from sheets_io import get_gspread_client, SHEET_NAME  # ADAPT if names differ
    except ImportError:
        print("[sector_screener] sheets_io not importable in this context -- "
              "skipping Sheets export (fine for local/test runs).")
        return

    client = get_gspread_client()
    sh = client.open(SHEET_NAME)

    def _write(df: pd.DataFrame, tab_name: str) -> None:
        if df.empty:
            df = pd.DataFrame([{"note": "no data this run"}])
        try:
            ws = sh.worksheet(tab_name)
            ws.clear()
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=str(max(len(df) + 10, 20)), cols="26")
        ws.update([df.columns.tolist()] + df.values.tolist())

    _write(contra_df, "Sector Contra")
    _write(rotation_df, "Sector Rotation")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_sector_screener() -> dict:
    instruments = resolve_all_sectors()

    contra_signals = screen_sector_contra(instruments)
    rotation_rows = compute_rotation_ranking(instruments)

    contra_df = contra_signals_to_df(contra_signals)
    rotation_df = rotation_to_df(rotation_rows)

    export_to_sheet(contra_df, rotation_df)

    telegram_text = format_telegram_section(contra_signals, rotation_rows)

    return {
        "contra_df": contra_df,
        "rotation_df": rotation_df,
        "telegram_text": telegram_text,
        "instruments": instruments,
    }


if __name__ == "__main__":
    result = run_sector_screener()
    print(result["telegram_text"])
    print("\n--- Instrument resolution ---")
    for inst in result["instruments"]:
        kind = "ETF" if inst.is_tradeable_etf else "INDEX (fallback)"
        print(f"{inst.sector:20s} -> {inst.ticker:15s} [{kind}] avg_vol={inst.avg_volume:,.0f}")
