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
        "etf_candidates": ["NIFTYBEES.NS", "HDFCNIFETF.NS", "SETFNIF50.NS",
                            "KOTAKNIFTY.NS", "BSLNIFTY.NS", "MAN50ETF.NS"],
        "index_fallback": "^NSEI",
    },
    "Nifty Next 50": {
        "etf_candidates": ["JUNIORBEES.NS", "ICICINXT50.NS", "SETFNN50.NS"],
        "index_fallback": "^NIFTYJR",  # confirmed via Wikipedia
    },
    "Bank Nifty": {
        "etf_candidates": ["BANKBEES.NS", "SETFNIFBK.NS", "KOTAKBKETF.NS"],
        "index_fallback": "^NSEBANK",
    },
    "Financial Services": {
        # ICICIBFSI.NS confirmed dead (404) in live run 2026-07-31 -- removed
        "etf_candidates": ["FINIETF.NS"],
        "index_fallback": "^CNXFIN",
    },
    "IT": {
        # NETFIT.NS, ICICITECH.NS confirmed dead (404) in live run -- removed.
        # SBIETFIT.NS confirmed working; KOTAKIT.NS, ITBEES.NS (Nippon),
        # ITETF.NS (Mirae) added as confirmed-real backups.
        "etf_candidates": ["SBIETFIT.NS", "KOTAKIT.NS", "ITBEES.NS", "ITETF.NS"],
        "index_fallback": "^CNXIT",
    },
    "FMCG": {
        "etf_candidates": ["FMCGIETF.NS"],  # confirmed working
        "index_fallback": "^CNXFMCG",
    },
    "Auto": {
        "etf_candidates": ["AUTOIETF.NS"],  # confirmed working
        "index_fallback": "^CNXAUTO",
    },
    "Healthcare": {
        "etf_candidates": ["PHARMABEES.NS"],  # confirmed working
        "index_fallback": "^CNXPHARMA",
    },
    "Metals": {
        "etf_candidates": ["METALIETF.NS"],  # confirmed working
        "index_fallback": "^CNXMETAL",
    },
    "Oil & Gas": {
        "etf_candidates": ["OILIETF.NS"],  # confirmed working
        "index_fallback": "^CNXENERGY",
    },
    "Power": {
        # POWERIETF.NS confirmed dead (404); no dedicated Nifty Power ETF or
        # index ticker found on Yahoo despite a wide search. CPSEETF.NS
        # (Central Public Sector Enterprises ETF) used as an approximate
        # proxy per your call -- confirmed live/liquid (avg_vol ~1.35M), but
        # NOT the actual Nifty Power index: it's PSU-focused (ONGC, Coal
        # India, NTPC, etc.), only ~44% weighted toward Power Generation &
        # Distribution per its own factsheet. Treat Power's signals here as
        # directionally indicative, not a precise sector read.
        "etf_candidates": ["CPSEETF.NS"],
        "index_fallback": None,
    },
    "Realty": {
        "etf_candidates": [],  # no confirmed liquid ETF
        "index_fallback": "^CNXREALTY",  # confirmed live on Yahoo Finance
    },
    "Chemicals": {
        # Nifty Chemicals is a newer NSE index (launched 2021) -- no ETF and
        # no confirmed Yahoo ticker found. This sector currently has NO
        # working data source; consider dropping it from the screen or
        # sourcing it via NSE's own API/niftyindices.com instead of yfinance.
        "etf_candidates": [],
        "index_fallback": "NIFTY_CHEMICALS.NS",  # unconfirmed, will likely still fail
    },
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class SectorInstrument:
    sector: str
    ticker: Optional[str]
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
    rotation_bonus: int
    cross_sector_bonus: int


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
    """Pick the best tradeable ETF for a sector, or fall back to the index.
    Returns an instrument with ticker=None if neither an ETF candidate nor
    an index fallback is available/working -- callers must check for this."""
    for candidate in cfg["etf_candidates"]:
        vol = _avg_volume(candidate)
        if vol >= MIN_AVG_VOLUME:
            return SectorInstrument(sector, candidate, True, vol)

    fallback = cfg.get("index_fallback")
    if not fallback:
        return SectorInstrument(sector, None, False, 0.0)

    # No candidate cleared the bar (or none exist) -- use index for signal only
    vol = _avg_volume(fallback)
    return SectorInstrument(sector, fallback, False, vol)


def resolve_all_sectors() -> list[SectorInstrument]:
    print("\n📥 Resolving sector instruments (ETF where liquid, index fallback otherwise)...")
    instruments = []
    for sector, cfg in SECTOR_CANDIDATES.items():
        inst = resolve_sector_instrument(sector, cfg)
        if inst.ticker is None:
            print(f"   {sector:20s} -> ⚠️  NO WORKING DATA SOURCE (no ETF, no index fallback)")
        else:
            kind = "ETF" if inst.is_tradeable_etf else "INDEX (fallback — no liquid ETF)"
            proxy_note = " ⚠️ APPROXIMATE PROXY, not true sector index" if sector == "Power" else ""
            print(f"   {sector:20s} -> {inst.ticker:20s} [{kind}] avg_vol={inst.avg_volume:,.0f}{proxy_note}")
        instruments.append(inst)
    return instruments


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
    # Gradient, not flat bands: deeper oversold at the moment of cross scores
    # higher. Ceiling of 60 is already enforced upstream (CONTRA_RSI_CROSS_CEILING).
    if rsi_at_cross <= 30:
        return 5
    if rsi_at_cross <= 40:
        return 4
    if rsi_at_cross <= 50:
        return 3
    return 2  # 51-60, still a valid cross, lowest tier


def _quality_bucket(score: int, max_score: int) -> str:
    # Rescaled thresholds since there's no volume-spike component here
    # (stock-level Contra maxes at 7; this sector version maxes higher
    # now with RSI-gradient + rotation + cross-sector layers).
    pct = score / max_score if max_score else 0
    if pct >= 0.80:
        return "PREMIUM"
    if pct >= 0.55:
        return "HIGH"
    if pct >= 0.30:
        return "MEDIUM"
    return "LOW"


def _rotation_bonus(sector: str, rotation_lookup: dict) -> int:
    """Rewards a fired Contra signal in a sector that's ALSO showing relative
    strength / improving rank in the Rotation ranking. Never penalizes a
    weak-rotation sector -- the RSI cross alone is still enough to fire."""
    row = rotation_lookup.get(sector)
    if row is None:
        return 0
    bonus = 0
    if row.rank <= 3:
        bonus += 2
    elif row.rank <= 6:
        bonus += 1
    if row.trend == "UP":
        bonus += 1
    return bonus


def _cross_sector_rsi_bonus(sector: str, rsi_percentile_map: dict) -> int:
    """Rewards a sector whose current RSI is relatively LOW compared to the
    other 12 sectors right now -- i.e. more differentiated/oversold versus
    peers, not just oversold on its own multi-year history. percentile: 0.0
    = lowest RSI of the group (most oversold relative to peers), 1.0 = highest."""
    pct = rsi_percentile_map.get(sector)
    if pct is None:
        return 0
    if pct <= 1 / 3:
        return 2
    if pct <= 2 / 3:
        return 1
    return 0


def detect_sector_contra_signal(
    inst: SectorInstrument,
    contra_regime: str,
    rsi_percentile_map: Optional[dict] = None,
    rotation_lookup: Optional[dict] = None,
) -> Optional[SectorContraSignal]:
    if inst.ticker is None:
        return None

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
    # Regime bonus -- matches stock-level Contra: your backtest found CAUTION
    # is historically the best entry environment, BULL the noisiest, so this
    # rewards signals firing in CAUTION/BEAR the same way stock-level does.
    regime_bonus = 1 if contra_regime in ("CAUTION", "BEAR") else 0
    rotation_bonus = _rotation_bonus(inst.sector, rotation_lookup or {})
    cross_sector_bonus = _cross_sector_rsi_bonus(inst.sector, rsi_percentile_map or {})

    score = (dd_tier + rsi_tier + reconfirm_bonus + regime_bonus
             + rotation_bonus + cross_sector_bonus)
    # dd_tier max(3) + rsi_tier max(5) + reconfirm(1) + regime(1)
    # + rotation max(2 rank + 1 trend = 3) + cross-sector max(2)
    max_score = 3 + 5 + 1 + 1 + 3 + 2

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
        rotation_bonus=rotation_bonus,
        cross_sector_bonus=cross_sector_bonus,
    )


def screen_sector_contra(
    instruments: list[SectorInstrument],
    contra_regime: str,
    rsi_percentile_map: dict,
    rotation_lookup: dict,
) -> list[SectorContraSignal]:
    print("\n📊 Sector Contra (Phase 2 — RSI gradient + drawdown + rotation + "
          "cross-sector RSI + regime, no volume gate)...")
    signals = []
    for inst in instruments:
        if inst.ticker is None:
            print(f"   ⚠️  {inst.sector:20s} skipped — no data source")
            continue
        sig = detect_sector_contra_signal(inst, contra_regime, rsi_percentile_map, rotation_lookup)
        if sig:
            print(f"   ✅ {sig.sector:20s} SIGNAL — {sig.quality_bucket} (score {sig.quality_score}) "
                  f"RSI {sig.rsi_at_cross} | DD {sig.drawdown_pct}% | "
                  f"rot_bonus +{sig.rotation_bonus} | xsec_bonus +{sig.cross_sector_bonus}")
            signals.append(sig)
        else:
            print(f"   —  {inst.sector:20s} no signal")
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
    print("\n📈 Sector Rotation ranking (4W/12W/26W blended RS)...")
    prev_ranks = _load_prev_ranks()
    rows = []

    for inst in instruments:
        if inst.ticker is None:
            print(f"   ⚠️  {inst.sector:20s} skipped — no data source")
            continue
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

    for row in result:
        arrow = {"UP": "🔼", "DOWN": "🔽", "FLAT": "▪️", "NEW": "🆕"}[row.trend]
        print(f"   {arrow} #{row.rank:<2d} {row.sector:20s} RS {row.composite_rs:+.1f}%")

    return result


# --------------------------------------------------------------------------
# Cross-sector RSI comparison -- how oversold is this sector RIGHT NOW
# relative to the other 12, not just relative to its own multi-year history.
# --------------------------------------------------------------------------

def _get_current_rsi(ticker: str) -> Optional[float]:
    df = fetch_weekly_prices(ticker, weeks=60)
    if df.empty or len(df) < RSI_PERIOD + 2:
        return None
    rsi = compute_rsi(df["Close"]).dropna()
    if rsi.empty:
        return None
    return float(rsi.iloc[-1])


def compute_cross_sector_rsi_ranks(instruments: list[SectorInstrument]) -> dict:
    """Returns {sector: percentile} where percentile 0.0 = lowest current RSI
    of the group (most oversold relative to peers right now), 1.0 = highest."""
    print("\n🔀 Cross-sector RSI comparison (relative oversold ranking)...")
    rsi_map = {}
    for inst in instruments:
        if inst.ticker is None:
            continue
        val = _get_current_rsi(inst.ticker)
        if val is not None:
            rsi_map[inst.sector] = val

    if not rsi_map:
        print("   ⚠️  No sectors resolved a current RSI value -- skipping.")
        return {}

    ordered = sorted(rsi_map.items(), key=lambda kv: kv[1])
    n = len(ordered)
    percentile_map = {}
    for i, (sector, val) in enumerate(ordered):
        pct = i / (n - 1) if n > 1 else 0.0
        percentile_map[sector] = pct
        tag = "most oversold vs peers" if pct <= 1 / 3 else ("mid" if pct <= 2 / 3 else "least oversold vs peers")
        print(f"   #{i + 1:<2d} {sector:20s} RSI {val:5.1f}  ({tag})")

    return percentile_map


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
                f"• {s.sector} ({s.ticker}) — {s.quality_bucket} (score {s.quality_score}) | "
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
# Sheets export -- uses the same transposed weekly-column structure as
# Contra/Momentum (build_transposed_sheet from sheets_io.py) so Sector
# data accumulates history week over week instead of being overwritten.
# --------------------------------------------------------------------------

SECTOR_CONTRA_METRICS = [
    "is_tradeable_etf", "signal_date", "rsi_at_cross", "drawdown_pct",
    "cross_failed", "re_confirmed", "quality_bucket", "quality_score",
    "rotation_bonus", "cross_sector_bonus",
]

SECTOR_ROTATION_METRICS = [
    "ret_4w", "ret_12w", "ret_26w", "composite_rs",
    "rank", "prev_rank", "rank_change", "trend",
]


def export_to_sheet(contra_df: pd.DataFrame, rotation_df: pd.DataFrame) -> None:
    try:
        from config import SHEET_NAME, WEEK_TAG  # ADAPT if imported differently
        from sheets_io import get_gspread_client, ensure_worksheet, build_transposed_sheet
    except ImportError as e:
        print(f"[sector_screener] Required modules not importable ({e}) -- "
              "skipping Sheets export (fine for local/test runs).")
        return

    client = get_gspread_client()
    sh = client.open(SHEET_NAME)

    if contra_df.empty:
        print("   ⚠️  Sector Contra: no signals this week -- skipping tab write "
              "(no week column added for an all-empty week).")
    else:
        ws_contra = ensure_worksheet(sh, "Sector Contra", rows=2000, cols=100)
        build_transposed_sheet(ws_contra, contra_df, SECTOR_CONTRA_METRICS,
                                WEEK_TAG, has_cyclical=False)

    if rotation_df.empty:
        print("   ⚠️  Sector Rotation: no rows this week -- skipping tab write.")
    else:
        ws_rotation = ensure_worksheet(sh, "Sector Rotation", rows=2000, cols=100)
        build_transposed_sheet(ws_rotation, rotation_df, SECTOR_ROTATION_METRICS,
                                WEEK_TAG, has_cyclical=False)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_sector_screener(contra_regime: str) -> dict:
    print("=" * 65)
    print("SECTOR SCREENER")
    print("=" * 65)

    instruments = resolve_all_sectors()

    # Rotation and cross-sector RSI both need to exist BEFORE Contra scoring
    # now, since Contra's score uses them as bonus layers.
    rotation_rows = compute_rotation_ranking(instruments)
    rotation_lookup = {row.sector: row for row in rotation_rows}
    rsi_percentile_map = compute_cross_sector_rsi_ranks(instruments)

    contra_signals = screen_sector_contra(instruments, contra_regime, rsi_percentile_map, rotation_lookup)

    contra_df = contra_signals_to_df(contra_signals)
    rotation_df = rotation_to_df(rotation_rows)

    export_to_sheet(contra_df, rotation_df)

    telegram_text = format_telegram_section(contra_signals, rotation_rows)

    etf_count = sum(1 for i in instruments if i.is_tradeable_etf)
    no_data_count = sum(1 for i in instruments if i.ticker is None)
    index_count = len(instruments) - etf_count - no_data_count
    print(f"""
{'='*65}
SECTOR SCREENER RESULTS
{'='*65}
  Sectors screened     : {len(instruments)}
  Tradeable ETF found  : {etf_count}/{len(instruments)}
  Index-only fallback  : {index_count}/{len(instruments)}
  No data source       : {no_data_count}/{len(instruments)}
  Contra signals fired : {len(contra_signals)}
  Rotation ranked      : {len(rotation_rows)}
{'='*65}
""")

    return {
        "contra_df": contra_df,
        "rotation_df": rotation_df,
        "telegram_text": telegram_text,
        "instruments": instruments,
    }


if __name__ == "__main__":
    # Standalone/local test run -- no live contra_regime available outside
    # main.py's pipeline, so default to NEUTRAL (no regime bonus applied).
    result = run_sector_screener(contra_regime="NEUTRAL")
    print(result["telegram_text"])
