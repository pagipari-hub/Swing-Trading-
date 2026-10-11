# ============================================================
# contra_screener.py — Contra (contrarian mean-reversion) strategy.
#
# Two-phase system:
#   Phase 1 — institutional accumulation via volume spikes on
#             beaten-down stocks
#   Phase 2 — entry when weekly RSI(14) crosses above its 14-week
#             SMA, RSI < 60 at the cross, still in contra territory
#
# Regime is a standalone 4-tier system (BULL/NEUTRAL/CAUTION/BEAR)
# based on Nifty500 RSI — intentionally separate from Momentum's
# binary Risk-On/Risk-Off regime in momentum_screener.py.
# ============================================================

import numpy as np
import pandas as pd
import ta

from config import (
    CONTRA_NIFTY_INDEX, CONTRA_GATE_DE_MAX, CONTRA_GATE_SALES_MIN,
    CONTRA_GATE_PROMOTER_MIN,
    CONVICTION_RSI_RECOVERY_MIN, CONVICTION_DRAWDOWN_IMPROVEMENT_MIN,
    CONVICTION_BUY_THRESHOLD,
)
from helpers import safe_get, yoy_growth, fmt
from roc_score import compute_contra_roc


def compute_sector_drawdown_baseline(nifty500, price_data, sector_lookup):
    """
    Median 52-week drawdown per sector — used by Contra's sector-relative
    weakness scoring (rewards falling MORE than the sector median, i.e.
    idiosyncratic decline rather than systemic sector-wide drag).
    """
    print("\n📊 Building sector drawdown baseline (for Contra)...")
    sector_drawdowns = {}
    for t in nifty500:
        df = price_data.get(t)
        if df is None or len(df) < 60:
            continue
        try:
            close = df["Close"].squeeze()
            high  = df["High"].squeeze()
            price = close.iloc[-1]
            high_52w = high.iloc[-52:].max()
            dd = (high_52w - price) / high_52w * 100
            sec = sector_lookup.get(t, "Unknown")
            sector_drawdowns.setdefault(sec, []).append(dd)
        except Exception:
            pass

    sector_median_drawdown = {
        sec: float(np.median(vals)) for sec, vals in sector_drawdowns.items() if vals
    }
    print(f"  Sector drawdown baseline built for {len(sector_median_drawdown)} sectors")
    return sector_median_drawdown


def compute_contra_regime(nifty500, price_data):
    """
    4-tier regime (BULL/NEUTRAL/CAUTION/BEAR) from Nifty500 weekly RSI.
    Backtesting showed CAUTION historically produced the best Contra
    signals (~62% win rate) while BULL was noisiest (~49%) — so this
    regime feeds a quality-score bonus rather than gating entries.
    """
    print("\n📊 Building Contra regime (4-tier)...")
    contra_regime = "NEUTRAL"

    try:
        nf = price_data.get(CONTRA_NIFTY_INDEX)
        if nf is None:
            closes = [price_data[t]["Close"].squeeze()
                      for t in nifty500
                      if price_data.get(t) is not None and len(price_data[t]) > 60]
            contra_nifty_close = pd.concat(closes, axis=1).median(axis=1)
        else:
            contra_nifty_close = nf["Close"].squeeze()

        n_rsi = ta.momentum.RSIIndicator(contra_nifty_close, window=14).rsi().iloc[-1]

        if n_rsi >= 60:
            contra_regime = "BULL"
        elif n_rsi >= 45:
            contra_regime = "NEUTRAL"
        elif n_rsi >= 35:
            contra_regime = "CAUTION"
        else:
            contra_regime = "BEAR"

        print(f"  📍 Contra regime: {contra_regime} (Nifty500 RSI {n_rsi:.1f})")
    except Exception as e:
        contra_regime = "NEUTRAL"
        print(f"  ⚠️  Contra regime check failed ({e}) — defaulting to NEUTRAL")

    return contra_regime


def _num(v):
    """Row values arrive as rounded floats or the "—" placeholder; return float or None."""
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pass_fail(flag):
    """True/False/None -> "PASS"/"FAIL"/"N/A" (the strings the Sheets formulas test against)."""
    return "N/A" if flag is None else ("PASS" if flag else "FAIL")


def compute_conviction(row, prev_row=None):
    """
    Conviction layer on top of the Contra total_score (row["score"]).
    Does NOT change total_score. Encodes the W24->W35 backtest lesson:
    winners had positive sales/PAT, RSI recovery and shrinking drawdown;
    losers had weak fundamentals, high leverage and a deepening drawdown.

    row      : a contra_rows dict (values may be floats or the "—" placeholder)
    prev_row : {"RSI": float, "drawdown": float} for the previous week, or None

    Missing inputs (growth/D-E not reported, no previous week) are treated
    as UNKNOWN: they get a 0 adjustment and show "N/A" / "—" rather than
    being scored as a failure. Unknown fundamentals can therefore never
    produce a VALUE TRAP or a BUY — the stock stays WATCH.

    Returns dict: conviction_score, fund_pass, lev_pass, rsi_delta,
    dd_delta, action.
    """
    prev_row = prev_row or {}

    total_score = _num(row.get("score")) or 0
    sales_gr    = _num(row.get("sales_gr"))
    pat_gr      = _num(row.get("pat_gr"))
    de_ratio    = _num(row.get("de_ratio"))
    rsi         = _num(row.get("RSI"))
    dd          = _num(row.get("drawdown"))
    prev_rsi    = _num(prev_row.get("RSI"))
    prev_dd     = _num(prev_row.get("drawdown"))

    # Fundamentals: both growth figures must be positive
    if sales_gr is None or pat_gr is None:
        fund_pass = None
    else:
        fund_pass = (sales_gr > 0) and (pat_gr > 0)

    # Leverage: D/E is not meaningful for Financial Services (always passes)
    if row.get("sector") == "Financial Services":
        lev_pass = True
    elif de_ratio is None:
        lev_pass = None
    else:
        lev_pass = de_ratio < 100

    # Week-over-week deltas (drawdown delta positive = drawdown shrinking)
    rsi_delta = (rsi - prev_rsi) if (rsi is not None and prev_rsi is not None) else None
    dd_delta  = (prev_dd - dd)   if (dd  is not None and prev_dd  is not None) else None

    score = total_score

    if fund_pass is True:    score += 20
    elif fund_pass is False: score -= 20

    if rsi_delta is not None:
        score += 10 if rsi_delta > CONVICTION_RSI_RECOVERY_MIN else -10

    if dd_delta is not None:
        score += 10 if dd_delta > CONVICTION_DRAWDOWN_IMPROVEMENT_MIN else -10

    score += {"PREMIUM": 15, "HIGH": 10, "MEDIUM": 5}.get(row.get("quality_bucket"), 0)

    if lev_pass is True:    score += 5
    elif lev_pass is False: score -= 5

    if row.get("low_confirmed") == "Yes":
        score += 5

    conviction_score = int(round(max(0, min(100, score))))

    if total_score >= 50 and conviction_score >= CONVICTION_BUY_THRESHOLD and fund_pass is True:
        action = "BUY"
    elif total_score >= 50 and fund_pass is False:
        action = "VALUE TRAP"
    else:
        action = "WATCH"

    return {
        "conviction_score": conviction_score,
        "fund_pass":        _pass_fail(fund_pass),
        "lev_pass":         _pass_fail(lev_pass),
        "rsi_delta":        fmt(rsi_delta, 1),
        "dd_delta":         fmt(dd_delta, 1),
        "action":           action,
    }


def run_contra_screener(nifty500, price_data, fund_data, q_fin_data,
                         sector_lookup, company_lookup, promoter_data,
                         sector_median_drawdown, contra_regime, week_tag):
    """Returns (contra_df, p2_df, p1_df)."""
    print("\n" + "=" * 65)
    print("CONTRA SCREENER")
    print("=" * 65)

    contra_rows = []
    prev_lookup = {}   # ticker -> previous week's RSI / drawdown / price (for conviction layer)
    
    for ticker in nifty500:
        df   = price_data.get(ticker)
        info = fund_data.get(ticker, {})
        if df is None or len(df) < 60:
            continue
    
        try:
            close   = df["Close"].squeeze()
            volume  = df["Volume"].squeeze()
            high_s  = df["High"].squeeze()
            low_s   = df["Low"].squeeze()
            price   = close.iloc[-1]
            company = company_lookup.get(ticker, ticker)
            sector  = sector_lookup.get(ticker, "Unknown")
    
            rsi_s  = ta.momentum.RSIIndicator(close, window=14).rsi()
            if rsi_s is None or rsi_s.dropna().shape[0] < 5:
                continue
            rsi_sma = rsi_s.rolling(14).mean()
            rsi     = rsi_s.iloc[-1]
            rsi_sm  = rsi_sma.iloc[-1]
    
            vol_avg        = volume.iloc[-14:-1].mean()
            vol_ratio_now  = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0
            peak_vol_ratio = 0
            for w in range(8):
                try:
                    v = volume.iloc[-(w+1)] / vol_avg if vol_avg > 0 else 0
                    peak_vol_ratio = max(peak_vol_ratio, v)
                except Exception:
                    pass
            vol_ratio = max(vol_ratio_now, peak_vol_ratio)
    
            high_52w = high_s.iloc[-52:].max()
            low_52w  = low_s.iloc[-52:].min()
            drawdown = (high_52w - price) / high_52w * 100
            pct_from_low = (price - low_52w) / low_52w * 100 if low_52w > 0 else 0
            low_confirmed = 25 <= pct_from_low <= 60

            # Previous week's readings, from the same weekly price series
            # (bars are weekly, so iloc[-2] is last week). Used only by
            # compute_conviction + wk_return after this loop.
            prev_high = high_s.iloc[-53:-1].max()
            prev_px   = close.iloc[-2]
            prev_lookup[ticker] = {
                "RSI":      rsi_s.iloc[-2],
                "price":    prev_px,
                "drawdown": (prev_high - prev_px) / prev_high * 100 if prev_high > 0 else None,
            }

            # ROC forward-test overlay — additive only, does NOT feed into
            # pillar_a/pillar_b/total below. See roc_score.py header.
            roc = compute_contra_roc(close, vol_ratio, drawdown)

            # promoter holding via Screener.in scrape (see promoter_scraper.py) —
            # yfinance's heldPercentInsiders does not populate for NSE tickers,
            # confirmed empirically across 6 weeks of live data. Screener.in
            # already reports this as a percentage (e.g. 71.8), not a 0-1
            # fraction, so no *100 conversion here.
            promoter = promoter_data.get(ticker)
            de_ratio = safe_get(info, "debtToEquity", 999)
    
            sales_gr = pat_gr = None
            qfin = q_fin_data.get(ticker)
            if qfin is not None and not qfin.empty:
                for k in ["Total Revenue", "Revenue"]:
                    if k in qfin.index:
                        sales_gr = yoy_growth(qfin.loc[k]); break
                for k in ["Net Income", "Net Income Common Stockholders"]:
                    if k in qfin.index:
                        pat_gr = yoy_growth(qfin.loc[k]); break
    
            net_income_positive = None
            if qfin is not None and not qfin.empty:
                for k in ["Net Income", "Net Income Common Stockholders"]:
                    if k in qfin.index:
                        try:
                            latest_ni = qfin.loc[k].dropna().iloc[0]
                            net_income_positive = latest_ni > 0
                        except Exception:
                            pass
                        break
    
            fails_gate = (
                (net_income_positive is False) or
                (de_ratio is not None and de_ratio != 999 and de_ratio > CONTRA_GATE_DE_MAX) or
                (sales_gr is not None and sales_gr <= CONTRA_GATE_SALES_MIN) or
                (promoter is not None and promoter < CONTRA_GATE_PROMOTER_MIN)
            )
    
            if fails_gate:
                contra_rows.append({
                    "week": week_tag, "ticker": ticker, "company": company, "sector": sector,
                    "price": fmt(price, 2), "score": 0, "phase": "Filtered (Weak Fundamentals)",
                    "RSI": fmt(rsi, 1), "drawdown": fmt(drawdown, 1),
                    "de_ratio": fmt(de_ratio, 2) if de_ratio != 999 else "—",
                    "cross_week": "—", "vol_ratio": fmt(vol_ratio, 2),
                    "promoter": fmt(promoter, 1) if promoter is not None else "—",
                    "sales_gr": fmt(sales_gr, 1) if sales_gr is not None else "—",
                    "pat_gr": fmt(pat_gr, 1) if pat_gr is not None else "—",
                    "pct_from_low": fmt(pct_from_low, 1), "low_confirmed": "—",
                    "pillar_a": 0, "pillar_b": 0, "quality_score": 0, "quality_bucket": "—",
                    "archetype_a": "No", "cross_failed": "—", "re_confirmed": "—",
                    "roc_3": "—", "roc_14": "—", "roc_zscore": "—",
                    "roc_score": "—", "roc_signal": "—",
                    "conviction_score": "—", "fund_pass": "—", "lev_pass": "—",
                    "rsi_delta": "—", "dd_delta": "—", "action": "—", "wk_return": "—",
                })
                continue
    
            sec_median_dd = sector_median_drawdown.get(sector, drawdown)
            relative_weakness = drawdown - sec_median_dd
    
            a1 = (20 if drawdown >= 40 else 15 if drawdown >= 30 else 10 if drawdown >= 20 else 0)
            a2 = (10 if (promoter is not None and promoter >= 50) else 5 if (promoter is not None and promoter >= 35) else 0)
            a3 = (10 if (de_ratio is not None and de_ratio != 999 and de_ratio < 0.5) else 5 if (de_ratio is not None and de_ratio != 999 and de_ratio < 1.5) else 0)
            a4 = 5 if low_confirmed else 0
            a5 = (5 if relative_weakness >= 15 else 3 if relative_weakness >= 5 else 0)
            pillar_a = a1 + a2 + a3 + a4 + a5
    
            cross_week = None
            rsi_at_cross = None
            for w in range(3):
                r_now   = rsi_s.iloc[-(w+1)]
                r_prev  = rsi_s.iloc[-(w+2)]
                rs_now  = rsi_sma.iloc[-(w+1)]
                rs_prev = rsi_sma.iloc[-(w+2)]
                if r_now > rs_now and r_prev <= rs_prev:
                    cross_week = w
                    rsi_at_cross = r_now
                    break
    
            if cross_week is not None and rsi_at_cross is not None and rsi_at_cross >= 60:
                cross_week = None
                rsi_at_cross = None
    
            cross_failed = False
            if cross_week is not None and cross_week > 0 and rsi <= rsi_sm:
                cross_failed = True
                cross_week = None
                rsi_at_cross = None
    
            re_confirmed = False
            if cross_week is not None:
                for w2 in range(cross_week + 1, min(cross_week + 6, len(rsi_s) - 1)):
                    r_n, r_p   = rsi_s.iloc[-(w2+1)], rsi_s.iloc[-(w2+2)]
                    rs_n, rs_p = rsi_sma.iloc[-(w2+1)], rsi_sma.iloc[-(w2+2)]
                    if r_n > rs_n and r_p <= rs_p:
                        re_confirmed = True
                        break
    
            if   cross_week == 0: b1 = 30
            elif cross_week == 1: b1 = 22
            elif cross_week == 2: b1 = 15
            elif rsi > rsi_sm:    b1 = 8
            else:                 b1 = 0
    
            if re_confirmed:
                b1 += 5
    
            b2 = (20 if rsi < 30 else 15 if rsi < 40 else 10 if rsi < 50 else 0)
            b3 = (10 if vol_ratio >= 3 else 7 if vol_ratio >= 2 else 4 if vol_ratio >= 1.5 else 0)
            pillar_b = b1 + b2 + b3
            total    = pillar_a + pillar_b
    
            if cross_week is not None and total >= 50:
                phase = "Phase 2 - Entry Ready"
            elif rsi < rsi_sm and drawdown >= 20 and vol_ratio >= 1.5 and low_confirmed:
                phase = "Phase 1 - Watchlist"
            elif total >= 30:
                phase = "On Radar"
            else:
                phase = "Below Threshold"
    
            if cross_week is not None and rsi_at_cross is not None:
                q = 0
                q += 3 if rsi_at_cross >= 55 else (2 if rsi_at_cross >= 45 else (1 if rsi_at_cross >= 35 else 0))
                q += 3 if drawdown >= 55 else (2 if drawdown >= 45 else (1 if drawdown >= 35 else 0))
                q += 1 if contra_regime in ["CAUTION", "BEAR"] else 0
                quality_score  = q
                quality_bucket = ("PREMIUM" if q >= 6 else "HIGH" if q >= 4 else "MEDIUM" if q >= 2 else "LOW")
            else:
                quality_score, quality_bucket = 0, "—"
    
            archetype_a = (
                drawdown >= 45 and cross_week is not None and
                rsi_at_cross is not None and 45 <= rsi_at_cross <= 60 and
                vol_ratio >= 2.0
            )
    
            contra_rows.append({
                "week": week_tag, "ticker": ticker, "company": company, "sector": sector,
                "price": fmt(price, 2), "score": total, "phase": phase,
                "RSI": fmt(rsi, 1), "drawdown": fmt(drawdown, 1),
                "de_ratio": fmt(de_ratio, 2) if de_ratio != 999 else "—",
                "cross_week": cross_week if cross_week is not None else "—",
                "vol_ratio": fmt(vol_ratio, 2),
                "promoter": fmt(promoter, 1) if promoter is not None else "—",
                "sales_gr": fmt(sales_gr, 1) if sales_gr is not None else "—",
                "pat_gr": fmt(pat_gr, 1) if pat_gr is not None else "—",
                "pct_from_low": fmt(pct_from_low, 1),
                "low_confirmed": "Yes" if low_confirmed else "No",
                "pillar_a": pillar_a, "pillar_b": pillar_b,
                "quality_score": quality_score, "quality_bucket": quality_bucket,
                "archetype_a": "Yes" if archetype_a else "No",
                "cross_failed": "Yes" if cross_failed else "No",
                "re_confirmed": "Yes" if re_confirmed else "No",
                "roc_3": fmt(roc["roc_3"], 2), "roc_14": fmt(roc["roc_14"], 2),
                "roc_zscore": fmt(roc["roc_zscore"], 2),
                "roc_score": fmt(roc["roc_score"], 1), "roc_signal": roc["roc_signal"],
                # Filled in after the loop (conviction layer + backtest helper)
                "conviction_score": "—", "fund_pass": "—", "lev_pass": "—",
                "rsi_delta": "—", "dd_delta": "—", "action": "—", "wk_return": "—",
            })
    
        except Exception:
            continue
    
    # ── Conviction layer + weekly return (does not touch "score") ──────────
    for r in contra_rows:
        try:
            prev = prev_lookup.get(r["ticker"])
            if prev is None:
                continue
            cur_px, prev_px = _num(r["price"]), _num(prev.get("price"))
            if cur_px is not None and prev_px:
                r["wk_return"] = fmt((cur_px / prev_px - 1) * 100, 2)
            if str(r["phase"]).startswith("Filtered"):
                continue   # gated-out rows keep "—" placeholders
            r.update(compute_conviction(r, prev))
        except Exception as e:
            print(f"  ⚠️  conviction failed for {r.get('ticker')}: {e}")
    
    contra_df = pd.DataFrame(contra_rows) if contra_rows else pd.DataFrame(columns=[
        "week","ticker","company","sector","price","score","phase",
        "RSI","drawdown","de_ratio","cross_week","vol_ratio","promoter",
        "sales_gr","pat_gr","pct_from_low","low_confirmed","pillar_a","pillar_b",
        "quality_score","quality_bucket","archetype_a","cross_failed","re_confirmed",
        "roc_3","roc_14","roc_zscore","roc_score","roc_signal",
        "conviction_score","fund_pass","lev_pass","rsi_delta","dd_delta","action","wk_return"
    ])
    
    p2_df = contra_df[contra_df["phase"] == "Phase 2 - Entry Ready"].copy()
    p1_df = contra_df[contra_df["phase"] == "Phase 1 - Watchlist"].copy()
    
    print(f"""
{'='*65}
CONTRA SCREENER RESULTS
{'='*65}
  Total rows built    : {len(contra_df)}
  Phase 2 Entry Ready : {len(p2_df)}
  Phase 1 Watchlist   : {len(p1_df)}
  On Radar            : {len(contra_df[contra_df['phase']=='On Radar'])}
  Below Threshold     : {len(contra_df[contra_df['phase']=='Below Threshold'])}
  Archetype A signals : {len(contra_df[contra_df['archetype_a']=='Yes'])}
  Failed crosses       : {len(contra_df[contra_df['cross_failed']=='Yes'])}
  Re-confirmed crosses : {len(contra_df[contra_df['re_confirmed']=='Yes'])}
  Conviction BUY       : {len(contra_df[contra_df['action']=='BUY'])}
  Conviction VALUE TRAP: {len(contra_df[contra_df['action']=='VALUE TRAP'])}
{'='*65}
""")

    return contra_df, p2_df, p1_df
