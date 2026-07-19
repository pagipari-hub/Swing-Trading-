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
)
from helpers import safe_get, yoy_growth, fmt


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


def run_contra_screener(nifty500, price_data, fund_data, q_fin_data,
                         sector_lookup, company_lookup, promoter_data,
                         sector_median_drawdown, contra_regime, week_tag):
    """Returns (contra_df, p2_df, p1_df)."""
    print("\n" + "=" * 65)
    print("CONTRA SCREENER")
    print("=" * 65)

    contra_rows = []
    
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
            })
    
        except Exception:
            continue
    
    contra_df = pd.DataFrame(contra_rows) if contra_rows else pd.DataFrame(columns=[
        "week","ticker","company","sector","price","score","phase",
        "RSI","drawdown","de_ratio","cross_week","vol_ratio","promoter",
        "sales_gr","pat_gr","pct_from_low","low_confirmed","pillar_a","pillar_b",
        "quality_score","quality_bucket","archetype_a","cross_failed","re_confirmed"
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
{'='*65}
""")

    return contra_df, p2_df, p1_df
