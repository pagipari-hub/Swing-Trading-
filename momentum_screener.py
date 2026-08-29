# ============================================================
# momentum_screener.py — Momentum strategy.
#
# Simpler binary Risk-On/Risk-Off regime filter, intentionally kept
# structurally separate from Contra's 4-tier regime and two-phase
# entry logic (see contra_screener.py).
# ============================================================

import numpy as np
import pandas as pd
import ta

from config import (
    NIFTY500_INDEX, WEEKS_VALID, RSI_UPPER, LIQ_HARD, VOL_WINDOW,
    RAW_MAX, FLOOR_SALES_GR_MIN, FLOOR_PAT_GR_MIN, FLOOR_DE_RATIO_MAX,
    FLOOR_PROMOTER_MIN, CYCLICAL_NSE,
)
from helpers import safe_get, pct_ret, yoy_growth, fmt
from roc_score import compute_momentum_roc


def run_momentum_screener(nifty500, price_data, fund_data, q_fin_data,
                           sector_lookup, company_lookup, promoter_data, week_tag):
    """
    Returns (momentum_df, mom_entry, mom_emerging, mom_watch, regime,
    idx_1m, idx_3m, idx_6m, n_rsi, n_price, n_sma30).

    The last three (n_rsi, n_price, n_sma30) are returned for the
    Telegram market-condition line in export.py; they're set to None
    if the regime check itself failed.
    """
    print("=" * 65)
    print("MOMENTUM SCREENER — Full Build")
    print("=" * 65)

    print("\n[1/5] Market Regime...")
    regime = "Risk-On"; regime_mult = 1.0
    nifty_close = None
    n_rsi = n_price = n_sma30 = None   # stay None if the regime check below fails
    
    try:
        nf = price_data.get(NIFTY500_INDEX)
        if nf is None:
            closes = [price_data[t]["Close"].squeeze()
                      for t in nifty500
                      if price_data.get(t) is not None and len(price_data[t]) > 60]
            nifty_close = pd.concat(closes, axis=1).median(axis=1)
        else:
            nifty_close = nf["Close"].squeeze()
    
        n_sma30 = nifty_close.rolling(30).mean().iloc[-1]
        n_rsi   = ta.momentum.RSIIndicator(nifty_close, window=14).rsi().iloc[-1]
        n_price = nifty_close.iloc[-1]
    
        if n_price < n_sma30 and n_rsi < 50:
            regime = "Risk-Off"; regime_mult = 0.85
            print(f"  ⚠️  RISK-OFF — RSI {n_rsi:.1f}, below 30W SMA")
        else:
            print(f"  ✅ Risk-On — RSI {n_rsi:.1f}")
    except Exception as e:
        print(f"  ⚠️  Regime check failed ({e})")
    
    print("\n[2/5] Index reference returns...")
    idx_1m = idx_3m = idx_6m = idx_4w = 0
    try:
        if nifty_close is not None:
            idx_1m = pct_ret(nifty_close, 4)  or 0
            idx_3m = pct_ret(nifty_close, 13) or 0
            idx_6m = pct_ret(nifty_close, 26) or 0
            idx_4w = idx_1m
            print(f"  1M: {idx_1m:.1f}%  3M: {idx_3m:.1f}%  6M: {idx_6m:.1f}%")
    except Exception:
        pass
    
    print("\n[3/5] Sector metrics...")
    sec_bucket = {}
    for ticker in nifty500:
        df = price_data.get(ticker)
        if df is None or len(df) < 60:
            continue
        try:
            close  = df["Close"].squeeze()
            sma20  = close.rolling(20).mean().iloc[-1]
            r4w    = pct_ret(close, 4)
            r6m    = pct_ret(close, 26)
            ab20   = close.iloc[-1] > sma20
            sec    = sector_lookup.get(ticker, "Unknown")
            sec_bucket.setdefault(sec, []).append({"r4w": r4w, "r6m": r6m, "ab20": ab20})
        except Exception:
            pass
    
    sector_stats    = {}
    sector_vs_idx   = {}
    sector_strength = {}
    
    for sec, items in sec_bucket.items():
        r4    = [x["r4w"] for x in items if x["r4w"] is not None]
        r6    = [x["r6m"] for x in items if x["r6m"] is not None]
        ab    = [x["ab20"] for x in items]
        med4w = np.median(r4) if r4 else 0
        med6m = np.median(r6) if r6 else 0
        brd   = (sum(ab) / len(ab) * 100) if ab else 0
        sector_stats[sec]  = {"4w": med4w, "6m": med6m, "breadth": brd}
        sector_vs_idx[sec] = med4w - idx_4w
    
    sorted_secs = sorted(sector_stats.items(), key=lambda x: x[1]["4w"], reverse=True)
    for rank, (sec, _) in enumerate(sorted_secs):
        sector_strength[sec] = rank + 1
    
    total_secs = len(sector_strength)
    
    def get_sector_cap(sec, score):
        if score >= 80:
            return 1
        rank     = sector_strength.get(sec, total_secs)
        strength = total_secs - rank + 1
        return min(max(1, round(strength / (total_secs / 3))), 3)
    
    print(f"  Sector metrics built for {len(sector_stats)} sectors")
    
    print("\n[4/5] Scoring stocks...")
    
    momentum_rows = []
    
    for ticker in nifty500:
        df   = price_data.get(ticker)
        info = fund_data.get(ticker, {})
        qfin = q_fin_data.get(ticker)
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
            is_cycl = sector in CYCLICAL_NSE
    
            avg_vol20 = volume.iloc[-21:-1].mean()
            traded_cr = (price * avg_vol20) / 1e7
            if traded_cr < LIQ_HARD:
                momentum_rows.append({
                    "week": week_tag, "ticker": ticker,
                    "company": company_lookup.get(ticker, ticker),
                    "sector": sector_lookup.get(ticker, "Unknown"),
                    "price": fmt(price, 2), "score": 0, "label": "Filtered (Low Liq)",
                    "RSI": "—", "rs_periods": "—", "earn_gr": "—", "pct_from_high": "—",
                    "vol_ratio": "—", "new_52w_high": "—", "new_5y_high": "—",
                    "pillar_a": 0, "pillar_b": 0, "pillar_c": 0, "pillar_d": 0,
                    "cross_failed": "—", "re_confirmed": "—",
                    "roc_5": "—", "roc_20": "—", "roc_score": "—", "roc_signal": "—",
                })
                continue
    
            t_eps   = safe_get(info, "trailingEps",    None)
            f_eps   = safe_get(info, "forwardEps",     None)
            earn_gr = safe_get(info, "earningsGrowth", 0)
            pe      = safe_get(info, "trailingPE",     999)
            if earn_gr and abs(earn_gr) < 5:
                earn_gr = earn_gr * 100
    
            if (t_eps is not None and t_eps < 0) and (f_eps is not None and f_eps < 0):
                momentum_rows.append({
                    "week": week_tag, "ticker": ticker, "company": company, "sector": sector,
                    "price": fmt(price, 2), "score": 0, "label": "Filtered (Neg EPS)",
                    "RSI": "—", "rs_periods": "—", "earn_gr": "—", "pct_from_high": "—",
                    "vol_ratio": "—", "new_52w_high": "—", "new_5y_high": "—",
                    "pillar_a": 0, "pillar_b": 0, "pillar_c": 0, "pillar_d": 0,
                    "cross_failed": "—", "re_confirmed": "—",
                    "roc_5": "—", "roc_20": "—", "roc_score": "—", "roc_signal": "—",
                })
                continue
    
            sales_gr = pat_gr = fwd_eps_gr = None
            if qfin is not None and not qfin.empty:
                for k in ["Total Revenue", "Revenue"]:
                    if k in qfin.index:
                        sales_gr = yoy_growth(qfin.loc[k]); break
                for k in ["Net Income", "Net Income Common Stockholders"]:
                    if k in qfin.index:
                        pat_gr = yoy_growth(qfin.loc[k]); break
            if t_eps and f_eps and t_eps > 0:
                fwd_eps_gr = ((f_eps - t_eps) / abs(t_eps)) * 100
    
            close_52w_prior = close.iloc[-53:-1].max() if len(close) >= 53 else close.iloc[:-1].max()
            close_5y_prior  = close.iloc[:-1].max()
    
            high_52w       = high_s.iloc[-52:].max()
            pct_from_high  = (high_52w - price) / high_52w * 100
    
            is_new_52w_high = price >= close_52w_prior
            is_new_5y_high  = price >= close_5y_prior
    
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma20_prev3 = close.rolling(20).mean().iloc[-4]
    
            rsi_s = ta.momentum.RSIIndicator(close, window=14).rsi()
            if rsi_s is None or rsi_s.dropna().shape[0] < 5:
                continue
            rsi_now = rsi_s.iloc[-1]
    
            rsi_slope_ok = (
                rsi_s.iloc[-2] > rsi_s.iloc[-3] and
                rsi_s.iloc[-3] > rsi_s.iloc[-4]
            )
    
            cross_week = rsi_at_cross = None
            for w in range(WEEKS_VALID + 1):
                r_now  = rsi_s.iloc[-(w+1)]
                r_prev = rsi_s.iloc[-(w+2)]
                if r_now >= 60 and r_prev < 60:
                    cross_week = w; rsi_at_cross = r_now; break
    
            if cross_week is not None and rsi_at_cross is not None and rsi_at_cross > RSI_UPPER:
                cross_week = rsi_at_cross = None
    
            cross_failed = False
            if cross_week is not None and cross_week > 0 and rsi_now < 60:
                cross_failed = True
                cross_week = None
                rsi_at_cross = None
    
            re_confirmed = False
            if cross_week is not None:
                for w2 in range(cross_week + 1, min(cross_week + 6, len(rsi_s) - 1)):
                    r_n, r_p = rsi_s.iloc[-(w2+1)], rsi_s.iloc[-(w2+2)]
                    if r_n >= 60 and r_p < 60:
                        re_confirmed = True
                        break
    
            vol_avg   = volume.iloc[-VOL_WINDOW-1:-1].mean()
            vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0
    
            vol_weeks_above = 0
            for w in range(5):
                try:
                    if volume.iloc[-(w+1)] > vol_avg:
                        vol_weeks_above += 1
                except Exception:
                    pass
    
            # ROC forward-test overlay — additive only, does NOT feed into
            # pillar_a/b/c/d or total below. See roc_score.py header.
            roc = compute_momentum_roc(close, vol_ratio, rsi_now)

            r1m = pct_ret(close, 4);  rs1 = (r1m or 0) - idx_1m
            r3m = pct_ret(close, 13); rs3 = (r3m or 0) - idx_3m
            r6m = pct_ret(close, 26); rs6 = (r6m or 0) - idx_6m
            rs_periods = sum(1 for rs in [rs1, rs3, rs6] if rs > 0)
    
            s_stats  = sector_stats.get(sector, {"4w": 0, "6m": 0, "breadth": 0})
            sec_4w   = s_stats["4w"]
            sec_6m   = s_stats["6m"]
            sec_brd  = s_stats["breadth"]
            sec_vs_i = sector_vs_idx.get(sector, 0)
            outperf  = (r6m or 0) - sec_6m
    
            is_watch = (
                cross_week is None and
                50 <= rsi_now < 60 and
                price > sma20 and price > sma50 and
                pct_from_high <= 15
            )
            is_emerging = (
                cross_week is None and
                52 <= rsi_now < 60 and
                price > sma20 and price > sma50 and
                pct_from_high <= 20 and
                (rs1 > 0 or rs3 > 0)
            )
    
            # promoter holding via Screener.in scrape (see promoter_scraper.py) —
            # yfinance's heldPercentInsiders does not populate for NSE tickers.
            # Already a percentage from Screener.in, no *100 conversion.
            promoter_pct  = promoter_data.get(ticker)
            de_ratio_mom  = safe_get(info, "debtToEquity", None)
            fails_floor = (
                (sales_gr     is not None and sales_gr     <= FLOOR_SALES_GR_MIN) or
                (pat_gr       is not None and pat_gr       <= FLOOR_PAT_GR_MIN)   or
                (de_ratio_mom is not None and de_ratio_mom  >  FLOOR_DE_RATIO_MAX) or
                (promoter_pct is not None and promoter_pct  <  FLOOR_PROMOTER_MIN)
            )
    
            a1 = (12 if pct_from_high <= 5  else
                   9 if pct_from_high <= 10 else
                   6 if pct_from_high <= 15 else
                   3 if pct_from_high <= 20 else 0)
    
            a2 = 6 if is_new_52w_high else 0
            a3 = 8 if is_new_5y_high else 0
    
            a4 = (8 if (price > sma20 and sma20 > sma50) else
                  4 if price > sma20 else 0)
    
            a5 = 0
            a5 += 6 if rs1 > 2 else (3 if rs1 > 0 else 0)
            a5 += 4 if rs3 > 2 else (2 if rs3 > 0 else 0)
            a5 += 2 if rs6 > 2 else (1 if rs6 > 0 else 0)
    
            a6 = (4 if (earn_gr and earn_gr > 20) else
                  2 if (earn_gr and earn_gr > 10) else
                  1 if (earn_gr and earn_gr > 0)  else 0)
    
            pillar_a = a1 + a2 + a3 + a4 + a5 + a6
    
            if cross_week is None:
                b1 = 0
            else:
                hot = rsi_at_cross if rsi_at_cross is not None else 60
                if cross_week == 0:
                    b1 = 20 if hot <= 65 else (14 if hot <= 70 else 8)
                elif cross_week == 1:
                    b1 = 15 if hot <= 65 else (10 if hot <= 70 else 5)
                else:
                    b1 = 10 if hot <= 65 else (6 if hot <= 70 else 3)
    
            if re_confirmed:
                b1 += 3
    
            b2 = 5 if (b1 > 0 and rsi_slope_ok) else 0
    
            tq_conditions = [
                sma20 > sma50,
                sma20 > sma20_prev3,
                price > sma20,
            ]
            tq_count = sum(1 for c in tq_conditions if c)
            b3 = 10 if tq_count == 3 else (5 if tq_count == 2 else 0)
    
            pillar_b = b1 + b2 + b3
    
            c1 = 5 if sec_4w>=5 else (3 if sec_4w>=2 else (1 if sec_4w>=0 else 0))
            c2 = 5 if sec_vs_i>=3 else (3 if sec_vs_i>=1 else (1 if sec_vs_i>=0 else 0))
            c3 = 5 if sec_brd>=70 else (3 if sec_brd>=60 else (1 if sec_brd>=50 else 0))
            pillar_c = c1+c2+c3
    
            d1 = (5 if vol_ratio >= 3   else
                  3 if vol_ratio >= 2   else
                  1 if vol_ratio >= 1.5 else 0)
    
            d2 = (8 if vol_weeks_above == 5 else
                  5 if vol_weeks_above == 4 else
                  3 if vol_weeks_above == 3 else 0)
    
            pillar_d = d1 + d2
    
            raw   = pillar_a + pillar_b + pillar_c + pillar_d
            total = round(round((raw / RAW_MAX) * 100) * regime_mult)
    
            if fails_floor:
                label = "Filtered (Weak Fundamentals)"
                total = 0
            elif cross_week is not None:
                label = ("🔥 High Momentum"       if total >= 80 else
                         "⚡ Momentum Candidate"  if total >= 65 else
                         "👀 Weak Signal")
            elif is_emerging:
                label = "🟢 Emerging Momentum"
            elif is_watch:
                label = "👁️ Watch"
            else:
                label = "Below Threshold"
    
            earn_gr_out = pat_gr if pat_gr is not None else earn_gr
    
            momentum_rows.append({
                "week": week_tag, "ticker": ticker, "company": company, "sector": sector,
                "price": fmt(price, 2), "score": total, "label": label,
                "RSI": fmt(rsi_now, 1), "rs_periods": rs_periods,
                "earn_gr": fmt(earn_gr_out, 1) if earn_gr_out is not None else "—",
                "pct_from_high": fmt(pct_from_high, 1), "vol_ratio": fmt(vol_ratio, 2),
                "new_52w_high": "Yes" if is_new_52w_high else "No",
                "new_5y_high": "Yes" if is_new_5y_high else "No",
                "pillar_a": pillar_a, "pillar_b": pillar_b,
                "pillar_c": pillar_c, "pillar_d": pillar_d,
                "cross_failed": "Yes" if cross_failed else "No",
                "re_confirmed": "Yes" if re_confirmed else "No",
                "roc_5": fmt(roc["roc_5"], 2), "roc_20": fmt(roc["roc_20"], 2),
                "roc_score": fmt(roc["roc_score"], 1), "roc_signal": roc["roc_signal"],
            })
    
        except Exception:
            continue
    
    momentum_df = pd.DataFrame(momentum_rows) if momentum_rows else pd.DataFrame(columns=[
        "week","ticker","company","sector","price","score","label",
        "RSI","rs_periods","earn_gr","pct_from_high","vol_ratio",
        "new_52w_high","new_5y_high",
        "pillar_a","pillar_b","pillar_c","pillar_d","cross_failed","re_confirmed",
        "roc_5","roc_20","roc_score","roc_signal"
    ])
    
    mom_entry    = momentum_df[momentum_df["label"].str.startswith(("🔥","⚡"), na=False)].copy()
    mom_emerging = momentum_df[momentum_df["label"] == "🟢 Emerging Momentum"].copy()
    mom_watch    = momentum_df[momentum_df["label"] == "👁️ Watch"].copy()
    
    print(f"""
{'='*65}
MOMENTUM SCREENER | Regime: {regime}
{'='*65}
  Total rows built   : {len(momentum_df)}
  Entry candidates   : {len(mom_entry)}
  Emerging Momentum  : {len(mom_emerging)}
  Watch list         : {len(mom_watch)}
  Failed crosses      : {len(momentum_df[momentum_df['cross_failed']=='Yes'])}
  Re-confirmed crosses: {len(momentum_df[momentum_df['re_confirmed']=='Yes'])}
{'='*65}
""")

    return (momentum_df, mom_entry, mom_emerging, mom_watch, regime,
            idx_1m, idx_3m, idx_6m, n_rsi, n_price, n_sma30)
