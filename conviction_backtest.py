# ============================================================
# conviction_backtest.py — Does the conviction layer actually help?
#
# Reads the "Contra Data" sheet (same transposed weekly structure
# performance_tracker.py reads), replays compute_conviction() for
# EVERY past week using that week's stored inputs (price, RSI,
# drawdown, sales_gr, pat_gr, de_ratio, quality_bucket,
# low_confirmed, total_score) and last week's RSI/drawdown, then
# compares forward returns by cohort.
#
# Needs no new data: the sheet already holds every input, so this
# works on W24 onwards immediately — you do not have to wait for
# the live conviction columns to build up history.
#
# Forward return = price N week-columns later / price this week - 1,
# for N = 1, 4, 8, 12 weeks, plus "td" = hold until the latest week.
# Run:  python conviction_backtest.py      (needs GOOGLE_SERVICE_ACCOUNT_JSON)
# Output: console table + conviction_backtest.csv (one row per ticker-week)
# ============================================================

import pandas as pd

from config import SHEET_NAME
from sheets_io import get_gspread_client
from performance_tracker import load_transposed_sheet, to_float
from contra_screener import compute_conviction

FWD_WEEKS = (1, 4, 8, 12)   # plus 'to_date' = hold until the latest week in the sheet


def build_history(tickers_meta, week_cols, metric_series):
    """One row per (ticker, week) with conviction fields + forward returns."""
    out = []
    for ticker, meta in tickers_meta.items():
        def val(metric, wk):
            return metric_series.get((ticker, metric), {}).get(wk, "—")

        prices = {wk: to_float(val("price", wk)) for wk in week_cols}

        for i, wk in enumerate(week_cols):
            price = prices.get(wk)
            score = to_float(val("total_score", wk))
            if price is None or score is None or score == 0:
                continue   # no data, or gated-out ("Filtered") row

            row = {
                "score": score, "sector": meta.get("sector", ""),
                "RSI": val("RSI", wk), "drawdown": val("drawdown", wk),
                "de_ratio": val("de_ratio", wk),
                "sales_gr": val("sales_gr", wk), "pat_gr": val("pat_gr", wk),
                "quality_bucket": val("quality_bucket", wk),
                "low_confirmed": val("low_confirmed", wk),
            }
            prev = None
            if i > 0:
                pw = week_cols[i - 1]
                prev = {"RSI": val("RSI", pw), "drawdown": val("drawdown", pw)}

            rec = {"ticker": ticker, "week": wk, "phase": val("phase", wk),
                   "total_score": score}
            rec.update(compute_conviction(row, prev))

            for n in FWD_WEEKS:
                p_fwd = prices.get(week_cols[i + n]) if i + n < len(week_cols) else None
                rec[f"fwd_{n}w"] = (p_fwd / price - 1) * 100 if p_fwd else None
            later = [prices[w] for w in week_cols[i + 1:] if prices.get(w)]
            rec["fwd_td"] = (later[-1] / price - 1) * 100 if later else None
            out.append(rec)
    return pd.DataFrame(out)


def cohorts(df):
    """Cohort name -> boolean mask. 'Strict filter' = the Contra Entry sheet rule."""
    d = pd.to_numeric(df["dd_delta"], errors="coerce")
    r = pd.to_numeric(df["rsi_delta"], errors="coerce")
    strict = ((df["total_score"] >= 50) & (df["conviction_score"] >= 70) &
              (df["fund_pass"] == "PASS") & (r > 0) & (d > 0) & (df["lev_pass"] == "PASS"))
    p2 = df["phase"] == "Phase 2 - Entry Ready"
    return {
        "All screened rows":           df["total_score"] > 0,
        "Phase 2 (old rule)":          p2,
        "total_score >= 50":           df["total_score"] >= 50,
        "BUY":                         df["action"] == "BUY",
        "VALUE TRAP":                  df["action"] == "VALUE TRAP",
        "score>=50 but WATCH":         (df["total_score"] >= 50) & (df["action"] == "WATCH"),
        "Phase 2 AND BUY":             p2 & (df["action"] == "BUY"),
        "Strict sheet filter":         strict,
    }


def summarize(df):
    """One compact cell per horizon: 'avg% (win%) nN'. n shrinks at longer horizons."""
    horizons = [(f"{n}w", f"fwd_{n}w") for n in FWD_WEEKS] + [("td", "fwd_td")]
    rows = []
    for name, mask in cohorts(df).items():
        sub = df[mask]
        rec = {"cohort": name, "n": len(sub)}
        for label, col in horizons:
            sr = sub[col].dropna()
            rec[label] = (f"{sr.mean():.1f} ({(sr > 0).mean() * 100:.0f}%) n{len(sr)}"
                          if len(sr) else "-")
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    gc = get_gspread_client()
    ws = gc.open(SHEET_NAME).worksheet("Contra Data")
    tickers_meta, week_cols, metric_series = load_transposed_sheet(ws)
    print(f"Loaded {len(tickers_meta)} tickers x {len(week_cols)} weeks "
          f"({week_cols[0]} .. {week_cols[-1]})")

    df = build_history(tickers_meta, week_cols, metric_series)
    if df.empty:
        print("No usable rows."); return
    df.to_csv("conviction_backtest.csv", index=False)

    print("\nForward return by cohort, cell = avg% (win%) nSamples. "
          "td = hold till latest week. A ticker can appear in many weeks — "
          "overlapping samples, treat as indicative:\n")
    print(summarize(df).to_string(index=False))
    print("\nWhat to look for: BUY / Strict filter should beat 'Phase 2 (old rule)' "
          "on avg AND win%, and VALUE TRAP should trail. Small n (<30) = noise.")


if __name__ == "__main__":
    main()
