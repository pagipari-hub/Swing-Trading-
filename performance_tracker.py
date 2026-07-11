# ============================================================
# Performance Tracker | Nifty 500 Screener
# Reads "Contra Data" + "Momentum Data" (the transposed, weekly
# sheets your screener already builds) and computes forward
# performance for every signal that has fired to date.
#
# Design notes:
#   - Uses ONLY data already sitting in the sheet (price + phase/
#     label rows per week). No re-fetch from yfinance needed.
#   - v1 tracks RAW forward performance: current return, peak
#     return since entry, drawdown from peak, weeks held.
#   - It does NOT yet replicate the exact Cell 9 exit rule (trail
#     stop on RSI-vs-14wk-SMA cross, hard stop at 52w low) because
#     that needs 14 consecutive weekly RSI readings to compute a
#     rolling SMA. Once you've run the weekly job for 14+ weeks,
#     tell me and I'll extend this to fire exits exactly like the
#     backtest does, so live results are apples-to-apples with the
#     validated 5-year numbers.
#   - Safe to run every week right after your existing export step
#     — it only reads, then writes one summary tab.
# ============================================================

import os
import json
import re
from datetime import datetime

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SHEET_NAME = "Nifty500 Screener"
OUTPUT_TAB = "Performance Tracker"

# Which phase/label values count as an "entry signal" per screener
CONTRA_ENTRY_PHASES = {"Phase 2 - Entry Ready"}
MOMENTUM_ENTRY_PREFIXES = ("🔥", "⚡")


# ── Auth (same pattern as the screener) ───────────────────────
def get_gspread_client():
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not sa_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON secret not set")
    sa_info = json.loads(sa_json)
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def week_sort_key(week_tag):
    """'2026-W24' -> (2026, 24), so week columns sort chronologically."""
    m = re.match(r"(\d{4})-W(\d{2})", str(week_tag))
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


# ── Read a transposed sheet back into a tidy structure ────────
def load_transposed_sheet(ws):
    """
    Returns:
      tickers_meta: {ticker: {"company":..., "sector":...}}
      week_cols:    sorted list of week tags present in the sheet
      metric_series: {(ticker, metric): {week_tag: value_str}}
    """
    rows = ws.get_all_values()
    if not rows:
        return {}, [], {}

    header = rows[0]
    # Fixed columns are: key, metric, ticker, company, sector, [is_cyclical]
    # Everything after that is a week column.
    if "is_cyclical" in header:
        first_week_idx = header.index("is_cyclical") + 1
    else:
        first_week_idx = 5  # key, metric, ticker, company, sector

    week_cols = [w for w in header[first_week_idx:] if w]
    week_cols_sorted = sorted(week_cols, key=week_sort_key)

    tickers_meta = {}
    metric_series = {}

    for row in rows[1:]:
        if len(row) < first_week_idx + 1:
            continue
        metric = row[1]
        ticker = row[2]
        company = row[3] if len(row) > 3 else ""
        sector = row[4] if len(row) > 4 else ""

        if ticker not in tickers_meta:
            tickers_meta[ticker] = {"company": company, "sector": sector}

        week_vals = {}
        for i, wk in enumerate(header[first_week_idx:]):
            col_idx = first_week_idx + i
            if col_idx < len(row) and wk:
                week_vals[wk] = row[col_idx]
        metric_series[(ticker, metric)] = week_vals

    return tickers_meta, week_cols_sorted, metric_series


def to_float(val):
    try:
        if val in (None, "", "—"):
            return None
        return float(val)
    except Exception:
        return None


# ── Core: build performance rows for one screener's signals ───
def compute_performance(tickers_meta, week_cols, metric_series,
                         phase_metric, is_entry_fn, screener_name):
    """
    For every (ticker, week) where is_entry_fn(phase_value) is True,
    walk forward through later weeks and compute return stats using
    the 'price' metric series for that ticker.
    """
    perf_rows = []

    for ticker, meta in tickers_meta.items():
        price_wk = metric_series.get((ticker, "price"), {})
        phase_wk = metric_series.get((ticker, phase_metric), {})
        if not price_wk or not phase_wk:
            continue

        for wi, wk in enumerate(week_cols):
            phase_val = phase_wk.get(wk, "—")
            if not is_entry_fn(phase_val):
                continue

            entry_price = to_float(price_wk.get(wk))
            if entry_price is None or entry_price <= 0:
                continue

            # Walk forward from this signal week to the latest week
            future_weeks = week_cols[wi:]
            peak_return = None
            peak_week = wk
            last_price = entry_price
            last_week = wk

            for fw in future_weeks:
                p = to_float(price_wk.get(fw))
                if p is None or p <= 0:
                    continue
                ret = (p / entry_price - 1) * 100
                if peak_return is None or ret > peak_return:
                    peak_return = ret
                    peak_week = fw
                last_price = p
                last_week = fw

            current_return = (last_price / entry_price - 1) * 100
            drawdown_from_peak = (
                (peak_return - current_return) if peak_return is not None else 0
            )
            weeks_held = week_sort_key(last_week)[1] - week_sort_key(wk)[1]
            # handles year rollover roughly; fine for a running tracker
            if weeks_held < 0:
                weeks_held += 52

            perf_rows.append({
                "screener": screener_name,
                "ticker": ticker,
                "company": meta.get("company", ""),
                "sector": meta.get("sector", ""),
                "signal_week": wk,
                "entry_price": round(entry_price, 2),
                "latest_week": last_week,
                "latest_price": round(last_price, 2),
                "weeks_held": weeks_held,
                "current_return_pct": round(current_return, 2),
                "peak_return_pct": round(peak_return, 2) if peak_return is not None else "—",
                "peak_week": peak_week,
                "drawdown_from_peak_pct": round(drawdown_from_peak, 2),
                "status": "OPEN" if last_week == week_cols[-1] else "STALE_DATA",
            })

    return pd.DataFrame(perf_rows)


def is_contra_entry(phase_val):
    return phase_val in CONTRA_ENTRY_PHASES


def is_momentum_entry(label_val):
    return isinstance(label_val, str) and label_val.startswith(MOMENTUM_ENTRY_PREFIXES)


# ── Summary stats (win rate, avg return) ───────────────────────
def summarize(perf_df):
    if perf_df.empty:
        return pd.DataFrame([{"metric": "No signals tracked yet", "value": "—"}])

    n = len(perf_df)
    wins = (perf_df["current_return_pct"] > 0).sum()
    win_rate = round(wins / n * 100, 1) if n else 0
    avg_ret = round(perf_df["current_return_pct"].mean(), 2)
    avg_winner = round(
        perf_df.loc[perf_df["current_return_pct"] > 0, "current_return_pct"].mean(), 2
    ) if wins else 0
    losers = perf_df["current_return_pct"] <= 0
    avg_loser = round(
        perf_df.loc[losers, "current_return_pct"].mean(), 2
    ) if losers.sum() else 0
    avg_weeks = round(perf_df["weeks_held"].mean(), 1)

    return pd.DataFrame([
        {"metric": "Total signals tracked", "value": n},
        {"metric": "Win rate (current return > 0)", "value": f"{win_rate}%"},
        {"metric": "Avg return (all signals)", "value": f"{avg_ret}%"},
        {"metric": "Avg winner", "value": f"{avg_winner}%"},
        {"metric": "Avg loser", "value": f"{avg_loser}%"},
        {"metric": "Avg weeks held", "value": avg_weeks},
        {"metric": "Note", "value": "Raw forward return — exit-rule simulation "
                                     "(trail stop / hard stop) not yet applied. "
                                     "Add once 14+ weeks of history exist."},
    ])


def col_letter(n):
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def write_sheet(ws, df, start_row=1, start_col=1, title=None):
    values = []
    if title:
        values.append([title])
        values.append([])
    if not df.empty:
        values.append(list(df.columns))
        values.extend(df.astype(str).values.tolist())
    else:
        values.append(["No data"])
    rng = f"{col_letter(start_col)}{start_row}"
    ws.update(values=values, range_name=rng)


def main():
    gc = get_gspread_client()
    sh = gc.open(SHEET_NAME)

    ws_contra = sh.worksheet("Contra Data")
    ws_mom = sh.worksheet("Momentum Data")

    contra_meta, contra_weeks, contra_series = load_transposed_sheet(ws_contra)
    mom_meta, mom_weeks, mom_series = load_transposed_sheet(ws_mom)

    print(f"Contra Data: {len(contra_meta)} tickers, {len(contra_weeks)} weeks")
    print(f"Momentum Data: {len(mom_meta)} tickers, {len(mom_weeks)} weeks")

    contra_perf = compute_performance(
        contra_meta, contra_weeks, contra_series,
        phase_metric="phase", is_entry_fn=is_contra_entry,
        screener_name="Contra",
    )
    mom_perf = compute_performance(
        mom_meta, mom_weeks, mom_series,
        phase_metric="label", is_entry_fn=is_momentum_entry,
        screener_name="Momentum",
    )

    all_perf = pd.concat([contra_perf, mom_perf], ignore_index=True)
    if not all_perf.empty:
        all_perf = all_perf.sort_values(
            ["screener", "signal_week"], ascending=[True, False]
        )

    contra_summary = summarize(contra_perf)
    mom_summary = summarize(mom_perf)

    try:
        ws_out = sh.worksheet(OUTPUT_TAB)
    except gspread.WorksheetNotFound:
        ws_out = sh.add_worksheet(title=OUTPUT_TAB, rows=5000, cols=20)

    ws_out.clear()

    # Layout: summaries at top, full signal log below
    row_cursor = 1
    write_sheet(ws_out, contra_summary, start_row=row_cursor, start_col=1,
                title=f"CONTRA SUMMARY (as of {datetime.now().strftime('%d %b %Y %H:%M')})")
    row_cursor += len(contra_summary) + 3

    write_sheet(ws_out, mom_summary, start_row=row_cursor, start_col=1,
                title="MOMENTUM SUMMARY")
    row_cursor += len(mom_summary) + 3

    write_sheet(ws_out, all_perf, start_row=row_cursor, start_col=1,
                title="SIGNAL LOG (every entry fired to date)")

    print(f"✅ Performance Tracker written — {len(all_perf)} signals logged")
    print(f"🔗 {sh.url}")


if __name__ == "__main__":
    main()
