"""
main_momentum.py
=================
Standalone entry point for the stock-level Momentum screener.
Deliberately decoupled from main.py (Contra + Sector) -- see momentum.yml,
which triggers this via workflow_dispatch only (no cron).

Shares all underlying modules (config.py, universe.py, data_fetch.py,
helpers.py, sheets_io.py, telegram_notify.py, promoter_scraper.py) with
the rest of the pipeline -- only the orchestration/trigger is separate.

Wired against the actual signatures in your uploaded momentum_screener.py,
sheets_io.py, and telegram_notify.py.
"""

from datetime import datetime

from config import WEEK_TAG, SHEET_NAME  # ADAPT: confirm SHEET_NAME lives in config.py

from universe import load_nifty500_universe
from data_fetch import load_market_data
from promoter_scraper import load_promoter_data
from momentum_screener import run_momentum_screener
from sheets_io import get_gspread_client, ensure_worksheet, build_transposed_sheet
from telegram_notify import send_telegram

MOMENTUM_TAB_NAME = "Momentum Data"  # ADAPT if your existing tab is named differently

# Metrics written as rows under the transposed-sheet structure. Excludes
# week/ticker/company/sector (those are the fixed leading columns) and
# "price", which export.py may or may not track weekly -- add it back in
# if your existing Momentum Data tab already has a price row.
MOMENTUM_METRICS = [
    "score", "label", "RSI", "rs_periods", "earn_gr", "pct_from_high",
    "vol_ratio", "new_52w_high", "new_5y_high",
    "pillar_a", "pillar_b", "pillar_c", "pillar_d",
    "cross_failed", "re_confirmed",
    # ROC forward-test overlay (roc_score.py) — additive, not part of total_score
    "roc_5", "roc_20", "roc_score", "roc_signal",
]


def main() -> None:
    print("✅ Momentum standalone run starting...")
    print(f"📅 Run date : {datetime.today().strftime('%d %b %Y')}")
    print(f"🗓️  Week tag : {WEEK_TAG}")

    nifty500_df, nifty500, sector_lookup, company_lookup = load_nifty500_universe()
    price_data, fund_data, q_fin_data = load_market_data(nifty500)
    promoter_data = load_promoter_data(nifty500)

    (momentum_df, mom_entry, mom_emerging, mom_watch, regime,
     idx_1m, idx_3m, idx_6m, n_rsi, n_price, n_sma30) = run_momentum_screener(
        nifty500, price_data, fund_data, q_fin_data,
        sector_lookup, company_lookup, promoter_data, WEEK_TAG,
    )

    if momentum_df.empty:
        print("[main_momentum] No rows produced -- skipping Sheets/Telegram.")
        return

    # ── Sheets export ──────────────────────────────────────────────────
    # NOTE (flagged, not guessed silently): sheets_io.py's header comment
    # says the Momentum tab carries an is_cyclical column, but the rows
    # momentum_screener.py builds don't actually include an "is_cyclical"
    # key -- it's computed per ticker (is_cycl) and never added to
    # momentum_rows. Defaulting has_cyclical=False so no empty column gets
    # written. If your live "Momentum Data" tab already has that column
    # populated, export.py must be patching it in separately -- tell me
    # and I'll add it here to match.
    client = get_gspread_client()
    sh = client.open(SHEET_NAME)
    ws = ensure_worksheet(sh, MOMENTUM_TAB_NAME)
    build_transposed_sheet(ws, momentum_df, MOMENTUM_METRICS, WEEK_TAG, has_cyclical=False)

    # ── Telegram summary ───────────────────────────────────────────────
    message = format_momentum_message(
        momentum_df, mom_entry, mom_emerging, mom_watch, regime,
        idx_1m, idx_3m, idx_6m, n_rsi,
    )
    send_telegram(message)

    print(f"[main_momentum] Done. {len(momentum_df)} rows processed.")


def format_momentum_message(momentum_df, mom_entry, mom_emerging, mom_watch,
                             regime, idx_1m, idx_3m, idx_6m, n_rsi) -> str:
    lines = ["📈 <b>Momentum Screener — Manual Run</b>", f"Week: {WEEK_TAG}", ""]

    regime_line = f"Regime: {regime}"
    if n_rsi is not None:
        regime_line += f" (Nifty500 RSI {n_rsi:.1f})"
    lines.append(regime_line)

    if idx_1m or idx_3m or idx_6m:
        lines.append(f"Nifty500 1M/3M/6M: {idx_1m:.1f}% / {idx_3m:.1f}% / {idx_6m:.1f}%")

    lines.append("")
    lines.append(f"🔥⚡ Entry candidates : {len(mom_entry)}")
    lines.append(f"🟢 Emerging Momentum : {len(mom_emerging)}")
    lines.append(f"👁️ Watch list        : {len(mom_watch)}")

    if not mom_entry.empty:
        lines.append("\n<b>Top Entry Candidates:</b>")
        top = mom_entry.copy()
        top["score"] = top["score"].astype(float)
        top = top.sort_values("score", ascending=False).head(10)
        for _, r in top.iterrows():
            lines.append(f"• {r['ticker']} ({r['sector']}) — {r['label']} | score {r['score']:.0f}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
