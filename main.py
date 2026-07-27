# ============================================================
# main.py — Weekly pipeline entry point.
# Run this via GitHub Actions: python main.py
#
# Momentum has been split out — it now runs standalone via
# main_momentum.py, triggered manually (see momentum.yml).
# This file no longer imports or calls momentum_screener at all.
# ============================================================

from datetime import datetime
from config import WEEK_TAG

from universe import load_nifty500_universe
from data_fetch import load_market_data
from promoter_scraper import load_promoter_data
from contra_screener import (
    compute_sector_drawdown_baseline,
    compute_contra_regime,
    run_contra_screener,
)
from sector_screener import run_sector_screener
from export import export_and_notify

# ADAPT: swap for whatever your telegram_notify.py actually exposes
# (used only for the sector Telegram section — Contra's own Telegram
# message still goes out through export_and_notify() as before).
from telegram_notify import send_telegram_message


def main():
    print("✅ All libraries loaded.")
    print(f"📅 Run date   : {datetime.today().strftime('%d %b %Y')}")
    print(f"🗓️  Week tag   : {WEEK_TAG}")
    print("📊 Universe   : Nifty 500 | Contra Screener + Sector Signals")

    # ── Universe (with 3-tier fallback: live NSE → /tmp cache → repo seed) ──
    nifty500_df, nifty500, sector_lookup, company_lookup = load_nifty500_universe()

    # ── Market data (price/fund cached weekly, q_fin always fresh) ─────────
    price_data, fund_data, q_fin_data = load_market_data(nifty500)

    # ── Promoter holding (Screener.in scrape, cached ~25 days — yfinance
    # does not reliably provide this field for NSE tickers) ────────────────
    promoter_data = load_promoter_data(nifty500)

    # ── Contra ───────────────────────────────────────────────────────────
    sector_median_drawdown = compute_sector_drawdown_baseline(
        nifty500, price_data, sector_lookup
    )
    contra_regime = compute_contra_regime(nifty500, price_data)

    contra_df, p2_df, p1_df = run_contra_screener(
        nifty500, price_data, fund_data, q_fin_data,
        sector_lookup, company_lookup, promoter_data,
        sector_median_drawdown, contra_regime, WEEK_TAG,
    )

    # ── Export + Telegram (Contra only now — momentum args dropped) ───────
    # NOTE: export_and_notify() previously also took momentum_df, mom_entry,
    # mom_emerging, mom_watch, regime, idx_1m/3m/6m, n_rsi, n_price, n_sma30.
    # If export.py's signature hasn't been updated to drop those params,
    # you'll need to either:
    #   (a) update export_and_notify()'s signature to remove them, or
    #   (b) pass empty placeholders (empty DataFrame / None / empty lists)
    #       for each dropped momentum arg so the call still matches.
    # I don't have export.py's source, so I can't safely edit it here —
    # this call assumes (a). If you'd rather do (b), tell me and I'll
    # write the placeholder version instead.
    export_and_notify(
        contra_df, p2_df, p1_df,
        sector_lookup, contra_regime,
    )

    # ── Sector Signals (Contra Phase 2 only + rotation ranking) ───────────
    sector_results = run_sector_screener()
    if not sector_results["contra_df"].empty or not sector_results["rotation_df"].empty:
        send_telegram_message(sector_results["telegram_text"])
    else:
        print("[main] Sector screener produced no data — skipping sector Telegram send.")


if __name__ == "__main__":
    main()
