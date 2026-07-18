# ============================================================
# main.py — Weekly pipeline entry point.
# Run this via GitHub Actions: python main.py
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
from momentum_screener import run_momentum_screener
from export import export_and_notify


def main():
    print("✅ All libraries loaded.")
    print(f"📅 Run date   : {datetime.today().strftime('%d %b %Y')}")
    print(f"🗓️  Week tag   : {WEEK_TAG}")
    print("📊 Universe   : Nifty 500 | Contra + Momentum Screener")

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

    # ── Momentum ─────────────────────────────────────────────────────────
    (momentum_df, mom_entry, mom_emerging, mom_watch, regime,
     idx_1m, idx_3m, idx_6m, n_rsi, n_price, n_sma30) = run_momentum_screener(
        nifty500, price_data, fund_data, q_fin_data,
        sector_lookup, company_lookup, promoter_data, WEEK_TAG,
    )

    # ── Export + Telegram (skips export automatically if both screeners
    # produced zero rows, e.g. an upstream universe/data failure) ─────────
    export_and_notify(
        contra_df, momentum_df, p2_df, p1_df,
        mom_entry, mom_emerging, mom_watch,
        sector_lookup, regime, contra_regime,
        idx_1m, idx_3m, idx_6m,
        n_rsi, n_price, n_sma30,
    )


if __name__ == "__main__":
    main()
