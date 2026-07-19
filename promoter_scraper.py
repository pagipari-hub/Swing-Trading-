# ============================================================
# promoter_scraper.py — Promoter holding % via Screener.in
#
# WHY THIS EXISTS: yfinance's .info dict does not reliably return
# ownership/holder data (heldPercentInsiders) for NSE-listed (.NS)
# tickers. Confirmed empirically — the 'promoter' column was blank
# for every single ticker across 6 consecutive weeks of live runs,
# even after fixing the field name in data_fetch.py. The gap is in
# yfinance's data coverage for Indian equities, not the field name.
#
# Screener.in embeds promoter holding directly in the page's
# <meta name="description"> tag, e.g.:
#   "TCS · Mkt Cap: ... · Promoter Holding: 71.8%"
# This is present in the raw HTML (no JS rendering needed), so a
# plain requests + regex scrape works.
#
# NOTE: If a pledge-data scraper for Screener.in already exists in
# this project, this should be MERGED into that scraper rather than
# run as a second, separate pass over the same 500 pages — hitting
# Screener.in twice per ticker per week doubles request volume and
# doubles the chance of getting rate-limited/blocked. This module is
# built standalone because that existing scraper wasn't available to
# integrate with directly; treat this as a merge candidate.
# ============================================================

import os
import re
import time
import pickle
import requests

CACHE_DIR       = "/tmp/nifty500_cache"
PROMOTER_FILE   = os.path.join(CACHE_DIR, "promoter_data.pkl")
# Promoter holding is disclosed quarterly (SEBI requirement), not
# weekly — no point re-scraping 500 pages every single run. Cache
# for 25 days; a run older than that rebuilds.
CACHE_MAX_AGE_SECONDS = 25 * 24 * 60 * 60

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
}

PROMOTER_RE = re.compile(r"Promoter Holding:\s*([\d.]+)\s*%")


def _screener_symbol(yf_ticker):
    """'TCS.NS' -> 'TCS'. Screener.in URLs use the bare NSE symbol."""
    return yf_ticker.replace(".NS", "")


def _fetch_one(yf_ticker):
    """
    Returns promoter holding as a float percentage (e.g. 71.8), or
    None if the page didn't load or the field wasn't found.
    Tries /consolidated/ first (most large/mid-caps), falls back to
    the standalone page if that 404s (smaller companies often only
    have standalone financials).
    """
    symbol = _screener_symbol(yf_ticker)
    urls = [
        f"https://www.screener.in/company/{symbol}/consolidated/",
        f"https://www.screener.in/company/{symbol}/",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            m = PROMOTER_RE.search(r.text)
            if m:
                return float(m.group(1))
        except Exception:
            continue
    return None


def _load_or_build(nifty500):
    if os.path.exists(PROMOTER_FILE):
        age = time.time() - os.path.getmtime(PROMOTER_FILE)
        if age < CACHE_MAX_AGE_SECONDS:
            with open(PROMOTER_FILE, "rb") as f:
                data = pickle.load(f)
            print(f"✅ promoter_data loaded from cache — {len(data)} tickers "
                  f"({age/86400:.1f} days old)")
            return data
        else:
            print(f"🗑️  promoter_data cache is {age/86400:.1f} days old "
                  f"(>25 day limit) — rebuilding...")

    print("📥 Building promoter_data from Screener.in (this will take a while — "
          "500 pages, rate-limited)...")
    data = {}
    errors = []
    for i, ticker in enumerate(nifty500):
        pct = _fetch_one(ticker)
        data[ticker] = pct
        if pct is None:
            errors.append(ticker)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(nifty500)} promoter done...")
        time.sleep(0.5)   # be polite — avoid getting rate-limited/blocked

    print(f"  ⚠️  Promoter scrape errors/missing: {len(errors)}/{len(nifty500)}")
    with open(PROMOTER_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ promoter_data saved — {len(data)} tickers")
    return data


def load_promoter_data(nifty500):
    """
    Returns {yf_ticker: promoter_pct_or_None}.
    promoter_pct is a plain float percentage (e.g. 71.8), matching
    the scale contra_screener.py / momentum_screener.py already
    expect (they previously did `* 100` on yfinance's 0-1 fraction —
    that multiplication is NOT needed for this data, since Screener.in
    already reports it as a percentage. See wiring note below.)
    """
    return _load_or_build(nifty500)
