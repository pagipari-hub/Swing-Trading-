# ============================================================
# universe.py — Load the Nifty 500 ticker/sector universe.
#
# Fallback chain (each tier only runs if the previous one failed):
#   1. Live NSE fetch (archives.nseindia.com)
#   2. /tmp cache from a prior successful run this week
#   3. Static seed CSV committed to the repo (nifty500_seed.csv)
#
# NSE is known to 503 GitHub Actions runner IPs intermittently, so
# tier 1 failing is expected and not itself a bug — the fallback
# chain exists specifically for that case.
# ============================================================

import os
import pickle
import requests
import pandas as pd
from io import StringIO

from config import NIFTY500_LIST_FILE, NIFTY500_SEED_FILE


def _fetch_live_nse():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.strip()
        df = df[["Symbol", "Company Name", "Industry"]].copy()
        df.columns = ["symbol", "company", "sector"]
        df["symbol"]    = df["symbol"].str.strip()
        df["sector"]    = df["sector"].str.strip()
        df["yf_ticker"] = df["symbol"] + ".NS"
        print(f"✅ Loaded {len(df)} Nifty 500 stocks from NSE (live)")
        return df
    except Exception as e:
        print(f"⚠️  NSE fetch failed: {e}")
        return pd.DataFrame(columns=["symbol", "company", "sector", "yf_ticker"])


def _load_tmp_cache():
    if not os.path.exists(NIFTY500_LIST_FILE):
        return pd.DataFrame(columns=["symbol", "company", "sector", "yf_ticker"])
    with open(NIFTY500_LIST_FILE, "rb") as f:
        df = pickle.load(f)
    print(f"  ↩️  Falling back to /tmp cache — {len(df)} stocks "
          f"(may be stale; live NSE fetch was unavailable)")
    return df


def _load_repo_seed():
    if not os.path.exists(NIFTY500_SEED_FILE):
        return pd.DataFrame(columns=["symbol", "company", "sector", "yf_ticker"])
    df = pd.read_csv(NIFTY500_SEED_FILE)
    df.columns = df.columns.str.strip()
    df = df[["Symbol", "Company Name", "Industry"]].copy()
    df.columns = ["symbol", "company", "sector"]
    df["symbol"]    = df["symbol"].str.strip()
    df["sector"]    = df["sector"].str.strip()
    df["yf_ticker"] = df["symbol"] + ".NS"
    print(f"  ↩️  Falling back to repo seed file — {len(df)} stocks "
          f"(static snapshot; may drift from live NSE index changes over time)")
    return df


def load_nifty500_universe():
    """
    Returns (nifty500_df, nifty500, sector_lookup, company_lookup).
    Tries live NSE, then /tmp cache, then the repo-committed seed CSV.
    Refreshes the /tmp cache whenever ANY tier succeeds, so the most
    recent good data is always what gets cached forward.
    """
    nifty500_df = _fetch_live_nse()

    if nifty500_df.empty:
        nifty500_df = _load_tmp_cache()

    if nifty500_df.empty:
        nifty500_df = _load_repo_seed()

    if nifty500_df.empty:
        print("  ❌ No fallback available (live NSE, /tmp cache, and repo seed all "
              "failed/missing) — cannot proceed.")
    else:
        with open(NIFTY500_LIST_FILE, "wb") as f:
            pickle.dump(nifty500_df, f)

    nifty500       = nifty500_df["yf_ticker"].tolist()
    sector_lookup  = dict(zip(nifty500_df["yf_ticker"], nifty500_df["sector"]))
    company_lookup = dict(zip(nifty500_df["yf_ticker"], nifty500_df["company"]))

    print(f"\n📊 Sector Distribution:")
    print(nifty500_df["sector"].value_counts().to_string())
    print(f"\n🔖 Sample: {nifty500[:5]}")

    return nifty500_df, nifty500, sector_lookup, company_lookup
