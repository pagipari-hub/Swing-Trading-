# ============================================================
# data_fetch.py — price_data + fund_data (weekly cache),
#                  q_fin_data (always fresh, no cache).
# ============================================================

import os
import time
import pickle
import yfinance as yf

from config import PRICE_FILE, FUND_FILE, CACHE_MAX_AGE_DAYS


def cache_age_days(filepath):
    return (time.time() - os.path.getmtime(filepath)) / 86400.0


def is_cache_stale(filepath, max_age_days=CACHE_MAX_AGE_DAYS):
    """
    True if the cache file is older than max_age_days, based on its
    mtime. This is independent of whatever caching layer restored the
    file (e.g. GitHub Actions actions/cache preserves original mtime
    on restore) — so a cache that keeps getting silently re-restored
    under a broken/constant key will still get caught and rejected
    here, rather than being trusted just because the file exists and
    passes structural validation.
    """
    if not os.path.exists(filepath):
        return True
    return cache_age_days(filepath) > max_age_days


def validate_price_data(data, label="price_data"):
    """
    Sanity-check cached price data before trusting it.

    Checks a small sample of known-liquid, always-listed tickers. If
    those come back empty/NaN/zero, the whole cache is treated as
    suspect and gets rebuilt from scratch instead of silently used.
    """
    sentinel_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    present = [t for t in sentinel_tickers if t in data]

    if not present:
        print(f"  ⚠️  {label} validation: none of the sentinel tickers found at all")
        return False

    good = 0
    for t in present:
        df = data.get(t)
        try:
            if df is None or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            if close.dropna().empty:
                continue
            last_price = close.dropna().iloc[-1]
            if last_price is None or last_price != last_price:  # NaN check
                continue
            if last_price <= 0:
                continue
            good += 1
        except Exception:
            continue

    ok = good >= max(1, len(present) // 2)
    if not ok:
        print(f"  ⚠️  {label} validation FAILED: only {good}/{len(present)} sentinel "
              f"tickers had usable price data. Treating cache as corrupted.")
    else:
        print(f"  ✅ {label} validation passed: {good}/{len(present)} sentinel tickers OK")
    return ok


def load_or_build(filepath, build_fn, label, validate_fn=None):
    """
    Load from cache if it exists AND is fresh enough AND (when a
    validator is given) passes structural validation. Otherwise
    rebuild from scratch and overwrite the cache file.

    Staleness is checked BEFORE structural validation and is checked
    unconditionally (no validate_fn required) — a cache can be
    perfectly well-formed and still be weeks old if the cache-key
    layer restored the same snapshot repeatedly. See CACHE_MAX_AGE_DAYS
    in config.py for why this exists as a second line of defense.
    """
    if os.path.exists(filepath):
        if is_cache_stale(filepath):
            age = cache_age_days(filepath)
            print(f"🗑️  {label} cache is {age:.1f} days old "
                  f"(> {CACHE_MAX_AGE_DAYS}-day limit) — forcing rebuild "
                  f"regardless of what restored it...")
            try:
                os.remove(filepath)
            except OSError:
                pass
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            if validate_fn is None or validate_fn(data, label):
                age = cache_age_days(filepath)
                print(f"✅ {label} loaded from cache — {len(data)} tickers "
                      f"({age:.1f} days old)")
                return data
            else:
                print(f"🗑️  {label} cache rejected as invalid — forcing rebuild...")
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    print(f"📥 Building {label}...")
    data = build_fn()
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ {label} saved — {len(data)} tickers")
    return data


def build_price_data(nifty500):
    data = {}; errors = []
    for i, ticker in enumerate(nifty500):
        try:
            df = yf.download(ticker, period="5y", interval="1wk",
                             auto_adjust=True, progress=False)
            data[ticker] = df if len(df) > 50 else None
        except Exception:
            errors.append(ticker); data[ticker] = None
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} price done...")
        time.sleep(0.1)
    print(f"  ⚠️  Price errors: {len(errors)}")
    return data


def build_fund_data(nifty500):
    data = {}; errors = []
    # NOTE: "promoterHolding" is not a real yfinance .info key and always
    # returns None. "heldPercentInsiders" is the field that actually
    # carries this data (as a 0-1 fraction) — every downstream promoter
    # check in contra_screener.py / momentum_screener.py reads THIS key,
    # so it must be fetched here or those checks silently never fire.
    fields = ["trailingPE","earningsGrowth","trailingEps","forwardEps",
              "revenueGrowth","profitMargins","heldPercentInsiders",
              "floatShares","sharesOutstanding","debtToEquity",
              "longName","sector","industry"]
    for i, ticker in enumerate(nifty500):
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {k: info.get(k) for k in fields}
        except Exception:
            errors.append(ticker); data[ticker] = {}
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} fund done...")
        time.sleep(0.1)
    print(f"  ⚠️  Fund errors: {len(errors)}")
    return data


def build_q_fin_data(nifty500):
    """Always fresh — no cache."""
    data = {}; errors = []
    for i, ticker in enumerate(nifty500):
        try:
            data[ticker] = yf.Ticker(ticker).quarterly_financials
        except Exception:
            errors.append(ticker); data[ticker] = None
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} qfin done...")
        time.sleep(0.1)
    print(f"  ⚠️  QFin errors: {len(errors)}")
    return data


def load_market_data(nifty500):
    """Returns (price_data, fund_data, q_fin_data)."""
    price_data = load_or_build(
        PRICE_FILE, lambda: build_price_data(nifty500), "price_data",
        validate_fn=validate_price_data,
    )
    fund_data = load_or_build(
        FUND_FILE, lambda: build_fund_data(nifty500), "fund_data",
    )

    print(f"\n📥 Building q_fin_data (always fresh)...")
    q_fin_data = build_q_fin_data(nifty500)

    print(f"\n✅ All data ready.")
    print(f"   price_data : {sum(1 for v in price_data.values() if v is not None)} valid")
    print(f"   fund_data  : {sum(1 for v in fund_data.values() if v)} valid")
    print(f"   q_fin_data : {sum(1 for v in q_fin_data.values() if v is not None)} valid")

    return price_data, fund_data, q_fin_data
