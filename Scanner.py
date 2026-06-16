# ============================================================
# Contra + Momentum Screener | NSE Nifty 500 | yfinance
# GitHub Actions version — append rows to "Contra Data" and
# "Momentum Data" sheets; all 500 stocks every week.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import gspread
import pickle
import os
import json
import warnings
import time
import re
from io import StringIO
from datetime import datetime
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.2f}".format)

# ── Week tag ─────────────────────────────────────────────────
_iso     = datetime.now().isocalendar()
WEEK_TAG = f"{_iso.year}-W{_iso.week:02d}"   # e.g. 2026-W24
TIMESTAMP = datetime.now().strftime("%d %b %Y %H:%M")

print("✅ All libraries loaded.")
print(f"📅 Run date   : {datetime.today().strftime('%d %b %Y')}")
print(f"🗓️  Week tag   : {WEEK_TAG}")
print(f"📊 Universe   : Nifty 500 | Contra + Momentum Screener")

# ── Telegram config ──────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        print(f"⚠️  Telegram error: {e}")

# ── Google Sheets auth via Service Account ───────────────────
def get_gspread_client():
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not sa_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON secret not set")
    sa_info = json.loads(sa_json)
    scopes  = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)

# ============================================================
# SECTION 1 — Load Nifty 500 Tickers from NSE
# ============================================================

def load_nifty500_tickers():
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
        print(f"✅ Loaded {len(df)} Nifty 500 stocks from NSE")
        return df
    except Exception as e:
        print(f"⚠️  NSE fetch failed: {e}")
        return pd.DataFrame(columns=["symbol","company","sector","yf_ticker"])

nifty500_df    = load_nifty500_tickers()
nifty500       = nifty500_df["yf_ticker"].tolist()
sector_lookup  = dict(zip(nifty500_df["yf_ticker"], nifty500_df["sector"]))
company_lookup = dict(zip(nifty500_df["yf_ticker"], nifty500_df["company"]))

print(f"\n📊 Sector Distribution:")
print(nifty500_df["sector"].value_counts().to_string())
print(f"\n🔖 Sample: {nifty500[:5]}")

# ============================================================
# SECTION 2 — Cache: price_data + fund_data (weekly)
#              q_fin_data: always fresh (no cache)
# ============================================================

CACHE_DIR     = "/tmp/nifty500_cache"
PRICE_FILE    = os.path.join(CACHE_DIR, "price_data.pkl")
FUND_FILE     = os.path.join(CACHE_DIR, "fund_data.pkl")
# NOTE: q_fin_data is intentionally NOT cached — fresh every run

os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_build(filepath, build_fn, label):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"✅ {label} loaded from cache — {len(data)} tickers")
        return data
    print(f"📥 Building {label}...")
    data = build_fn()
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ {label} saved — {len(data)} tickers")
    return data

def build_price_data():
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

def build_fund_data():
    data = {}; errors = []
    fields = ["trailingPE","earningsGrowth","trailingEps","forwardEps",
              "revenueGrowth","profitMargins","promoterHolding",
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

def build_q_fin_data():
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

price_data = load_or_build(PRICE_FILE, build_price_data, "price_data")
fund_data  = load_or_build(FUND_FILE,  build_fund_data,  "fund_data")

print(f"\n📥 Building q_fin_data (always fresh)...")
q_fin_data = build_q_fin_data()

print(f"\n✅ All data ready.")
print(f"   price_data : {sum(1 for v in price_data.values() if v is not None)} valid")
print(f"   fund_data  : {sum(1 for v in fund_data.values() if v)} valid")
print(f"   q_fin_data : {sum(1 for v in q_fin_data.values() if v is not None)} valid")

# ============================================================
# SECTION 3 — Helpers
# ============================================================

def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except Exception:
        return default

def pct_ret(series, periods):
    try:
        s = series.dropna()
        if len(s) < periods + 1:
            return None
        return (s.iloc[-1] / s.iloc[-(periods+1)] - 1) * 100
    except Exception:
        return None

def yoy_growth(series):
    try:
        vals = series.dropna()
        if len(vals) < 5:
            return None
        latest = vals.iloc[0]; yr_ago = vals.iloc[4]
        if yr_ago == 0 or np.isnan(yr_ago):
            return None
        return ((latest - yr_ago) / abs(yr_ago)) * 100
    except Exception:
        return None

def fmt(val, decimals=2, fallback="—"):
    """Format a numeric value; return fallback string if None/nan/999."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or val == 999)):
            return fallback
        return round(float(val), decimals)
    except Exception:
        return fallback

# ============================================================
# SECTION 4 — CONTRA SCREENER (all 500 stocks)
# ============================================================

print("\n" + "=" * 65)
print("CONTRA SCREENER")
print("=" * 65)

contra_rows = []   # will hold ALL stocks (500) for sheet export

for ticker in nifty500:
    df   = price_data.get(ticker)
    info = fund_data.get(ticker, {})
    if df is None or len(df) < 60:
        continue

    try:
        close   = df["Close"].squeeze()
        volume  = df["Volume"].squeeze()
        price   = close.iloc[-1]
        company = company_lookup.get(ticker, ticker)
        sector  = sector_lookup.get(ticker, "Unknown")

        # ── RSI(14) and its 14-week SMA ──────────────────────
        rsi_s  = ta.momentum.RSIIndicator(close, window=14).rsi()
        if rsi_s is None or rsi_s.dropna().shape[0] < 5:
            continue
        rsi_sma = rsi_s.rolling(14).mean()
        rsi     = rsi_s.iloc[-1]
        rsi_sm  = rsi_sma.iloc[-1]

        # ── Volume ratio ──────────────────────────────────────
        vol_avg   = volume.iloc[-14:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0

        # ── 52-week drawdown ─────────────────────────────────
        high_52w = df["High"].squeeze().iloc[-52:].max()
        drawdown = (high_52w - price) / high_52w * 100

        # ── Fundamentals ──────────────────────────────────────
        promoter = safe_get(info, "promoterHolding", 0)
        de_ratio = safe_get(info, "debtToEquity",    999)

        # ── Sales & PAT growth from quarterly financials ──────
        sales_gr = pat_gr = None
        qfin = q_fin_data.get(ticker)
        if qfin is not None and not qfin.empty:
            for k in ["Total Revenue", "Revenue"]:
                if k in qfin.index:
                    sales_gr = yoy_growth(qfin.loc[k]); break
            for k in ["Net Income", "Net Income Common Stockholders"]:
                if k in qfin.index:
                    pat_gr = yoy_growth(qfin.loc[k]); break

        # ── Pillar A: Value / Fundamentals (max 40) ───────────
        a1 = (20 if drawdown >= 40 else
              15 if drawdown >= 30 else
              10 if drawdown >= 20 else 0)
        a2 = (10 if (promoter and promoter >= 50) else
               5 if (promoter and promoter >= 35) else 0)
        a3 = (10 if (de_ratio and de_ratio < 0.5) else
               5 if (de_ratio and de_ratio < 1.5) else 0)
        pillar_a = a1 + a2 + a3

        # ── RSI cross (within last 3 weeks) ──────────────────
        cross_week = None
        for w in range(3):
            r_now   = rsi_s.iloc[-(w+1)]
            r_prev  = rsi_s.iloc[-(w+2)]
            rs_now  = rsi_sma.iloc[-(w+1)]
            rs_prev = rsi_sma.iloc[-(w+2)]
            if r_now > rs_now and r_prev <= rs_prev:
                cross_week = w
                break

        # ── Pillar B: Momentum / Signal (max 60) ─────────────
        if   cross_week == 0: b1 = 30
        elif cross_week == 1: b1 = 22
        elif cross_week == 2: b1 = 15
        elif rsi > rsi_sm:    b1 = 8
        else:                 b1 = 0

        b2 = (20 if rsi < 30 else
              15 if rsi < 40 else
              10 if rsi < 50 else 0)
        b3 = (10 if vol_ratio >= 3   else
               7 if vol_ratio >= 2   else
               4 if vol_ratio >= 1.5 else 0)
        pillar_b = b1 + b2 + b3
        total    = pillar_a + pillar_b

        # ── Phase label ───────────────────────────────────────
        if cross_week is not None and total >= 50:
            phase = "Phase 2 - Entry Ready"
        elif rsi < rsi_sm and drawdown >= 20 and vol_ratio >= 1.5:
            phase = "Phase 1 - Watchlist"
        elif total >= 30:
            phase = "On Radar"
        else:
            phase = "Below Threshold"

        # ── Append row (exact column spec) ───────────────────
        contra_rows.append({
            "week":       WEEK_TAG,
            "ticker":     ticker,
            "company":    company,
            "sector":     sector,
            "price":      fmt(price, 2),
            "score":      total,
            "phase":      phase,
            "RSI":        fmt(rsi, 1),
            "drawdown":   fmt(drawdown, 1),
            "de_ratio":   fmt(de_ratio, 2) if de_ratio != 999 else "—",
            "cross_week": cross_week if cross_week is not None else "—",
            "vol_ratio":  fmt(vol_ratio, 2),
            "promoter":   fmt(promoter, 1) if promoter else "—",
            "sales_gr":   fmt(sales_gr, 1) if sales_gr is not None else "—",
            "pat_gr":     fmt(pat_gr, 1)   if pat_gr   is not None else "—",
            "pillar_a":   pillar_a,
            "pillar_b":   pillar_b,
        })

    except Exception:
        continue

contra_df = pd.DataFrame(contra_rows) if contra_rows else pd.DataFrame(columns=[
    "week","ticker","company","sector","price","score","phase",
    "RSI","drawdown","de_ratio","cross_week","vol_ratio","promoter",
    "sales_gr","pat_gr","pillar_a","pillar_b"
])

# Derived views (for Telegram summary only — not written to sheets)
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
{'='*65}
""")

# ============================================================
# SECTION 5 — MOMENTUM SCREENER (all 500 stocks)
# ============================================================

print("=" * 65)
print("MOMENTUM SCREENER — Full Build")
print("=" * 65)

WEEKS_VALID   = 3
RSI_UPPER     = 75
DIST_HIGH_MAX = 20
LIQ_HARD      = 10
LIQ_THIN      = 20
VOL_WINDOW    = 13
RAW_MAX       = 120
WATCH_MIN_A   = 20

CYCLICAL_NSE = {
    "Metals & Mining", "Oil Gas & Consumable Fuels",
    "Construction", "Construction Materials",
    "Automobile and Auto Components", "Capital Goods",
    "Power", "Chemicals"
}

NIFTY500_INDEX = "^CRSLDX"

# ── Market Regime ────────────────────────────────────────────
print("\n[1/5] Market Regime...")
regime = "Risk-On"; regime_mult = 1.0
nifty_close = None

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

# ── Index reference returns ──────────────────────────────────
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

# ── Sector metrics ───────────────────────────────────────────
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

# ── Score all 500 stocks ─────────────────────────────────────
print("\n[4/5] Scoring stocks...")

momentum_rows = []   # ALL stocks go here

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

        # ── Liquidity filter ─────────────────────────────────
        avg_vol20 = volume.iloc[-21:-1].mean()
        traded_cr = (price * avg_vol20) / 1e7
        if traded_cr < LIQ_HARD:
            # Still record the stock but with zero score / "Filtered" label
            # so "all 500 stocks" requirement is met
            momentum_rows.append({
                "week":          WEEK_TAG,
                "ticker":        ticker,
                "company":       company_lookup.get(ticker, ticker),
                "sector":        sector_lookup.get(ticker, "Unknown"),
                "price":         fmt(price, 2),
                "score":         0,
                "label":         "Filtered (Low Liq)",
                "RSI":           "—",
                "rs_periods":    "—",
                "earn_gr":       "—",
                "pct_from_high": "—",
                "vol_ratio":     "—",
                "atr_exp":       "—",
                "pillar_a":      0,
                "pillar_b":      0,
                "pillar_c":      0,
                "pillar_d":      0,
            })
            continue

        # ── Earnings / fundamentals ──────────────────────────
        t_eps   = safe_get(info, "trailingEps",    None)
        f_eps   = safe_get(info, "forwardEps",     None)
        earn_gr = safe_get(info, "earningsGrowth", 0)
        pe      = safe_get(info, "trailingPE",     999)
        if earn_gr and abs(earn_gr) < 5:
            earn_gr = earn_gr * 100

        if (t_eps is not None and t_eps < 0) and \
           (f_eps is not None and f_eps < 0):
            momentum_rows.append({
                "week":          WEEK_TAG,
                "ticker":        ticker,
                "company":       company,
                "sector":        sector,
                "price":         fmt(price, 2),
                "score":         0,
                "label":         "Filtered (Neg EPS)",
                "RSI":           "—",
                "rs_periods":    "—",
                "earn_gr":       "—",
                "pct_from_high": "—",
                "vol_ratio":     "—",
                "atr_exp":       "—",
                "pillar_a":      0,
                "pillar_b":      0,
                "pillar_c":      0,
                "pillar_d":      0,
            })
            continue

        # ── Quarterly financials ─────────────────────────────
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

        # ── Price metrics ────────────────────────────────────
        high_52w      = high_s.iloc[-52:].max()
        high_10w      = high_s.iloc[-10:].max()
        pct_from_high = (high_52w - price) / high_52w * 100

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        # ── RSI ──────────────────────────────────────────────
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

        # ── Volume ───────────────────────────────────────────
        vol_avg   = volume.iloc[-VOL_WINDOW-1:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0

        vol_persist = 0
        for w in range(1, 5):
            if volume.iloc[-(w+1)] > vol_avg:
                vol_persist += 1
            else:
                break

        # ── ATR expansion ────────────────────────────────────
        atr_s = ta.volatility.AverageTrueRange(high_s, low_s, close, window=14).average_true_range()
        try:
            atr_exp = (atr_s.iloc[-1] / atr_s.iloc[-9] - 1) * 100
        except Exception:
            atr_exp = 0

        # ── Relative strength vs index ───────────────────────
        r1m = pct_ret(close, 4);  rs1 = (r1m or 0) - idx_1m
        r3m = pct_ret(close, 13); rs3 = (r3m or 0) - idx_3m
        r6m = pct_ret(close, 26); rs6 = (r6m or 0) - idx_6m
        rs_periods = sum(1 for rs in [rs1, rs3, rs6] if rs > 0)

        # ── Sector metrics ───────────────────────────────────
        s_stats  = sector_stats.get(sector, {"4w": 0, "6m": 0, "breadth": 0})
        sec_4w   = s_stats["4w"]
        sec_6m   = s_stats["6m"]
        sec_brd  = s_stats["breadth"]
        sec_vs_i = sector_vs_idx.get(sector, 0)
        outperf  = (r6m or 0) - sec_6m

        # ── is_watch flag (for label only) ───────────────────
        is_watch = (
            cross_week is None and
            50 <= rsi_now < 60 and
            price > sma20 and price > sma50 and
            pct_from_high <= 15
        )

        # ── PILLAR A: Price position + RS (max 40) ───────────
        a1 = 15 if pct_from_high<=5 else (10 if pct_from_high<=10 else 6)
        a2 = 10 if (price>sma20 and price>sma50) else (5 if price>sma20 else 0)
        a3 = 5  if outperf>10 else (3 if outperf>=5 else (1 if outperf>=0 else 0))
        a4 = 0
        if is_cycl:
            a4 = 5 if (pe and pe < 40) else 0
        else:
            a4 += 3 if (earn_gr and earn_gr > 15) else 0
            a4 += 2 if (pe and pe < 40) else 0
        a5 = 0
        if rs_periods >= 2:
            a5 += 3 if rs1 > 0 else 0
            a5 += 4 if rs3 > 0 else 0
            a5 += 3 if rs6 > 0 else 0
        pillar_a = a1+a2+a3+a4+a5

        # ── PILLAR B: RSI signal + breakout (max 65) ─────────
        if cross_week is None:   b1 = 0
        elif cross_week == 0:    b1 = 30
        elif cross_week == 1:    b1 = 22
        elif cross_week == 2:    b1 = 15
        else:                    b1 = 8
        if b1 > 0 and rsi_slope_ok:
            b1 = min(b1+5, 35)

        b2 = 0
        if rsi_at_cross is not None:
            b2 = 10 if rsi_at_cross<=65 else (6 if rsi_at_cross<=70 else 3)

        b3 = 10 if vol_ratio>=3 else (8 if vol_ratio>=2.5 else
             (5 if vol_ratio>=2 else (2 if vol_ratio>=1.5 else 0)))

        b4 = 0
        b4 += 4 if price > high_10w else 0
        b4 += 3 if vol_ratio >= 1.5 else 0
        try:
            rng = high_s.iloc[-1] - low_s.iloc[-1]
            b4 += 3 if rng > 0 and (price - low_s.iloc[-1])/rng >= 0.75 else 0
        except Exception:
            pass

        b5 = 10 if vol_persist>=3 else (5 if vol_persist==2 else
             (2 if vol_persist==1 else 0))
        b6 = 5 if atr_exp>=20 else (3 if atr_exp>=10 else 0)
        pillar_b = b1+b2+b3+b4+b5+b6

        # ── PILLAR C: Sector strength (max 15) ───────────────
        c1 = 5 if sec_4w>=5 else (3 if sec_4w>=2 else (1 if sec_4w>=0 else 0))
        c2 = 5 if sec_vs_i>=3 else (3 if sec_vs_i>=1 else (1 if sec_vs_i>=0 else 0))
        c3 = 5 if sec_brd>=70 else (3 if sec_brd>=60 else (1 if sec_brd>=50 else 0))
        pillar_c = c1+c2+c3

        # ── PILLAR D: Earnings / fundamentals (max 15) ───────
        d1 = 5 if (sales_gr and sales_gr>=25) else (3 if (sales_gr and sales_gr>=15)
             else (1 if (sales_gr and sales_gr>=5) else 0))
        d2 = 5 if (pat_gr and pat_gr>=30) else (3 if (pat_gr and pat_gr>=15)
             else (1 if (pat_gr and pat_gr>=5) else 0))
        d3 = 5 if (fwd_eps_gr and fwd_eps_gr>=20) else (3 if (fwd_eps_gr and fwd_eps_gr>=10)
             else (1 if (fwd_eps_gr and fwd_eps_gr>=0) else 0))
        pillar_d = d1+d2+d3

        raw   = pillar_a + pillar_b + pillar_c + pillar_d
        total = round(round((raw / RAW_MAX) * 100) * regime_mult)

        # ── Label ────────────────────────────────────────────
        if cross_week is not None:
            label = ("🔥 High Momentum"      if total >= 76 else
                     "⚡ Momentum Candidate" if total >= 61 else
                     "📈 Building Up"        if total >= 41 else
                     "👀 Weak Signal")
        elif is_watch:
            label = "👁️ Watch"
        else:
            label = "Below Threshold"

        # ── earn_gr for sheet (use YoY qfin if available) ────
        earn_gr_out = pat_gr if pat_gr is not None else earn_gr

        # ── Append row (exact column spec) ───────────────────
        momentum_rows.append({
            "week":          WEEK_TAG,
            "ticker":        ticker,
            "company":       company,
            "sector":        sector,
            "price":         fmt(price, 2),
            "score":         total,
            "label":         label,
            "RSI":           fmt(rsi_now, 1),
            "rs_periods":    rs_periods,
            "earn_gr":       fmt(earn_gr_out, 1) if earn_gr_out is not None else "—",
            "pct_from_high": fmt(pct_from_high, 1),
            "vol_ratio":     fmt(vol_ratio, 2),
            "atr_exp":       fmt(atr_exp, 1),
            "pillar_a":      pillar_a,
            "pillar_b":      pillar_b,
            "pillar_c":      pillar_c,
            "pillar_d":      pillar_d,
        })

    except Exception:
        continue

momentum_df = pd.DataFrame(momentum_rows) if momentum_rows else pd.DataFrame(columns=[
    "week","ticker","company","sector","price","score","label",
    "RSI","rs_periods","earn_gr","pct_from_high","vol_ratio","atr_exp",
    "pillar_a","pillar_b","pillar_c","pillar_d"
])

# Derived views for Telegram summary
mom_entry = momentum_df[momentum_df["label"].str.startswith(("🔥","⚡","📈"), na=False)].copy()
mom_watch = momentum_df[momentum_df["label"] == "👁️ Watch"].copy()

print(f"""
{'='*65}
MOMENTUM SCREENER | Regime: {regime}
{'='*65}
  Total rows built : {len(momentum_df)}
  Entry candidates : {len(mom_entry)}
  Watch list       : {len(mom_watch)}
{'='*65}
""")

# ============================================================
# SECTION 6 — Export to Google Sheets (append rows)
# ============================================================

print("\n[5/5] Exporting to Google Sheets...")

SHEET_NAME = "Nifty500 Screener"

# Exact column order per spec
CONTRA_COLS  = ["week","ticker","company","sector","price",
                "score","phase","RSI","drawdown","de_ratio",
                "cross_week","vol_ratio","promoter","sales_gr",
                "pat_gr","pillar_a","pillar_b"]

MOMENTUM_COLS = ["week","ticker","company","sector","price",
                 "score","label","RSI","rs_periods","earn_gr",
                 "pct_from_high","vol_ratio","atr_exp",
                 "pillar_a","pillar_b","pillar_c","pillar_d"]

def ensure_worksheet(sh, name, rows=2000, cols=25):
    """Get existing worksheet or create it with a header row."""
    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=rows, cols=cols)
        return ws

def append_to_sheet(ws, df, cols):
    """
    Append df rows to ws.
    - If sheet is empty (no data rows), write header first then data.
    - If sheet already has data, append only data rows.
    """
    available = [c for c in cols if c in df.columns]
    df_out    = df[available].copy().fillna("").astype(str)
    data_rows = df_out.values.tolist()

    # Check whether the sheet already has a header
    existing  = ws.get_all_values()
    has_header = len(existing) > 0 and len(existing[0]) > 0

    if not has_header:
        # Write header + data starting at A1
        ws.update(values=[available] + data_rows, range_name="A1")
        print(f"   ✅ '{ws.title}' — header + {len(data_rows)} rows written")
    else:
        # Append data below last row
        next_row = len(existing) + 1
        if data_rows:
            ws.update(values=data_rows,
                      range_name=f"A{next_row}")
        print(f"   ✅ '{ws.title}' — {len(data_rows)} rows appended (sheet now {next_row + len(data_rows) - 1} rows)")

try:
    gc = get_gspread_client()

    try:
        sh = gc.open(SHEET_NAME)
        print(f"✅ Opened existing: {SHEET_NAME}")
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
        print(f"📄 Created new: {SHEET_NAME}")

    # ── Contra Data ───────────────────────────────────────────
    ws_contra = ensure_worksheet(sh, "Contra Data", rows=60000, cols=20)
    append_to_sheet(ws_contra, contra_df, CONTRA_COLS)

    # ── Momentum Data ─────────────────────────────────────────
    ws_mom = ensure_worksheet(sh, "Momentum Data", rows=60000, cols=20)
    append_to_sheet(ws_mom, momentum_df, MOMENTUM_COLS)

    # Sheets "Contra Entry", "Mom Entry", "Dashboard" are managed
    # separately by user — Scanner does NOT touch them.

    print(f"""
{'='*55}
✅ EXPORT COMPLETE — {TIMESTAMP}
{'='*55}
Sheet    : {SHEET_NAME}
Week tag : {WEEK_TAG}

Contra Data  : {len(contra_df)} rows appended
  Phase 2    : {len(p2_df)}
  Phase 1    : {len(p1_df)}

Momentum Data: {len(momentum_df)} rows appended
  Entry      : {len(mom_entry)}
  Watch      : {len(mom_watch)}
  Regime     : {regime}
{'='*55}
""")
    print(f"🔗 Link: {sh.url}")

    # ── Telegram Summary ──────────────────────────────────────
    p2_top  = (p2_df.sort_values("score", ascending=False)
               .head(5)[["ticker","score"]].to_string(index=False)
               if not p2_df.empty else "None")
    mom_top = (mom_entry.sort_values("score", ascending=False)
               .head(5)[["ticker","score","label"]].to_string(index=False)
               if not mom_entry.empty else "None")

    tg_msg = f"""📊 <b>Nifty 500 Screener — {WEEK_TAG}</b>

<b>CONTRA:</b>
  Phase 2 Entry Ready : {len(p2_df)}
  Phase 1 Watchlist   : {len(p1_df)}
  Total screened      : {len(contra_df)}

Top P2:
<pre>{p2_top}</pre>

<b>MOMENTUM ({regime}):</b>
  Entry  : {len(mom_entry)}
  Watch  : {len(mom_watch)}
  Total  : {len(momentum_df)}

Top Momentum:
<pre>{mom_top}</pre>

🔗 Sheet updated ✅"""

    send_telegram(tg_msg)

except Exception as e:
    err_msg = f"❌ Export failed: {e}"
    print(err_msg)
    import traceback; traceback.print_exc()
    send_telegram(f"❌ Screener FAILED on {TIMESTAMP}\nError: {e}")
