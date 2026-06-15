# ============================================================
# Contra + Momentum Screener | NSE Nifty 500 | yfinance
# GitHub Actions version — no Colab dependency
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

print("✅ All libraries loaded.")
print(f"📅 Run date: {datetime.today().strftime('%d %b %Y')}")
print(f"📊 Universe: Nifty 500 | Contra + Momentum Screener")

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
# CELL 2 — Load Nifty 500 Tickers from NSE
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

nifty500_df   = load_nifty500_tickers()
nifty500      = nifty500_df["yf_ticker"].tolist()
sector_lookup  = dict(zip(nifty500_df["yf_ticker"], nifty500_df["sector"]))
company_lookup = dict(zip(nifty500_df["yf_ticker"], nifty500_df["company"]))

print(f"\n📊 Sector Distribution:")
print(nifty500_df["sector"].value_counts().to_string())
print(f"\n🔖 Sample: {nifty500[:5]}")

# ============================================================
# CELL 3 — Cache: price_data + fund_data + q_fin_data
# Uses local /tmp folder (GitHub Actions runner)
# ============================================================

CACHE_DIR     = "/tmp/nifty500_cache"
PRICE_FILE    = os.path.join(CACHE_DIR, "price_data.pkl")
FUND_FILE     = os.path.join(CACHE_DIR, "fund_data.pkl")
QFIN_FILE     = os.path.join(CACHE_DIR, "q_fin_data.pkl")
NIFTY500_FILE = os.path.join(CACHE_DIR, "nifty500_df.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_build(filepath, build_fn, label):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"✅ {label} loaded from cache — {len(data)} tickers")
        return data
    else:
        print(f"📥 Building {label}...")
        data = build_fn()
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ {label} saved — {len(data)} tickers")
        return data

def build_price_data():
    data   = {}
    errors = []
    for i, ticker in enumerate(nifty500):
        try:
            df = yf.download(ticker, period="5y", interval="1wk",
                             auto_adjust=True, progress=False)
            data[ticker] = df if len(df) > 50 else None
        except Exception:
            errors.append(ticker)
            data[ticker] = None
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} price done...")
        time.sleep(0.1)
    print(f"  ⚠️  Price errors: {len(errors)}")
    return data

def build_fund_data():
    data   = {}
    errors = []
    fields = ["trailingPE","earningsGrowth","trailingEps","forwardEps",
              "revenueGrowth","profitMargins","promoterHolding",
              "floatShares","sharesOutstanding","debtToEquity",
              "longName","sector","industry"]
    for i, ticker in enumerate(nifty500):
        try:
            info = yf.Ticker(ticker).info
            data[ticker] = {k: info.get(k) for k in fields}
        except Exception:
            errors.append(ticker)
            data[ticker] = {}
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} fund done...")
        time.sleep(0.1)
    print(f"  ⚠️  Fund errors: {len(errors)}")
    return data

def build_q_fin_data():
    data   = {}
    errors = []
    for i, ticker in enumerate(nifty500):
        try:
            data[ticker] = yf.Ticker(ticker).quarterly_financials
        except Exception:
            errors.append(ticker)
            data[ticker] = None
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nifty500)} qfin done...")
        time.sleep(0.1)
    print(f"  ⚠️  QFin errors: {len(errors)}")
    return data

price_data = load_or_build(PRICE_FILE,   build_price_data,  "price_data")
fund_data  = load_or_build(FUND_FILE,    build_fund_data,   "fund_data")
q_fin_data = load_or_build(QFIN_FILE,    build_q_fin_data,  "q_fin_data")

with open(NIFTY500_FILE, "wb") as f:
    pickle.dump(nifty500_df, f)

print(f"\n✅ All cache ready.")
print(f"   price_data : {sum(1 for v in price_data.values() if v is not None)} valid tickers")
print(f"   fund_data  : {sum(1 for v in fund_data.values() if v) } valid tickers")
print(f"   q_fin_data : {sum(1 for v in q_fin_data.values() if v is not None)} valid tickers")

# ============================================================
# CELL 4 — Contra Screener
# ============================================================

print("=" * 65)
print("CONTRA SCREENER")
print("=" * 65)

def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except Exception:
        return default

rows_all = []

for ticker in nifty500:
    df   = price_data.get(ticker)
    info = fund_data.get(ticker, {})
    if df is None or len(df) < 60:
        continue

    try:
        close    = df["Close"].squeeze()
        volume   = df["Volume"].squeeze()
        price    = close.iloc[-1]
        company  = company_lookup.get(ticker, ticker)
        sector   = sector_lookup.get(ticker, "Unknown")

        rsi_s   = ta.momentum.RSIIndicator(close, window=14).rsi()
        if rsi_s is None or rsi_s.dropna().shape[0] < 5:
            continue
        rsi_sma  = rsi_s.rolling(14).mean()
        rsi      = rsi_s.iloc[-1]
        rsi_sm   = rsi_sma.iloc[-1]

        vol_avg   = volume.iloc[-14:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0

        high_52w  = df["High"].squeeze().iloc[-52:].max()
        drawdown  = (high_52w - price) / high_52w * 100

        promoter  = safe_get(info, "promoterHolding", 0)
        de_ratio  = safe_get(info, "debtToEquity",    999)
        earn_gr   = safe_get(info, "earningsGrowth",  0)
        if earn_gr and abs(earn_gr) < 5:
            earn_gr = earn_gr * 100
        pe        = safe_get(info, "trailingPE", 999)

        a1 = (20 if drawdown >= 40 else
              15 if drawdown >= 30 else
              10 if drawdown >= 20 else 0)
        a2 = (10 if (promoter and promoter >= 50) else
               5 if (promoter and promoter >= 35) else 0)
        a3 = (10 if (de_ratio and de_ratio < 0.5) else
               5 if (de_ratio and de_ratio < 1.5) else 0)
        pillar_a = a1 + a2 + a3

        cross_week = None
        for w in range(3):
            r_now   = rsi_s.iloc[-(w+1)]
            r_prev  = rsi_s.iloc[-(w+2)]
            rs_now  = rsi_sma.iloc[-(w+1)]
            rs_prev = rsi_sma.iloc[-(w+2)]
            if r_now > rs_now and r_prev <= rs_prev:
                cross_week = w
                break

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

        rsi_cross_up = cross_week is not None
        if rsi_cross_up and total >= 50:
            phase = "Phase 2 - Entry Ready"
        elif rsi < rsi_sm and drawdown >= 20 and vol_ratio >= 1.5:
            phase = "Phase 1 - Watchlist"
        elif total >= 30:
            phase = "On Radar"
        else:
            phase = "Below Threshold"

        rows_all.append({
            "ticker":      ticker,
            "company":     company,
            "sector":      sector,
            "price":       round(price, 2),
            "drawdown":    round(drawdown, 1),
            "rsi":         round(rsi, 1),
            "rsi_sma":     round(rsi_sm, 1),
            "cross_week":  cross_week if cross_week is not None else "—",
            "vol_ratio":   round(vol_ratio, 2),
            "promoter":    round(promoter, 1) if promoter else "—",
            "de_ratio":    round(de_ratio, 2) if de_ratio != 999 else "—",
            "earn_gr":     round(earn_gr, 1) if earn_gr else "—",
            "pe":          round(pe, 1) if pe != 999 else "—",
            "pillar_a":    pillar_a,
            "pillar_b":    pillar_b,
            "total_score": total,
            "phase":       phase,
        })

    except Exception:
        continue

CONTRA_EMPTY_COLS = ["ticker","company","sector","price","drawdown","rsi","rsi_sma","cross_week","vol_ratio","promoter","de_ratio","earn_gr","pe","pillar_a","pillar_b","total_score","phase"]

if rows_all:
    master_df = (pd.DataFrame(rows_all)
                 .sort_values("total_score", ascending=False)
                 .reset_index(drop=True))
else:
    print("⚠️  No contra stocks found")
    master_df = pd.DataFrame(columns=CONTRA_EMPTY_COLS)

p2_final = (master_df[master_df["phase"] == "Phase 2 - Entry Ready"]
            .reset_index(drop=True)) if not master_df.empty else pd.DataFrame(columns=CONTRA_EMPTY_COLS)

p1_all = master_df[master_df["phase"] == "Phase 1 - Watchlist"].copy()
sec_count_p1 = {}; p1_kept = []
for _, r in p1_all.sort_values("total_score", ascending=False).iterrows():
    sec = r["sector"]
    if sec_count_p1.get(sec, 0) < 3:
        p1_kept.append(r)
        sec_count_p1[sec] = sec_count_p1.get(sec, 0) + 1
p1_final = pd.DataFrame(p1_kept).reset_index(drop=True) if p1_kept else pd.DataFrame(columns=master_df.columns)

print(f"""
{'='*65}
CONTRA SCREENER RESULTS
{'='*65}
  Phase 2 Entry Ready : {len(p2_final)}
  Phase 1 Watchlist   : {len(p1_final)}
  On Radar            : {len(master_df[master_df['phase']=='On Radar'])}
  Total screened      : {len(master_df)}
{'='*65}
""")

# ============================================================
# CELL 5 — Momentum Screener
# ============================================================

print("=" * 65)
print("MOMENTUM SCREENER — Full Build")
print("=" * 65)

WEEKS_VALID    = 3
RSI_UPPER      = 75
DIST_HIGH_MAX  = 20
LIQ_HARD       = 10
LIQ_THIN       = 20
VOL_WINDOW     = 13
RAW_MAX        = 120
WATCH_MIN_A    = 20

CYCLICAL_NSE = {
    "Metals & Mining", "Oil Gas & Consumable Fuels",
    "Construction", "Construction Materials",
    "Automobile and Auto Components", "Capital Goods",
    "Power", "Chemicals"
}

NIFTY500_INDEX = "^CRSLDX"

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

print("\n[1/6] Market Regime...")
regime = "Risk-On"; regime_mult = 1.0

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

print("\n[2/6] Index reference returns...")
try:
    idx_1m = pct_ret(nifty_close, 4)
    idx_3m = pct_ret(nifty_close, 13)
    idx_6m = pct_ret(nifty_close, 26)
    idx_4w = pct_ret(nifty_close, 4)
    print(f"  1M: {idx_1m:.1f}%  3M: {idx_3m:.1f}%  6M: {idx_6m:.1f}%")
except Exception:
    idx_1m = idx_3m = idx_6m = idx_4w = 0

print("\n[3/6] Sector metrics...")

sec_bucket = {}
for ticker in nifty500:
    df = price_data.get(ticker)
    if df is None or len(df) < 60:
        continue
    try:
        close   = df["Close"].squeeze()
        sma20   = close.rolling(20).mean().iloc[-1]
        r4w     = pct_ret(close, 4)
        r6m     = pct_ret(close, 26)
        ab20    = close.iloc[-1] > sma20
        sec     = sector_lookup.get(ticker, "Unknown")
        sec_bucket.setdefault(sec, []).append({"r4w": r4w, "r6m": r6m, "ab20": ab20})
    except Exception:
        pass

sector_stats    = {}
sector_vs_idx   = {}
sector_strength = {}

for sec, items in sec_bucket.items():
    r4  = [x["r4w"] for x in items if x["r4w"] is not None]
    r6  = [x["r6m"] for x in items if x["r6m"] is not None]
    ab  = [x["ab20"] for x in items]
    med4w = np.median(r4) if r4 else 0
    med6m = np.median(r6) if r6 else 0
    brd   = (sum(ab) / len(ab) * 100) if ab else 0
    sector_stats[sec]  = {"4w": med4w, "6m": med6m, "breadth": brd}
    sector_vs_idx[sec] = med4w - (idx_4w or 0)

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

print("\n[4/6] Scoring stocks...")

rows_entry = []; rows_watch = []

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

        avg_vol20   = volume.iloc[-21:-1].mean()
        traded_cr   = (price * avg_vol20) / 1e7
        if traded_cr < LIQ_HARD:
            continue
        liq_flag = "thin" if traded_cr < LIQ_THIN else "ok"

        t_eps    = safe_get(info, "trailingEps",    None)
        f_eps    = safe_get(info, "forwardEps",     None)
        earn_gr  = safe_get(info, "earningsGrowth", 0)
        pe       = safe_get(info, "trailingPE",     999)
        if earn_gr and abs(earn_gr) < 5:
            earn_gr = earn_gr * 100

        if (t_eps is not None and t_eps < 0) and \
           (f_eps is not None and f_eps < 0):
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

        high_52w      = high_s.iloc[-52:].max()
        high_10w      = high_s.iloc[-10:].max()
        pct_from_high = (high_52w - price) / high_52w * 100
        if pct_from_high > DIST_HIGH_MAX:
            continue

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

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

        if cross_week is not None and rsi_at_cross > RSI_UPPER:
            continue

        vol_avg   = volume.iloc[-VOL_WINDOW-1:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0

        vol_persist = 0
        for w in range(1, 5):
            if volume.iloc[-(w+1)] > vol_avg:
                vol_persist += 1
            else:
                break

        atr_s = ta.volatility.AverageTrueRange(high_s, low_s, close, window=14).average_true_range()
        try:
            atr_exp = (atr_s.iloc[-1] / atr_s.iloc[-9] - 1) * 100
        except Exception:
            atr_exp = 0

        r1m = pct_ret(close, 4);  rs1 = (r1m or 0) - (idx_1m or 0)
        r3m = pct_ret(close, 13); rs3 = (r3m or 0) - (idx_3m or 0)
        r6m = pct_ret(close, 26); rs6 = (r6m or 0) - (idx_6m or 0)
        rs_periods = sum(1 for rs in [rs1, rs3, rs6] if rs > 0)

        s_stats  = sector_stats.get(sector, {"4w":0,"6m":0,"breadth":0})
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

        if cross_week is None and not is_watch:
            continue

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
            rng = df["High"].squeeze().iloc[-1] - df["Low"].squeeze().iloc[-1]
            b4 += 3 if rng > 0 and (price - df["Low"].squeeze().iloc[-1])/rng >= 0.75 else 0
        except Exception:
            pass

        b5 = 10 if vol_persist>=3 else (5 if vol_persist==2 else
             (2 if vol_persist==1 else 0))
        b6 = 5 if atr_exp>=20 else (3 if atr_exp>=10 else 0)
        pillar_b = b1+b2+b3+b4+b5+b6

        c1 = 5 if sec_4w>=5 else (3 if sec_4w>=2 else (1 if sec_4w>=0 else 0))
        c2 = 5 if sec_vs_i>=3 else (3 if sec_vs_i>=1 else (1 if sec_vs_i>=0 else 0))
        c3 = 5 if sec_brd>=70 else (3 if sec_brd>=60 else (1 if sec_brd>=50 else 0))
        pillar_c = c1+c2+c3

        d1 = 5 if (sales_gr and sales_gr>=25) else (3 if (sales_gr and sales_gr>=15)
             else (1 if (sales_gr and sales_gr>=5) else 0))
        d2 = 5 if (pat_gr and pat_gr>=30) else (3 if (pat_gr and pat_gr>=15)
             else (1 if (pat_gr and pat_gr>=5) else 0))
        d3 = 5 if (fwd_eps_gr and fwd_eps_gr>=20) else (3 if (fwd_eps_gr and fwd_eps_gr>=10)
             else (1 if (fwd_eps_gr and fwd_eps_gr>=0) else 0))
        pillar_d = d1+d2+d3

        raw   = pillar_a + pillar_b + pillar_c + pillar_d
        total = round(round((raw / RAW_MAX) * 100) * regime_mult)

        label = ("🔥 High Momentum"      if total >= 76 else
                 "⚡ Momentum Candidate" if total >= 61 else
                 "📈 Building Up"        if total >= 41 else
                 "👀 Weak Signal")

        row = {
            "ticker":        ticker,
            "company":       company,
            "sector":        sector,
            "is_cyclical":   "✓" if is_cycl else "",
            "price":         round(price, 2),
            "pct_from_high": round(pct_from_high, 1),
            "rsi":           round(rsi_now, 1),
            "rsi_cross":     f"{cross_week}w ago" if cross_week is not None else "—",
            "rsi_at_cross":  round(rsi_at_cross, 1) if rsi_at_cross else "—",
            "vol_ratio":     round(vol_ratio, 2),
            "vol_persist":   vol_persist,
            "rs_periods":    rs_periods,
            "atr_exp_pct":   round(atr_exp, 1),
            "sales_gr":      round(sales_gr, 1) if sales_gr is not None else "—",
            "pat_gr":        round(pat_gr, 1) if pat_gr is not None else "—",
            "fwd_eps_gr":    round(fwd_eps_gr, 1) if fwd_eps_gr is not None else "—",
            "pe_ratio":      round(pe, 1) if pe not in (None, 999) else "—",
            "traded_cr":     round(traded_cr, 1),
            "liq_flag":      liq_flag,
            "sec_breadth":   round(sec_brd, 1),
            "regime":        regime,
            "pillar_a":      pillar_a,
            "pillar_b":      pillar_b,
            "pillar_c":      pillar_c,
            "pillar_d":      pillar_d,
            "raw_score":     raw,
            "total_score":   total,
            "label":         label,
        }

        if cross_week is not None:
            rows_entry.append(row)
        elif is_watch:
            rows_watch.append(row)

    except Exception:
        continue

print("\n[5/6] Sector caps...")

def apply_sector_cap(rows):
    df_  = pd.DataFrame(rows).sort_values("total_score", ascending=False)
    sec_count = {}; kept = []; overflow = []
    for _, r in df_.iterrows():
        sec = r["sector"]
        cap = get_sector_cap(sec, r["total_score"])
        cnt = sec_count.get(sec, 0)
        if cnt < cap:
            kept.append(r); sec_count[sec] = cnt + 1
        else:
            overflow.append(r)
    return kept, overflow

print("\n[6/6] Building output...")

DCOLS = ["ticker","company","sector","is_cyclical","price","pct_from_high",
         "rsi","rsi_cross","vol_ratio","vol_persist","rs_periods","atr_exp_pct",
         "sales_gr","pat_gr","fwd_eps_gr","pe_ratio","traded_cr","liq_flag",
         "sec_breadth","regime","pillar_a","pillar_b","pillar_c","pillar_d",
         "total_score","label"]

if not rows_entry:
    print("⚠️  No momentum entry stocks found")
    mom_entry = pd.DataFrame(columns=DCOLS)
    mom_sector_overflow = pd.DataFrame(columns=DCOLS)
else:
    kept, overflow = apply_sector_cap(rows_entry)
    mom_entry = (pd.DataFrame(kept).sort_values("total_score", ascending=False)
                 .reset_index(drop=True) if kept else pd.DataFrame(columns=DCOLS))
    mom_sector_overflow = (pd.DataFrame(overflow).sort_values("total_score", ascending=False)
                           .reset_index(drop=True) if overflow else pd.DataFrame(columns=DCOLS))

mom_watch = pd.DataFrame(rows_watch) if rows_watch else pd.DataFrame(columns=DCOLS)
if not mom_watch.empty and "pillar_a" in mom_watch.columns:
    mom_watch = (mom_watch[mom_watch["pillar_a"] >= WATCH_MIN_A]
                 .sort_values(["pct_from_high","rsi"], ascending=[True,False])
                 .reset_index(drop=True))

print(f"""
{'='*65}
MOMENTUM SCREENER | Regime: {regime}
{'='*65}
  mom_entry          : {len(mom_entry)} stocks
  mom_watch          : {len(mom_watch)} stocks
  mom_sector_overflow: {len(mom_sector_overflow)} stocks
{'='*65}
""")

# ============================================================
# CELL 6 — Export to Google Sheets (Service Account auth)
# ============================================================

SHEET_NAME    = "Nifty500 Screener"
MAX_SNAPSHOTS = 999

CONTRA_COLS   = ["ticker", "company", "sector", "price", "total_score", "phase"]
MOMENTUM_COLS = ["ticker", "company", "sector", "is_cyclical", "price", "total_score", "label"]
CONTRA_TABS   = {"P2 Entry", "P1 Watch", "All Contra"}
MOMENTUM_TABS = {"Mom Entry", "Mom Watch", "Sec Overflow"}

try:
    gc = get_gspread_client()

    try:
        sh = gc.open(SHEET_NAME)
        print(f"✅ Opened existing: {SHEET_NAME}")
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
        print(f"📄 Created new: {SHEET_NAME}")

    timestamp = datetime.now().strftime("%d %b %Y %H:%M")
    _iso = datetime.now().isocalendar()
    date_tag  = f"{_iso.year}-W{_iso.week:02d}"

    TAB_GROUPS = [
        ("P2 Entry",     p2_final),
        ("P1 Watch",     p1_final),
        ("All Contra",   master_df),
        ("Mom Entry",    mom_entry),
        ("Mom Watch",    mom_watch),
        ("Sec Overflow", mom_sector_overflow),
    ]

    def write_tab(spreadsheet, tab_name, df, base_name):
        if base_name in MOMENTUM_TABS:
            show_cols = MOMENTUM_COLS
        else:
            show_cols = CONTRA_COLS

        df = df.copy()
        df.insert(0, "week", date_tag)
        show_cols = ["week"] + show_cols
        available = [c for c in show_cols if c in df.columns]
        df_out    = df[available].copy().fillna("").astype(str)
        data      = [df_out.columns.tolist()] + df_out.values.tolist()

        try:
            spreadsheet.del_worksheet(spreadsheet.worksheet(tab_name))
        except gspread.WorksheetNotFound:
            pass

        ws = spreadsheet.add_worksheet(
            title=tab_name,
            rows=max(len(df_out) + 10, 50),
            cols=len(available) + 2
        )
        ws.update(values=[[f"Updated: {timestamp} | {len(df_out)} stocks"]], range_name="A1")
        ws.update(values=data, range_name="A3")
        ws.format(f"A3:{chr(64 + len(available))}3", {
            "backgroundColor": {"red": 0.13, "green": 0.13, "blue": 0.23},
            "textFormat": {
                "bold": True,
                "foregroundColor": {"red": 1, "green": 1, "blue": 1}
            },
            "horizontalAlignment": "CENTER"
        })
        ws.freeze(rows=3)
        print(f"   📊 '{tab_name}' → {len(df_out)} rows")

    print(f"\n📝 Writing snapshot [{date_tag}]...")
    for base_name, df in TAB_GROUPS:
        if df is not None and not df.empty:
            write_tab(sh, f"{base_name} [{date_tag}]", df, base_name)

    print(f"\n🧹 Pruning old snapshots (keeping last {MAX_SNAPSHOTS})...")

    all_tabs      = sh.worksheets()
    tab_titles    = [ws.title for ws in all_tabs]
    date_pattern  = re.compile(r"\[(\d{4}-W\d{2})\]")

    found_dates = sorted(set(
        m.group(1)
        for t in tab_titles
        for m in [date_pattern.search(t)] if m
    ), key=lambda d: d)

    dates_to_delete = found_dates[:-MAX_SNAPSHOTS] if len(found_dates) > MAX_SNAPSHOTS else []

    deleted = 0
    for ws in all_tabs:
        m = date_pattern.search(ws.title)
        if m and m.group(1) in dates_to_delete:
            sh.del_worksheet(ws)
            print(f"   🗑️  Deleted: {ws.title}")
            deleted += 1

    if deleted == 0:
        print(f"   ✅ Nothing to prune")

    retained = [d for d in found_dates if d not in dates_to_delete]

    print(f"""
{'='*55}
✅ EXPORT COMPLETE — {timestamp}
{'='*55}
Sheet     : {SHEET_NAME}
Snapshot  : {date_tag}
Retained  : {retained}

Contra:
  P2 Entry     : {len(p2_final)} stocks
  P1 Watch     : {len(p1_final)} stocks
  All Contra   : {len(master_df)} stocks

Momentum:
  Mom Entry    : {len(mom_entry)} stocks
  Mom Watch    : {len(mom_watch)} stocks
  Sec Overflow : {len(mom_sector_overflow)} stocks
{'='*55}
""")
    print(f"🔗 Link: {sh.url}")

    # ── Telegram Summary ──────────────────────────────────────
    p2_top = p2_final.head(5)[["ticker","total_score"]].to_string(index=False) if not p2_final.empty else "None"
    mom_top = mom_entry.head(5)[["ticker","total_score","label"]].to_string(index=False) if not mom_entry.empty else "None"

    tg_msg = f"""📊 <b>Nifty 500 Screener — {date_tag}</b>

<b>CONTRA:</b>
  P2 Entry Ready : {len(p2_final)}
  P1 Watchlist   : {len(p1_final)}

Top P2:
<pre>{p2_top}</pre>

<b>MOMENTUM ({regime}):</b>
  Entry  : {len(mom_entry)}
  Watch  : {len(mom_watch)}

Top Entry:
<pre>{mom_top}</pre>

🔗<a href="{sh.url}">Google Sheet kholo</a>"""

    send_telegram(tg_msg)

except Exception as e:
    print(f"❌ Export failed: {e}")
    send_telegram(f"❌ Screener FAILED on {datetime.now().strftime('%d %b %Y')}\nError: {e}")
