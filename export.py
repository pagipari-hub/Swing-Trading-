# ============================================================
# export.py — Write Contra results to Google Sheets and send the
# weekly Telegram summary.
#
# Momentum export/Telegram removed -- Momentum now runs standalone
# via main_momentum.py (see momentum.yml), which writes its own
# "Momentum Data" tab and sends its own Telegram message.
#
# Guard: if Contra produced zero rows (e.g. an upstream data failure
# like the Nifty 500 universe fetch failing with no fallback
# available), skip the export entirely rather than overwriting an
# existing week's column with all-zero data.
# ============================================================

import time
import gspread

from config import SHEET_NAME, WEEK_TAG, TIMESTAMP, CONTRA_METRICS
from sheets_io import get_gspread_client, ensure_worksheet, build_transposed_sheet
from telegram_notify import send_telegram

# Retry transient Google API failures (5xx — the service's problem, not
# ours, and it typically clears within seconds). Don't retry on things
# like auth errors (403) or "not found" (404) — those won't fix
# themselves by waiting, so fail fast on those instead.
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 5   # 5s, 10s, 20s, 40s


def _is_retryable(exc):
    if isinstance(exc, gspread.exceptions.APIError):
        try:
            status = exc.response.status_code
        except Exception:
            status = None
        return status in RETRYABLE_STATUS_CODES
    return False


def _with_retries(fn, label):
    """Runs fn() with exponential backoff on transient (5xx) API errors."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"  ⚠️  {label} failed (attempt {attempt}/{MAX_RETRIES}): {e} "
                  f"— retrying in {wait}s...")
            time.sleep(wait)
    raise last_exc


def export_and_notify(contra_df, p2_df, p1_df, contra_regime):
    """
    Exports Contra results to Google Sheets and sends the Telegram
    summary. Returns True if the export ran, False if it was skipped
    due to the zero-row guard.

    NOTE: the "Market" line previously shown in Telegram (Risk-On/Off,
    Nifty RSI, 1M/3M/6M index returns) came from Momentum's regime
    computation, which no longer runs here. Replaced with contra_regime
    only. Tell me if you want independent Nifty-level context re-added.
    """
    if len(contra_df) == 0:
        msg = (f"❌ Contra screener produced 0 rows for {WEEK_TAG} (likely upstream "
               f"data failure) — skipping Google Sheets export to avoid overwriting "
               f"existing data.")
        print(msg)
        send_telegram(msg)
        return False

    contra_export = contra_df.rename(columns={"score": "total_score"}).copy()

    try:
        gc = get_gspread_client()

        try:
            sh = _with_retries(lambda: gc.open(SHEET_NAME), "Open spreadsheet")
            print(f"✅ Opened existing spreadsheet: {SHEET_NAME}")
        except gspread.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            print(f"📄 Created new spreadsheet: {SHEET_NAME}")

        ws_contra = _with_retries(
            lambda: ensure_worksheet(sh, "Contra Data", rows=15000, cols=300),
            "Open Contra Data worksheet",
        )
        _with_retries(
            lambda: build_transposed_sheet(ws_contra, contra_export, CONTRA_METRICS,
                                            WEEK_TAG, has_cyclical=False),
            "Write Contra Data",
        )

        print(f"""
{'='*55}
✅ EXPORT COMPLETE — {TIMESTAMP}
{'='*55}
Sheet    : {SHEET_NAME}
Week tag : {WEEK_TAG}

Contra Data  : {len(contra_df)} rows appended
  Phase 2    : {len(p2_df)}
  Phase 1    : {len(p1_df)}
{'='*55}
""")
        print(f"🔗 Link: {sh.url}")

        mkt_line = f"📍 <b>Contra regime:</b> {contra_regime}"

        def fmt_top10_contra(df):
            top = df.sort_values("score", ascending=False).head(10)
            if top.empty:
                return "  —"
            lines = []
            for _, r in top.iterrows():
                sym = str(r["ticker"]).replace(".NS", "")
                lines.append(f"  {sym:<14} {r['score']}")
            return "\n".join(lines)

        p2_block = fmt_top10_contra(p2_df)

        tg_msg = (
            f"📊 <b>Nifty 500 Contra Screener — {WEEK_TAG}</b>\n"
            f"🕐 {TIMESTAMP}\n\n"
            f"{mkt_line}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔵 <b>CONTRA</b>\n"
            f"  Phase 2 Entry Ready : <b>{len(p2_df)}</b>\n"
            f"  Phase 1 Watchlist   : {len(p1_df)}\n"
            f"  Total screened      : {len(contra_df)}\n\n"
            f"<b>Top 10 Phase 2:</b>\n"
            f"<pre>{p2_block}</pre>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f'🔗 <a href="{sh.url}">Open Google Sheet</a>'
        )

        send_telegram(tg_msg)

    except Exception as e:
        err_msg = f"❌ Export failed: {e}"
        print(err_msg)
        import traceback; traceback.print_exc()
        send_telegram(f"❌ Screener FAILED on {TIMESTAMP}\nError: {e}")

    return True
