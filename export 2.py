# ============================================================
# export.py — Write Contra + Momentum results to Google Sheets
# and send the weekly Telegram summary.
#
# Guard: if BOTH screeners produced zero rows (e.g. an upstream
# data failure like the Nifty 500 universe fetch failing with no
# fallback available), skip the export entirely rather than
# overwriting an existing week's column with all-zero data.
# ============================================================

import gspread

from config import SHEET_NAME, WEEK_TAG, TIMESTAMP, CYCLICAL_NSE, CONTRA_METRICS, MOMENTUM_METRICS
from sheets_io import get_gspread_client, ensure_worksheet, build_transposed_sheet
from telegram_notify import send_telegram


def export_and_notify(contra_df, momentum_df, p2_df, p1_df,
                       mom_entry, mom_emerging, mom_watch,
                       sector_lookup, regime, contra_regime,
                       idx_1m, idx_3m, idx_6m,
                       n_rsi, n_price, n_sma30):
    """
    Exports both screeners to Google Sheets and sends the Telegram
    summary. Returns True if the export ran, False if it was skipped
    due to the zero-row guard.
    """
    if len(contra_df) == 0 and len(momentum_df) == 0:
        msg = (f"❌ Screener produced 0 rows for {WEEK_TAG} (likely upstream data "
               f"failure) — skipping Google Sheets export to avoid overwriting "
               f"existing data.")
        print(msg)
        send_telegram(msg)
        return False

    contra_export = contra_df.rename(columns={"score": "total_score"}).copy()
    
    momentum_export = momentum_df.rename(columns={"score": "total_score"}).copy()
    momentum_export["is_cyclical"] = momentum_export["ticker"].apply(
        lambda t: "Yes" if sector_lookup.get(t, "") in CYCLICAL_NSE else "No"
    )
    
    try:
        gc = get_gspread_client()
    
        try:
            sh = gc.open(SHEET_NAME)
            print(f"✅ Opened existing spreadsheet: {SHEET_NAME}")
        except gspread.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            print(f"📄 Created new spreadsheet: {SHEET_NAME}")
    
        ws_contra = ensure_worksheet(sh, "Contra Data", rows=15000, cols=300)
        build_transposed_sheet(ws_contra, contra_export, CONTRA_METRICS,
                               WEEK_TAG, has_cyclical=False)
    
        ws_mom = ensure_worksheet(sh, "Momentum Data", rows=15000, cols=300)
        build_transposed_sheet(ws_mom, momentum_export, MOMENTUM_METRICS,
                               WEEK_TAG, has_cyclical=True)
    
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
  Emerging   : {len(mom_emerging)}
  Watch      : {len(mom_watch)}
  Regime     : {regime}
{'='*55}
""")
        print(f"🔗 Link: {sh.url}")
    
        try:
            n_rsi_val  = round(float(n_rsi), 1)
            n_sma_diff = round(((float(n_price) / float(n_sma30)) - 1) * 100, 1)
            mkt_icon   = "🔴" if regime == "Risk-Off" else "🟢"
            mkt_line   = (
                f"{mkt_icon} <b>Market:</b> {regime} | "
                f"Nifty500 RSI {n_rsi_val} | "
                f"{'Below' if n_sma_diff < 0 else 'Above'} 30W SMA "
                f"({'+' if n_sma_diff >= 0 else ''}{n_sma_diff}%)\n"
                f"   1M: {idx_1m:+.1f}%  3M: {idx_3m:+.1f}%  6M: {idx_6m:+.1f}%\n"
                f"   Contra regime: {contra_regime}"
            )
        except Exception:
            mkt_line = f"{'🔴' if regime == 'Risk-Off' else '🟢'} <b>Market:</b> {regime}"
    
        def fmt_top10_contra(df):
            top = df.sort_values("score", ascending=False).head(10)
            if top.empty:
                return "  —"
            lines = []
            for _, r in top.iterrows():
                sym = str(r["ticker"]).replace(".NS", "")
                lines.append(f"  {sym:<14} {r['score']}")
            return "\n".join(lines)
    
        def fmt_top10_mom(df):
            top = df.sort_values("score", ascending=False).head(10)
            if top.empty:
                return "  —"
            lines = []
            for _, r in top.iterrows():
                sym   = str(r["ticker"]).replace(".NS", "")
                label = str(r["label"]).replace("🔥 ","🔥").replace("⚡ ","⚡")
                lines.append(f"  {sym:<14} {r['score']}  {label}")
            return "\n".join(lines)
    
        p2_block  = fmt_top10_contra(p2_df)
        mom_block = fmt_top10_mom(mom_entry)
    
        tg_msg = (
            f"📊 <b>Nifty 500 Screener — {WEEK_TAG}</b>\n"
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
            f"🟠 <b>MOMENTUM</b>\n"
            f"  Entry candidates : <b>{len(mom_entry)}</b>\n"
            f"  Emerging         : {len(mom_emerging)}\n"
            f"  Watch list       : {len(mom_watch)}\n"
            f"  Total screened   : {len(momentum_df)}\n\n"
            f"<b>Top 10 Entry:</b>\n"
            f"<pre>{mom_block}</pre>\n"
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
