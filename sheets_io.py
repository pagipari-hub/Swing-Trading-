# ============================================================
# sheets_io.py — Google Sheets auth + transposed-sheet read/write.
#
# Sheet structure (per tab):
#   Col A : key  (ticker|metric)  — for VLOOKUP
#   Col B : metric name
#   Col C : ticker
#   Col D : company
#   Col E : sector
#   Col F : is_cyclical  (Momentum only; Contra sheet skips this)
#   Next+ : W23, W24, W25 ...  (week columns, added right)
# ============================================================

import os
import json
import time
import gspread
from google.oauth2.service_account import Credentials


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


def ensure_worksheet(sh, name, rows=15000, cols=300):
    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=rows, cols=cols)
        return ws

def col_letter(n):
    result = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result

def build_transposed_sheet(ws, df, metrics, week_tag, has_cyclical=False):
    fixed_cols   = 6 if has_cyclical else 5
    first_wk_col = fixed_cols + 1

    existing = ws.get_all_values()

    if not existing or not existing[0]:
        print(f"   📄 '{ws.title}' — first run, building structure...")

        header = ["key", "metric", "ticker", "company", "sector"]
        if has_cyclical:
            header.append("is_cyclical")
        header.append(week_tag)

        all_rows = [header]

        tickers_sorted = sorted(df["ticker"].unique())
        for ticker in tickers_sorted:
            row_df  = df[df["ticker"] == ticker]
            if row_df.empty:
                continue
            r       = row_df.iloc[0]
            company = str(r.get("company", ""))
            sector  = str(r.get("sector", ""))
            is_cyc  = str(r.get("is_cyclical", "")) if has_cyclical else None

            for metric in metrics:
                key = f"{ticker}|{metric}"
                val = str(r.get(metric, "—")) if not row_df.empty else "—"
                row = [key, metric, ticker, company, sector]
                if has_cyclical:
                    row.append(is_cyc)
                row.append(val)
                all_rows.append(row)

        ws.update(values=all_rows, range_name="A1")
        print(f"   ✅ '{ws.title}' — {len(all_rows)-1} rows written, week col: {col_letter(first_wk_col+1)} ({week_tag})")
        return

    header_row = existing[0]

    if week_tag in header_row:
        week_col_idx = header_row.index(week_tag) + 1
        print(f"   ⚠️  '{ws.title}' — {week_tag} already exists at col {col_letter(week_col_idx)}, overwriting...")
    else:
        week_col_idx = len(header_row) + 1
        ws.update(values=[[week_tag]],
                  range_name=f"{col_letter(week_col_idx)}1")
        print(f"   📅 '{ws.title}' — New week col {col_letter(week_col_idx)} = {week_tag}")

    existing_keys = {row[0] for row in existing[1:] if row}
    tickers_sorted = sorted(df["ticker"].unique())
    new_rows_to_append = []
    for ticker in tickers_sorted:
        row_df = df[df["ticker"] == ticker]
        if row_df.empty:
            continue
        r       = row_df.iloc[0]
        company = str(r.get("company", ""))
        sector  = str(r.get("sector", ""))
        is_cyc  = str(r.get("is_cyclical", "")) if has_cyclical else None
        for metric in metrics:
            key = f"{ticker}|{metric}"
            if key in existing_keys:
                continue
            row = [key, metric, ticker, company, sector]
            if has_cyclical:
                row.append(is_cyc)
            row += [""] * (week_col_idx - len(row) - 1)
            row.append(str(r.get(metric, "—")))
            new_rows_to_append.append(row)

    if new_rows_to_append:
        ws.append_rows(new_rows_to_append, value_input_option="RAW")
        print(f"   ➕ '{ws.title}' — {len(new_rows_to_append)} new metric rows appended "
              f"(e.g. cross_failed/re_confirmed on a pre-existing sheet)")
        existing = ws.get_all_values()

    val_map = {}
    for _, r in df.iterrows():
        ticker = r.get("ticker", "")
        for metric in metrics:
            key = f"{ticker}|{metric}"
            val_map[key] = str(r.get(metric, "—"))

    updates = []
    for row_idx, row in enumerate(existing[1:], start=2):
        if not row:
            continue
        key = row[0] if row else ""
        val = val_map.get(key, "—")
        updates.append({
            "range": f"{col_letter(week_col_idx)}{row_idx}",
            "values": [[val]]
        })

    chunk = 500
    for i in range(0, len(updates), chunk):
        ws.batch_update(updates[i:i+chunk])
        if i + chunk < len(updates):
            time.sleep(1)

    print(f"   ✅ '{ws.title}' — {len(updates)} cells updated for {week_tag}")

