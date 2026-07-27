"""
main_momentum.py
=================
Standalone entry point for the stock-level Momentum screener.
Deliberately decoupled from main.py (Contra + Sector) -- see momentum.yml,
which triggers this via workflow_dispatch only (no cron).

Shares all underlying modules (config.py, universe.py, data_fetch.py,
helpers.py, sheets_io.py, telegram_notify.py) with the rest of the
pipeline -- only the orchestration/trigger is separate.
"""

from momentum_screener import run_momentum_screener   # ADAPT: match actual function name
from sheets_io import export_momentum_to_sheet         # ADAPT: match actual function name
from telegram_notify import send_telegram_message       # ADAPT: match actual function name


def main() -> None:
    print("[main_momentum] Starting standalone Momentum run...")

    results_df = run_momentum_screener()

    if results_df is None or results_df.empty:
        print("[main_momentum] No Momentum signals this run -- skipping Sheets/Telegram.")
        return

    export_momentum_to_sheet(results_df)

    message = format_momentum_telegram_message(results_df)
    send_telegram_message(message)

    print(f"[main_momentum] Done. {len(results_df)} signal(s) processed.")


def format_momentum_telegram_message(df) -> str:
    """ADAPT: reuse whatever formatting momentum_screener.py already produces
    if it exists -- this is a placeholder so main_momentum.py is runnable
    standalone without guessing at your existing formatter's signature."""
    lines = ["📈 *Momentum Screener — Manual Run*", ""]
    lines.append(f"Signals found: {len(df)}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
