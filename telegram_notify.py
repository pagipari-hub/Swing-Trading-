# ============================================================
# telegram_notify.py — Weekly Telegram summary delivery
# ============================================================

import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


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
