Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# GBP→EUR Telegram Alerts

Simple cron-friendly script that watches the GBP→EUR rate, stores recent history, and sends a Telegram alert when the move exceeds a static or volatility-adjusted threshold.

## Requirements
- Python 3.14 used/tested; should work on 3.11+ as well
- Install dependencies: `pip install -r requirements.txt`
- A Telegram bot token and the chat ID to send alerts to

## Configuration (.env)
Create a `.env` file in the repo root:
```
BOT_TOKEN="123456789:AA...your_bot_token..."
CHAT_ID=123456789        # chat or user ID to notify
FX_URL="https://api.frankfurter.app/latest?from=GBP&to=EUR"
```
Notes:
- Leave quotes around values if they contain special characters.
- `CHAT_ID` can be a user ID (DM) or a group/channel ID if the bot is invited to that chat.

## Getting Telegram credentials
1) Create the bot  
   - In Telegram, message `@BotFather` → `/newbot` → name it and get the HTTP API token (`BOT_TOKEN`).
2) Get your chat ID  
   - Start a chat with your bot (send any message).  
   - Call the Telegram API:  
     ```
     curl "https://api.telegram.org/bot<your token>/getUpdates"
     ```
   - Look for `"chat":{"id": ...}` in the JSON; use that as `CHAT_ID`.  
   - For groups, add the bot to the group and send a message first so `getUpdates` shows the group chat ID.

## Running manually
```
python gbp_telegram_alert.py
```
Creates/updates `gbp_last_rates.json` for history and writes rotating logs to `gbp_alert.log` (1 MB per file, 5 backups).

## Cron setup
Recommended: every 10 minutes (good balance for 1h detection and API usage). Example crontab entry:
```
*/10 * * * * cd /home/myname/exchange_rate_monitor && . .venv/bin/activate && set -a && [ -f .env ] && . .env && set +a && python gbp_telegram_alert.py >> cron.log 2>&1
```
Adjust the path to your repo and virtualenv as needed.

## How alerts work
- Static thresholds: 0.2% (1h), 0.5% (6h), 1.0% (1d).
- Rolling-vol thresholds: uses ~30 days of stored rates to estimate hourly volatility; per horizon threshold = max(vol-based, static, 0.1%). Falls back to static when not enough history is available.
- Alerts send via Telegram Markdown with up/down direction and the latest rate.

## Logs and data
- Alerts and status logs: `gbp_alert.log` (rotates, keeps 5 backups).
- Raw script output (from cron): `cron.log` (not rotated by the script; rotate via logrotate/cron if desired). Example logrotate entry (`/etc/logrotate.d/exchange_rate_monitor`) for daily rotation with compression:
  ```
  /home/myname/exchange_rate_monitor/cron.log {
      daily
      rotate 7
      compress
      delaycompress
      missingok
      notifempty
      copytruncate
  }
  ```
- Stored rates: `gbp_last_rates.json` (keeps only the needed lookback window).
