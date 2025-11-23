Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# FX Telegram Alerts

Simple cron-friendly script that watches a configurable currency pair, pulls daily ECB rates from Frankfurter, and sends a Telegram alert when the move exceeds a static or volatility-adjusted threshold.

## Requirements
- Python 3.14 used/tested; should work on 3.11+ as well
- Install dependencies: `pip install -r requirements.txt`
- A Telegram bot token and the chat ID to send alerts to
- Console logs use `rich` for nicer output; file logs stay plain for easy parsing.

## Configuration
### `.env`
Create a `.env` file in the repo root:
```
BOT_TOKEN="123456789:AA...your_bot_token..."
CHAT_ID=123456789        # chat or user ID to notify
```
Notes:
- Leave quotes around values if they contain special characters.
- `CHAT_ID` can be a user ID (DM) or a group/channel ID if the bot is invited to that chat.

### `config.toml`
Control the currency pair and thresholds:
```toml
[currencies]
base = "GBP"
quote = "EUR"

[alerts]
min_alert_threshold = 0.1
vol_multiplier = 1.75
vol_min_points = 6

[[alerts.periods]]
period = "1d"
threshold = 0.75

[[alerts.periods]]
period = "3d"
threshold = 1.5

[[alerts.periods]]
period = "7d"
threshold = 2.5

[[alerts.periods]]
period = "14d"
threshold = 3.5

[[alerts.periods]]
period = "30d"
threshold = 5.0

[[alerts.periods]]
period = "90d"
threshold = 7.5
```
Adjust/add alert periods as needed. The Frankfurter API publishes one fix per working day around 16:00 CET; shorter periods will still use the latest and closest available prior date.

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
```bash
python gbp_telegram_alert.py
# or
./run.sh
```
Writes rotating logs to `logs/gbp_alert.log` (1 MB per file, 5 backups).

### DEV_MODE
For end-to-end testing outside publish times (or on weekends), set `DEV_MODE=true` in `.env`. This skips the weekend guard and freshness wait and uses whatever the API returns.

## Cron setup (daily after ECB publish)
Frankfurter publishes once per working day around 16:00 CET. Run after that time and skip weekends:
```
# Server timezone CET/CEST
10 16 * * 1-5 cd /<PATH_TO_REPO> && ./run.sh
# If server runs in UTC, trigger at 15:10 UTC to match 16:10 CET in winter
# 10 15 * * 1-5 cd /<PATH_TO_REPO> && ./run.sh
```
Adjust the path to your repo and virtualenv as needed.

## How alerts work
- Static thresholds: configured in `alerts.periods` per timeframe.
- Rolling-vol thresholds: uses historical rates fetched each run; per horizon threshold = max(vol-based, static, 0.1%). Falls back to static when not enough history is available.
- No local database: each run pulls the required history from Frankfurter and computes in memory.
- Alerts send via Telegram Markdown with up/down direction and the latest rate.

## Logs
- Alerts and status logs: `logs/gbp_alert.log` (rotates, keeps 5 backups).
- Raw script output (from cron): `logs/cron.log` (not rotated by the script; rotate via logrotate/cron if desired). Example logrotate entry (`/etc/logrotate.d/exchange_rate_monitor`) for daily rotation with compression:
  ```
  /<PATH_TO_REPO>/logs/cron.log {
      daily
      rotate 7
      compress
      delaycompress
      missingok
      notifempty
      copytruncate
  }
  ```
