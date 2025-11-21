#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

mkdir -p logs

source .venv/bin/activate

python gbp_telegram_alert.py >> logs/cron.log 2>&1
