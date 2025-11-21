#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

LOG_DIR="$script_dir/logs"
ENV_FILE="$script_dir/.env"

mkdir -p "$LOG_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE (see README.md for required settings)." >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  python_cmd="$PYTHON"
elif [[ -x "$script_dir/.venv/bin/python" ]]; then
  python_cmd="$script_dir/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="$(command -v python3)"
else
  echo "Could not find python3. Install Python 3.11+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

exec "$python_cmd" "$script_dir/gbp_telegram_alert.py" >> "$LOG_DIR/cron.log" 2>&1
