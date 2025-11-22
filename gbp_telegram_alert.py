#!.venv/bin python3
"""GBP→EUR alerts with static and volatility-aware thresholds."""

import datetime
import itertools
import logging
import math
import pathlib
import statistics
from logging.handlers import RotatingFileHandler
from typing import Any

import pydantic
import pydantic_settings
import requests
import tinydb

from config import RuntimeConfig, build_runtime_config, load_app_config
from data import Datafile, put_new_rate_into_db, sorted_samples

DEV_FORCE_INSERTION = True  # Set to True to always insert new rate for testing


class Settings(pydantic_settings.BaseSettings):
    """Environment configuration for Telegram and FX source."""

    bot_token: str
    chat_id: int
    fx_url: pydantic.HttpUrl


ENV = Settings(_env_file=".env")
CONFIG_FILE = pathlib.Path(__file__).parent / "config.toml"
DATA_FILE = pathlib.Path(__file__).parent / "data" / "gbp_last_rates.json"
DB_FILE = DATA_FILE.parent / "gbp_alert_db.json"
LOG_FILE = pathlib.Path(__file__).parent / "logs" / "gbp_alert.log"


# ---------------------------


def format_alert_message(
    breaches: dict[str, dict[str, float]],
    volatilities: dict[str, float | None],
    config: RuntimeConfig,
) -> str:
    """Render a human-readable alert summary."""
    lines: list[str] = ["*GBP→EUR ALERTS*", ""]
    for table_name, hits in sorted(breaches.items()):
        period = _period_from_table_name(table_name, config)
        if period is None or period not in config.alert_change_over_time:
            continue
        delta_percent = hits.get("static", hits.get("dynamic"))
        if delta_percent is None:
            continue
        static_threshold, dynamic_threshold = _thresholds_for_period(
            period, volatilities.get(table_name), config
        )
        triggered = []
        if "static" in hits:
            triggered.append(f"static ≥ {static_threshold:.2f}%")
        if "dynamic" in hits and dynamic_threshold is not None:
            triggered.append(f"dynamic ≥ {dynamic_threshold:.2f}%")
        period_label = _format_period(period)
        lines.append(
            f"- {period_label}: Δ {delta_percent:+.2f}% ({'; '.join(triggered)})"
        )

    return "\n".join(lines)


# ---------------------------


def get_current_rate(url: pydantic.HttpUrl) -> Datafile:
    """Fetch the current GBP→EUR rate from the given API URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return Datafile(rate=float(data["rates"]["EUR"]), source=url)


def _sigma_per_hour(
    samples: list[tuple[datetime.datetime, float]], vol_min_points: int
) -> float | None:
    """Return volatility (percent per sqrt hour) from timestamp/rate samples."""
    if len(samples) < vol_min_points:
        return None

    terms = []
    for (tsa, ra), (tsb, rb) in itertools.pairwise(samples):
        delta_hours = (tsb - tsa).total_seconds() / 3600
        if delta_hours > 0:
            terms.append((math.log(rb / ra) ** 2) / delta_hours)

    if len(terms) < max(3, vol_min_points - 1):
        return None

    return math.sqrt(statistics.fmean(terms)) * 100


def _period_from_table_name(
    table_name: str, config: RuntimeConfig
) -> datetime.timedelta | None:
    """Derive the timedelta represented by the TinyDB table name."""
    return config.table_periods.get(table_name)


def _format_period(period: datetime.timedelta) -> str:
    """Return a short label for a timedelta (e.g., 1h, 6h, 1d)."""
    total_seconds = int(period.total_seconds())
    if total_seconds % 86400 == 0:
        days = total_seconds // 86400
        return f"{days}d"
    if total_seconds % 3600 == 0:
        hours = total_seconds // 3600
        return f"{hours}h"
    if total_seconds % 60 == 0:
        minutes = total_seconds // 60
        return f"{minutes}m"
    return f"{total_seconds}s"


def _thresholds_for_period(
    period: datetime.timedelta, sigma_per_hour: float | None, config: RuntimeConfig
) -> tuple[float, float | None]:
    """Return static and dynamic thresholds for a period."""
    static_threshold = max(
        config.min_alert_threshold, config.alert_change_over_time[period]
    )
    if sigma_per_hour is None:
        return static_threshold, None

    delta_hours = period.total_seconds() / 3600
    dynamic_threshold = max(
        config.min_alert_threshold,
        sigma_per_hour * math.sqrt(delta_hours) * config.vol_multiplier,
    )
    return static_threshold, dynamic_threshold


def filter_tables_over_thresholds(
    deltas: dict[str, float | None],
    volatilities: dict[str, float | None],
    config: RuntimeConfig,
) -> dict[str, dict[str, float]]:
    """Return tables whose percent change exceeds static and/or dynamic thresholds."""
    breaches: dict[str, dict[str, float]] = {}
    for table_name, delta_percent in deltas.items():
        if delta_percent is None:
            continue
        period = _period_from_table_name(table_name, config)
        if period is None or period not in config.alert_change_over_time:
            continue
        sigma_per_hour = volatilities.get(table_name)
        static_threshold, dynamic_threshold = _thresholds_for_period(
            period, sigma_per_hour, config
        )

        hits: dict[str, float] = {}
        if abs(delta_percent) >= static_threshold:
            hits["static"] = delta_percent
        if dynamic_threshold is not None and abs(delta_percent) >= dynamic_threshold:
            hits["dynamic"] = delta_percent

        if hits:
            breaches[table_name] = hits

    return breaches


def send_telegram_message(text: str, bot_token: str, chat_id: int) -> dict[str, Any]:
    """Send a Telegram message using the bot token and chat ID."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": str(chat_id),
        "text": text,
        "parse_mode": "Markdown",
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def calculate_volatilities_per_period(
    db: tinydb.TinyDB, table_names: list[str], config: RuntimeConfig
) -> dict[str, float | None]:
    """Calculate volatilities for each period table provided."""
    return {
        table_name: _sigma_per_hour(
            sorted_samples(db.table(table_name)), config.vol_min_points
        )
        for table_name in table_names
    }


def calculate_delta_percents(
    db: tinydb.TinyDB, table_names: list[str]
) -> dict[str, float | None]:
    """Compute percent change between oldest and newest entries per period table."""
    deltas: dict[str, float | None] = {}
    for table_name in table_names:
        samples = sorted_samples(db.table(table_name))
        if len(samples) < 2:
            deltas[table_name] = None
            continue
        _, oldest_rate = samples[0]
        _, newest_rate = samples[-1]
        deltas[table_name] = ((newest_rate - oldest_rate) / oldest_rate) * 100
    return deltas


def main() -> None:
    """Fetch, evaluate, and alert on GBP→EUR moves."""
    app_config = load_app_config(CONFIG_FILE)
    runtime_config = build_runtime_config(app_config)

    data_dir = DATA_FILE.parent
    log_dir = LOG_FILE.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(LOG_FILE, app_config.logging.max_bytes, app_config.logging.backups)

    with tinydb.TinyDB(DB_FILE) as db:
        new_rate = get_current_rate(ENV.fx_url)
        logging.info("Fetched new rate: %s from %s", new_rate.rate, new_rate.source)
        tables_updated = put_new_rate_into_db(
            db, new_rate, runtime_config, force=DEV_FORCE_INSERTION
        )
        if not tables_updated:
            logging.info("No new entries added to any period table; exiting.")
            return
        volatilities = calculate_volatilities_per_period(
            db, tables_updated, runtime_config
        )
        logging.info("Calculated volatilities per period: %s", volatilities)
        deltas = calculate_delta_percents(db, tables_updated)
        logging.info("Calculated delta percents per period: %s", deltas)
        breaches = filter_tables_over_thresholds(deltas, volatilities, runtime_config)
        logging.info("Tables exceeding thresholds: %s", breaches)
        if breaches:
            message_text = format_alert_message(breaches, volatilities, runtime_config)
            send_telegram_message(message_text, ENV.bot_token, ENV.chat_id)
            logging.info("Alert sent to Telegram.")


def setup_logging(log_file: pathlib.Path, max_bytes: int, backups: int):
    """Set up logging with rotating file handler and console output."""
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backups
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


if __name__ == "__main__":
    main()
