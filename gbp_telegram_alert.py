#!/usr/bin/env python3
"""Daily FX alerts using the Frankfurter (ECB) API."""

import datetime
import itertools
import logging
import math
import pathlib
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import RotatingFileHandler
from typing import Any, Iterable

import pydantic_settings
import requests
from rich.logging import RichHandler

from config import RuntimeConfig, build_runtime_config, load_app_config

FRANKFURTER_API_BASE = "https://api.frankfurter.dev/v1"
REQUEST_TIMEOUT_SECONDS = 15
MAX_WAIT_SECONDS = 3600  # wait up to 1 hour for today's fix
RETRY_SLEEP_SECONDS = 300  # retry every 5 minutes
DEV_MODE = False  # default, can be overridden by .env

CONFIG_FILE = pathlib.Path(__file__).parent / "config.toml"
LOG_FILE = pathlib.Path(__file__).parent / "logs" / "gbp_alert.log"


class Settings(pydantic_settings.BaseSettings):
    """Environment configuration for Telegram."""

    bot_token: str
    chat_id: int
    dev_mode: bool = False


ENV = Settings(_env_file=".env")
DEV_MODE = ENV.dev_mode


# ---------------------------


def format_alert_message(
    breaches: dict[str, dict[str, float]],
    volatilities: dict[str, float | None],
    config: RuntimeConfig,
    pair_label: str,
) -> str:
    """Render a human-readable alert summary."""
    lines: list[str] = [f"*{pair_label} ALERTS*", ""]
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


def fetch_latest_rate(base: str, quote: str) -> tuple[datetime.date, float]:
    """Fetch the latest available rate for the currency pair."""
    url = f"{FRANKFURTER_API_BASE}/latest"
    response = requests.get(
        url, params={"base": base, "symbols": quote}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()
    date = datetime.date.fromisoformat(data["date"])
    rate = float(data["rates"][quote])
    return date, rate


def fetch_timeseries(
    base: str,
    quote: str,
    start_date: datetime.date,
    end_date: datetime.date | None,
) -> dict[datetime.date, float]:
    """Fetch historical rates for the currency pair across the given period."""
    end_segment = end_date.isoformat() if end_date else ""
    url = f"{FRANKFURTER_API_BASE}/{start_date.isoformat()}..{end_segment}"
    response = requests.get(
        url, params={"base": base, "symbols": quote}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()
    return {
        datetime.date.fromisoformat(date_str): float(rates[quote])
        for date_str, rates in data["rates"].items()
        if quote in rates
    }


def wait_for_fresh_rate(
    base: str,
    quote: str,
    *,
    dev_mode: bool = False,
    max_wait_seconds: int = MAX_WAIT_SECONDS,
    retry_sleep_seconds: int = RETRY_SLEEP_SECONDS,
) -> tuple[datetime.date, float] | None:
    """Poll until the latest date is today or until max_wait_seconds elapse."""
    if dev_mode:
        date, rate = fetch_latest_rate(base, quote)
        logging.info(
            "DEV_MODE enabled; accepting latest %s→%s: %s (%s) without waiting.",
            base,
            quote,
            rate,
            date,
        )
        return date, rate

    deadline = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        seconds=max_wait_seconds
    )

    while True:
        fetch_started = datetime.datetime.now(datetime.UTC)
        date, rate = fetch_latest_rate(base, quote)
        logging.info("Fetched latest %s→%s: %s (%s)", base, quote, rate, date)

        if date == datetime.date.today():
            return date, rate

        if fetch_started >= deadline:
            logging.warning(
                "Latest published date is still %s (not today); giving up after waiting.",
                date,
            )
            return None

        sleep_for = min(
            retry_sleep_seconds,
            int((deadline - fetch_started).total_seconds()),
        )
        logging.info(
            "Latest date is %s (today is %s); waiting up to %s seconds before giving up.",
            date,
            datetime.date.today(),
            int((deadline - fetch_started).total_seconds()),
        )
        logging.info(
            "Latest date is %s (today is %s); retrying in %s seconds.",
            date,
            datetime.date.today(),
            sleep_for,
        )
        time.sleep(max(1, sleep_for))


def _history_samples(
    history: dict[datetime.date, float],
) -> list[tuple[datetime.datetime, float]]:
    """Convert daily history into timestamped samples for volatility math."""
    return sorted((_date_to_timestamp(date), rate) for date, rate in history.items())


def _date_to_timestamp(date: datetime.date) -> datetime.datetime:
    """Attach a deterministic time to a date (00:00 UTC)."""
    return datetime.datetime.combine(date, datetime.time(0, 0, tzinfo=datetime.UTC))


def _baseline_date(
    dates: Iterable[datetime.date],
    latest_date: datetime.date,
    period: datetime.timedelta,
) -> datetime.date | None:
    """Pick the newest available date at or before the target window."""
    sorted_dates = sorted(dates)
    target_date = latest_date - period
    candidates = [date for date in sorted_dates if date <= target_date]
    if candidates:
        return candidates[-1]

    fallback = [date for date in sorted_dates if date < latest_date]
    if fallback:
        return fallback[-1]
    return None


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
    """Derive the timedelta represented by the table name."""
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
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _start_date_for_history(
    latest_date: datetime.date, config: RuntimeConfig
) -> datetime.date:
    """Choose a start date that covers the longest alert period plus weekend slack."""
    max_period = max(config.alert_change_over_time.keys(), default=datetime.timedelta())
    buffer_days = 3  # cover weekends/holidays
    return latest_date - max_period - datetime.timedelta(days=buffer_days)


def _period_metrics(
    args: tuple[
        datetime.timedelta,
        str,
        dict[datetime.date, float],
        list[tuple[datetime.datetime, float]],
        list[datetime.date],
        datetime.date,
        float,
        int,
    ],
) -> tuple[str, float | None, float | None]:
    """Compute volatility and delta for a single period (for parallel execution)."""
    (
        period,
        table_name,
        history,
        samples,
        sorted_dates,
        latest_date,
        latest_rate,
        vol_min_points,
    ) = args

    latest_ts = _date_to_timestamp(latest_date)
    cutoff = latest_ts - period
    window_samples = [s for s in samples if cutoff <= s[0] <= latest_ts]
    volatility = _sigma_per_hour(window_samples, vol_min_points)

    baseline_date = _baseline_date(sorted_dates, latest_date, period)
    delta_percent: float | None = None
    if baseline_date is not None:
        baseline_rate = history.get(baseline_date)
        if baseline_rate not in (None, 0):
            delta_percent = ((latest_rate - baseline_rate) / baseline_rate) * 100

    return table_name, volatility, delta_percent


def compute_metrics_parallel(
    history: dict[datetime.date, float],
    latest_date: datetime.date,
    latest_rate: float,
    config: RuntimeConfig,
    executor: ProcessPoolExecutor,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Calculate volatilities and deltas concurrently across all periods."""
    samples = _history_samples(history)
    sorted_dates = sorted(history.keys())
    tasks = (
        (
            period,
            table_name,
            history,
            samples,
            sorted_dates,
            latest_date,
            latest_rate,
            config.vol_min_points,
        )
        for period, table_name in config.table_names.items()
    )

    volatilities: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for table_name, volatility, delta_percent in executor.map(_period_metrics, tasks):
        volatilities[table_name] = volatility
        deltas[table_name] = delta_percent
    return volatilities, deltas


def main() -> None:
    """Fetch, evaluate, and alert on FX moves."""
    app_config = load_app_config(CONFIG_FILE)
    runtime_config = build_runtime_config(app_config)

    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(LOG_FILE, app_config.logging.max_bytes, app_config.logging.backups)

    base = runtime_config.base_currency
    quote = runtime_config.quote_currency
    pair_label = f"{base}→{quote}"
    logging.info("Starting %s alert run.", pair_label)

    if not DEV_MODE and datetime.date.today().weekday() >= 5:
        logging.warning(
            "Today is a weekend; ECB does not publish rates. Exiting early."
        )
        return

    with ProcessPoolExecutor() as executor:
        latest = executor.submit(
            wait_for_fresh_rate, base, quote, dev_mode=DEV_MODE
        ).result()
        if latest is None:
            logging.warning(
                "Exiting without alerts because today's rate was not published."
            )
            return
        latest_date, latest_rate = latest

        start_date = _start_date_for_history(latest_date, runtime_config)
        history = fetch_timeseries(base, quote, start_date, latest_date)
        if not history:
            logging.warning(
                "No historical data returned from Frankfurter; using only the latest rate."
            )
        history[latest_date] = latest_rate  # ensure latest sample is present
        logging.info(
            "Loaded %s data points from %s to %s.",
            len(history),
            min(history).isoformat(),
            max(history).isoformat(),
        )

        volatilities, deltas = compute_metrics_parallel(
            history, latest_date, latest_rate, runtime_config, executor
        )
        logging.info("Calculated volatilities per period: %s", volatilities)
        logging.info("Calculated delta percents per period: %s", deltas)
        breaches = filter_tables_over_thresholds(deltas, volatilities, runtime_config)
        logging.info("Tables exceeding thresholds: %s", breaches)
        if breaches:
            message_text = format_alert_message(
                breaches, volatilities, runtime_config, pair_label
            )
            send_telegram_message(message_text, ENV.bot_token, ENV.chat_id)
            logging.info("Alert sent to Telegram.")
        else:
            logging.info("No alerts triggered.")


def setup_logging(log_file: pathlib.Path, max_bytes: int, backups: int):
    """Set up logging with rotating file handler and console output."""
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backups
    )
    file_handler.setFormatter(formatter)
    rich_handler = RichHandler(
        markup=False,
        rich_tracebacks=False,
        show_time=True,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, rich_handler],
        format="%(message)s",
    )


if __name__ == "__main__":
    main()
