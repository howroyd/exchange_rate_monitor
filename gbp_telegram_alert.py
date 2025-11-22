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
from tinydb import where
from tinydb.table import Table


class Settings(pydantic_settings.BaseSettings):
    """Environment configuration for Telegram and FX source."""

    bot_token: str
    chat_id: int
    fx_url: pydantic.HttpUrl


ENV = Settings(_env_file=".env")
ALERT_CHANGE_OVER_TIME = {
    # Tuned to catch meaningful but not extreme moves for GBP→EUR
    datetime.timedelta(hours=1): 0.2,
    datetime.timedelta(hours=6): 0.5,
    datetime.timedelta(days=1): 1.0,
    datetime.timedelta(days=7): 2.5,
    datetime.timedelta(days=14): 3.5,
    datetime.timedelta(days=30): 5.0,
}
TABLE_NAMES = {
    period: str(int(period.total_seconds())) for period in ALERT_CHANGE_OVER_TIME
}
TABLE_PERIODS = {name: period for period, name in TABLE_NAMES.items()}
DATA_FILE = pathlib.Path(__file__).parent / "data" / "gbp_last_rates.json"
DB_FILE = DATA_FILE.parent / "gbp_alert_db.json"
LOG_FILE = pathlib.Path(__file__).parent / "logs" / "gbp_alert.log"
DATA_FILE.parent.mkdir(exist_ok=True, parents=True)
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)
LOG_MAX_BYTES = 1_000_000  # ~1MB per file
LOG_BACKUPS = 5
MIN_ALERT_THRESHOLD = 0.1
VOL_MULTIPLIER = 1.75
VOL_MIN_POINTS = 6
# ---------------------------


class Datafile(pydantic.BaseModel):
    """Stored exchange rate sample."""

    rate: float
    source: pydantic.HttpUrl
    ts: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC)
    )


# ---------------------------


class AlertMessage(pydantic.BaseModel):
    """Telegram message content summarizing threshold breaches."""

    breaches: dict[str, dict[str, float]]
    volatilities: dict[str, float | None]

    def __str__(self) -> str:
        lines: list[str] = ["*GBP→EUR ALERTS*", ""]
        for table_name, hits in sorted(self.breaches.items()):
            period = _period_from_table_name(table_name)
            if period is None or period not in ALERT_CHANGE_OVER_TIME:
                continue
            delta_percent = hits.get("static", hits.get("dynamic"))
            if delta_percent is None:
                continue
            static_threshold, dynamic_threshold = _thresholds_for_period(
                period, self.volatilities.get(table_name)
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


def _sigma_per_hour(samples: list[tuple[datetime.datetime, float]]) -> float | None:
    """Return volatility (percent per sqrt hour) from timestamp/rate samples."""
    if len(samples) < VOL_MIN_POINTS:
        return None

    terms = []
    for (tsa, ra), (tsb, rb) in itertools.pairwise(samples):
        delta_hours = (tsb - tsa).total_seconds() / 3600
        if delta_hours > 0:
            terms.append((math.log(rb / ra) ** 2) / delta_hours)

    if len(terms) < max(3, VOL_MIN_POINTS - 1):
        return None

    return math.sqrt(statistics.fmean(terms)) * 100


def _period_from_table_name(table_name: str) -> datetime.timedelta | None:
    """Derive the timedelta represented by the TinyDB table name."""
    return TABLE_PERIODS.get(table_name)


def _table_name_for_period(period: datetime.timedelta) -> str:
    """Return the TinyDB table name for a timedelta."""
    return TABLE_NAMES.get(period, str(int(period.total_seconds())))


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


def _sorted_samples(table: Table) -> list[tuple[datetime.datetime, float]]:
    """Return table rows as sorted (timestamp, rate) tuples."""
    return sorted(
        (
            datetime.datetime.fromisoformat(entry["ts"]).replace(tzinfo=datetime.UTC),
            float(entry["rate"]),
        )
        for entry in table.all()
    )


def _thresholds_for_period(
    period: datetime.timedelta, sigma_per_hour: float | None
) -> tuple[float, float | None]:
    """Return static and dynamic thresholds for a period."""
    static_threshold = max(MIN_ALERT_THRESHOLD, ALERT_CHANGE_OVER_TIME[period])
    if sigma_per_hour is None:
        return static_threshold, None

    delta_hours = period.total_seconds() / 3600
    dynamic_threshold = max(
        MIN_ALERT_THRESHOLD, sigma_per_hour * math.sqrt(delta_hours) * VOL_MULTIPLIER
    )
    return static_threshold, dynamic_threshold


def filter_tables_over_thresholds(
    deltas: dict[str, float | None], volatilities: dict[str, float | None]
) -> dict[str, dict[str, float]]:
    """Return tables whose percent change exceeds static and/or dynamic thresholds."""
    breaches: dict[str, dict[str, float]] = {}
    for table_name, delta_percent in deltas.items():
        if delta_percent is None:
            continue
        period = _period_from_table_name(table_name)
        if period is None or period not in ALERT_CHANGE_OVER_TIME:
            continue
        sigma_per_hour = volatilities.get(table_name)
        static_threshold, dynamic_threshold = _thresholds_for_period(
            period, sigma_per_hour
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


def trim_table(
    table: Table, period: datetime.timedelta, *, reset_item_ids: bool = False
):
    """Remove entries older than the specified period from the TinyDB table."""
    cutoff = datetime.datetime.now(datetime.UTC) - period

    for entry in table.all():
        ts_str = entry["ts"]
        ts = datetime.datetime.fromisoformat(ts_str).replace(tzinfo=datetime.UTC)
        if ts < cutoff:
            table.remove(where("ts") == ts_str)
            logging.info(f"Removed old entry {ts_str} from {period} table.")

    if reset_item_ids:
        all_entries = table.all()
        table.truncate()
        for entry in sorted(all_entries, key=lambda e: e["ts"]):
            table.insert(dict(entry.items()))
        table_name = _table_name_for_period(period)
        logging.info("Reset item IDs in %s table.", table_name)


def put_new_rate_into_db(
    db: tinydb.TinyDB, new_rate: Datafile, *, force: bool = False
) -> list[str]:
    """Persist the new rate into each period table if enough time has passed."""
    tables_updated = []

    for period in ALERT_CHANGE_OVER_TIME:
        table_name = _table_name_for_period(period)
        table = db.table(table_name)
        trim_table(table, period)

        entries = table.all()
        if entries and not force:
            last_ts = max(
                datetime.datetime.fromisoformat(entry["ts"]).replace(
                    tzinfo=datetime.UTC
                )
                for entry in entries
            )
            if (new_rate.ts - last_ts) < period:
                continue

        data = new_rate.model_dump(mode="json")
        _item_id = table.insert(data)
        trim_table(table, period, reset_item_ids=True if _item_id > 1000 else False)
        tables_updated.append(table_name)

    return tables_updated


def calculate_volatilities_per_period(
    db: tinydb.TinyDB, table_names: list[str]
) -> dict[str, float | None]:
    """Calculate volatilities for each period table provided."""
    return {
        table_name: _sigma_per_hour(_sorted_samples(db.table(table_name)))
        for table_name in table_names
    }


def calculate_delta_percents(
    db: tinydb.TinyDB, table_names: list[str]
) -> dict[str, float | None]:
    """Compute percent change between oldest and newest entries per period table."""
    deltas: dict[str, float | None] = {}
    for table_name in table_names:
        samples = _sorted_samples(db.table(table_name))
        if len(samples) < 2:
            deltas[table_name] = None
            continue
        _, oldest_rate = samples[0]
        _, newest_rate = samples[-1]
        deltas[table_name] = ((newest_rate - oldest_rate) / oldest_rate) * 100
    return deltas


def main() -> None:
    """Fetch, evaluate, and alert on GBP→EUR moves."""
    with tinydb.TinyDB(DB_FILE) as db:
        new_rate = get_current_rate(ENV.fx_url)
        logging.info("Fetched new rate: %s from %s", new_rate.rate, new_rate.source)
        tables_updated = put_new_rate_into_db(db, new_rate)
        if not tables_updated:
            logging.info("No new entries added to any period table; exiting.")
            return
        volatilities = calculate_volatilities_per_period(db, tables_updated)
        logging.info("Calculated volatilities per period: %s", volatilities)
        deltas = calculate_delta_percents(db, tables_updated)
        logging.info("Calculated delta percents per period: %s", deltas)
        breaches = filter_tables_over_thresholds(deltas, volatilities)
        logging.info("Tables exceeding thresholds: %s", breaches)
        if breaches:
            message = AlertMessage(breaches=breaches, volatilities=volatilities)
            send_telegram_message(str(message), ENV.bot_token, ENV.chat_id)
            logging.info("Alert sent to Telegram.")


def setup_logging():
    """Set up logging with rotating file handler and console output."""
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


if __name__ == "__main__":
    setup_logging()
    main()
