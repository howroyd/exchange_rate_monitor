#!.venv/bin python3
"""GBP→EUR alerts with static and volatility-aware thresholds."""

import datetime
import json
import logging
import math
import pathlib
import statistics
from logging.handlers import RotatingFileHandler
from typing import Any

import pydantic
import pydantic_settings
import requests


class Settings(pydantic_settings.BaseSettings):
    """Environment configuration for Telegram and FX source."""

    bot_token: str
    chat_id: int
    fx_url: pydantic.HttpUrl


ENV = Settings()
ALERT_CHANGE_OVER_TIME = {
    # Tuned to catch meaningful but not extreme moves for GBP→EUR
    datetime.timedelta(hours=1): 0.2,
    datetime.timedelta(hours=6): 0.5,
    datetime.timedelta(days=1): 1.0,
}
DATA_FILE = pathlib.Path(__file__).parent / "gbp_last_rates.json"
LOG_FILE = pathlib.Path(__file__).parent / "gbp_alert.log"
LOG_MAX_BYTES = 1_000_000  # ~1MB per file
LOG_BACKUPS = 5
MIN_ALERT_THRESHOLD = 0.1
VOL_MULTIPLIER = 1.75
VOL_LOOKBACK = datetime.timedelta(days=30)
VOL_MIN_POINTS = 6
logger = logging.getLogger("gbp_alert")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
# ---------------------------


class Datafile(pydantic.BaseModel):
    """Stored exchange rate sample."""

    rate: float
    source: pydantic.HttpUrl
    ts: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)


# ---------------------------


class Message(pydantic.BaseModel):
    """Telegram message content for alerts."""

    new_rate: float
    movement: float = pydantic.Field(
        default_factory=lambda data: data["new_rate"] - data["old_rate"]
    )
    direction: str = pydantic.Field(
        default_factory=lambda data: "up 📈" if data["movement"] > 0 else "down 📉"
    )

    def __str__(self):
        return (
            f"*GBP→EUR ALERT*\n\n"
            f"Movement: *{self.movement:.2f}* {self.direction}\n"
            f"New rate: `{self.new_rate:.6f}`\n\n"
            f"Check Wise if you want to act."
        )


# ---------------------------


def get_current_rate(url: pydantic.HttpUrl) -> Datafile:
    """Fetch the current GBP→EUR rate from the given API URL."""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return Datafile(rate=float(data["rates"]["EUR"]), source=url)


def load_last_rates(fn: pathlib.Path) -> list[Datafile]:
    """Load stored rates from disk, sorted by timestamp."""
    if not fn.exists():
        return []
    as_json = json.loads(fn.read_text())
    return sorted(
        [
            Datafile(**data_dict, ts=datetime.datetime.fromisoformat(ts))
            for ts, data_dict in as_json.items()
        ],
        key=lambda r: r.ts,
    )


def clean_rates(rates: list[Datafile]) -> list[Datafile]:
    """Drop samples older than the longest configured lookback."""
    oldest_delta_to_keep = max(max(ALERT_CHANGE_OVER_TIME.keys()), VOL_LOOKBACK)
    return [r for r in rates if r.ts >= datetime.datetime.now() - oldest_delta_to_keep]


def save_rates(fn: pathlib.Path, rates: list[Datafile]):
    """Persist rate samples to disk as JSON."""
    as_dict = {
        r.ts.isoformat(): json.loads(r.model_dump_json(exclude={"ts"})) for r in rates
    }
    fn.write_text(json.dumps(as_dict, indent=2))


def calculate_delta_percents(
    new_rate: Datafile, last_rates: list[Datafile]
) -> dict[datetime.timedelta, float]:
    """Compute percent changes versus the oldest rate within each window."""
    if not last_rates:
        return {}
    deltas = {}
    now = datetime.datetime.now()
    for delta in ALERT_CHANGE_OVER_TIME.keys():
        cutoff = now - delta
        relevant_rates = [r for r in last_rates if r.ts >= cutoff]
        if relevant_rates:
            oldest_rate = relevant_rates[0]
            percent_change = (
                (new_rate.rate - oldest_rate.rate) / oldest_rate.rate
            ) * 100
            deltas[delta] = percent_change
    return deltas


def calculate_hourly_volatility(last_rates: list[Datafile]) -> float | None:
    """Estimate hourly volatility (percent per sqrt hour) from stored samples."""
    cutoff = datetime.datetime.now() - VOL_LOOKBACK
    window = [r for r in sorted(last_rates, key=lambda r: r.ts) if r.ts >= cutoff]
    if len(window) < VOL_MIN_POINTS:
        return None

    variance_terms = []
    prev = window[0]
    for rate in window[1:]:
        delta_hours = (rate.ts - prev.ts).total_seconds() / 3600
        if delta_hours <= 0:
            prev = rate
            continue
        log_return = math.log(rate.rate / prev.rate)
        variance_terms.append((log_return**2) / delta_hours)
        prev = rate

    if len(variance_terms) < max(3, VOL_MIN_POINTS - 1):
        return None

    # Convert to percent per sqrt-hour
    sigma_per_hour = math.sqrt(statistics.fmean(variance_terms)) * 100
    return sigma_per_hour


def threshold_for_delta(
    delta: datetime.timedelta, sigma_per_hour: float | None
) -> float:
    """Return threshold for a window based on static levels and rolling vol."""
    static_threshold = ALERT_CHANGE_OVER_TIME[delta]
    if sigma_per_hour is None:
        return static_threshold

    delta_hours = delta.total_seconds() / 3600
    vol_component = sigma_per_hour * math.sqrt(delta_hours) * VOL_MULTIPLIER
    return max(vol_component, MIN_ALERT_THRESHOLD, static_threshold)


def send_telegram_message(msg: Message) -> dict[str, Any]:
    """Send a Telegram message using the bot token and chat ID."""
    url = f"https://api.telegram.org/bot{ENV.bot_token}/sendMessage"
    payload = {"chat_id": ENV.chat_id, "text": str(msg), "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> None:
    """Fetch, evaluate, and alert on GBP→EUR moves."""
    try:
        new_rate = get_current_rate(ENV.fx_url)
        logger.info(f"Fetched new rate: {new_rate.rate} from {new_rate.source}")
    except Exception as e:
        logger.error(f"ERROR fetching rate: {e}")
        return

    last_rates = load_last_rates(DATA_FILE)

    hourly_vol = calculate_hourly_volatility(last_rates)
    if hourly_vol is None:
        logger.info("Rolling volatility unavailable; using static thresholds.")
    else:
        logger.info(f"Hourly vol estimate: {hourly_vol:.4f}% (per sqrt hour)")

    diff_percents = calculate_delta_percents(new_rate, last_rates)

    for delta, percent in diff_percents.items():
        threshold = threshold_for_delta(delta, hourly_vol)
        if abs(percent) >= threshold:
            logger.info(
                f"ALERT condition met for {delta}: {percent:.3f}% change (threshold: {threshold:.3f}%)"
            )
            message = Message(new_rate=new_rate.rate, movement=percent)
            send_telegram_message(message)
            logger.info("Alert sent to Telegram.")

    last_rates.append(new_rate)
    last_rates = clean_rates(last_rates)
    save_rates(DATA_FILE, last_rates)


if __name__ == "__main__":
    main()
