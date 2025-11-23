"""Configuration parsing and validation for FX alerts."""

import datetime
import pathlib
from dataclasses import dataclass

import pydantic
import tomllib


def parse_period_str(period_str: str) -> datetime.timedelta:
    """Convert shorthand period strings (e.g., 1h, 7d) into timedeltas."""
    if not period_str:
        msg = "Period string cannot be empty."
        raise ValueError(msg)
    unit = period_str[-1].lower()
    try:
        value = float(period_str[:-1])
    except ValueError as exc:  # pragma: no cover - defensive
        msg = f"Invalid period value: {period_str}"
        raise ValueError(msg) from exc

    if unit == "s":
        return datetime.timedelta(seconds=value)
    if unit == "m":
        return datetime.timedelta(minutes=value)
    if unit == "h":
        return datetime.timedelta(hours=value)
    if unit == "d":
        return datetime.timedelta(days=value)

    msg = f"Unsupported period unit in {period_str}. Use s, m, h, or d."
    raise ValueError(msg)


class AlertPeriod(pydantic.BaseModel):
    """Configured alert period and threshold."""

    period: str
    threshold: float

    @pydantic.field_validator("period")
    @classmethod
    def validate_period(cls, value: str) -> str:
        parse_period_str(value)
        return value


class CurrencySettings(pydantic.BaseModel):
    """Currency pair to monitor."""

    base: str
    quote: str

    @pydantic.field_validator("base", "quote")
    @classmethod
    def validate_code(cls, value: str) -> str:
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            msg = "Currency codes must be 3-letter alphabetic strings."
            raise ValueError(msg)
        return code


class AlertSettings(pydantic.BaseModel):
    """Alert thresholds and volatility parameters."""

    min_alert_threshold: float = 0.1
    vol_multiplier: float = 1.75
    vol_min_points: int = 6
    periods: list[AlertPeriod]


class LoggingSettings(pydantic.BaseModel):
    """Logging configuration."""

    max_bytes: int = 1_000_000
    backups: int = 5


class AppConfig(pydantic.BaseModel):
    """Application configuration loaded from TOML."""

    currencies: CurrencySettings
    alerts: AlertSettings
    logging: LoggingSettings = LoggingSettings()


@dataclass(frozen=True)
class RuntimeConfig:
    """Derived configuration used at runtime."""

    alert_change_over_time: dict[datetime.timedelta, float]
    table_names: dict[datetime.timedelta, str]
    table_periods: dict[str, datetime.timedelta]
    min_alert_threshold: float
    vol_multiplier: float
    vol_min_points: int
    base_currency: str
    quote_currency: str


def load_app_config(config_path: pathlib.Path) -> AppConfig:
    """Load and validate configuration from TOML."""
    with config_path.open("rb") as file:
        return AppConfig.model_validate(tomllib.load(file))


def build_runtime_config(app_config: AppConfig) -> RuntimeConfig:
    """Build derived runtime configuration from app config."""
    alert_change_over_time = {
        parse_period_str(period.period): period.threshold
        for period in app_config.alerts.periods
    }
    table_names = {
        period: str(int(period.total_seconds())) for period in alert_change_over_time
    }
    table_periods = {name: period for period, name in table_names.items()}
    return RuntimeConfig(
        alert_change_over_time=alert_change_over_time,
        table_names=table_names,
        table_periods=table_periods,
        min_alert_threshold=app_config.alerts.min_alert_threshold,
        vol_multiplier=app_config.alerts.vol_multiplier,
        vol_min_points=app_config.alerts.vol_min_points,
        base_currency=app_config.currencies.base,
        quote_currency=app_config.currencies.quote,
    )
