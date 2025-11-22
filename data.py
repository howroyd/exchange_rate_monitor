"""Database interaction helpers for GBP→EUR alerts."""

import datetime
import logging

import pydantic
import tinydb
from tinydb import where
from tinydb.table import Table

from config import RuntimeConfig


class Datafile(pydantic.BaseModel):
    """Stored exchange rate sample."""

    rate: float
    source: pydantic.HttpUrl
    ts: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC)
    )


def sorted_samples(table: Table) -> list[tuple[datetime.datetime, float]]:
    """Return table rows as sorted (timestamp, rate) tuples."""
    return sorted(
        (
            datetime.datetime.fromisoformat(entry["ts"]).replace(tzinfo=datetime.UTC),
            float(entry["rate"]),
        )
        for entry in table.all()
    )


def trim_table(
    table: Table,
    period: datetime.timedelta,
    table_name: str,
    *,
    reset_item_ids: bool = False,
):
    """Remove entries older than the specified period from the TinyDB table."""
    cutoff = datetime.datetime.now(datetime.UTC) - period

    for entry in table.all():
        ts_str = entry["ts"]
        ts = datetime.datetime.fromisoformat(ts_str).replace(tzinfo=datetime.UTC)
        if ts < cutoff:
            table.remove(where("ts") == ts_str)
            logging.info("Removed old entry %s from %s table.", ts_str, table_name)

    if reset_item_ids:
        all_entries = table.all()
        table.truncate()
        for entry in sorted(all_entries, key=lambda e: e["ts"]):
            table.insert(dict(entry.items()))
        logging.info("Reset item IDs in %s table.", table_name)


def put_new_rate_into_db(
    db: tinydb.TinyDB,
    new_rate: Datafile,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> list[str]:
    """Persist the new rate into each period table if enough time has passed."""
    tables_updated: list[str] = []

    for period in config.alert_change_over_time:
        table_name = config.table_names.get(period, str(int(period.total_seconds())))
        table = db.table(table_name)
        trim_table(table, period, table_name)

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
        item_id = table.insert(data)
        trim_table(table, period, table_name, reset_item_ids=item_id > 1000)
        tables_updated.append(table_name)

    return tables_updated
