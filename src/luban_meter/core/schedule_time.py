"""Calendar time for one-shot and five-field cron schedules."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from tzlocal import get_localzone

from luban_meter.log import get_logger

logger = get_logger(__name__)


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(
        timespec="microseconds"
    )


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def display_time(value, zone):
    if not value:
        return "-"
    return datetime.fromisoformat(value).astimezone(
        timezone_for(zone)
    ).isoformat(timespec="seconds")


def timezone_for(name):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        logger.exception("Schedule timezone resolution failed: %s", name)
        raise ValueError(f"unknown timezone: {name}") from exc


def next_cron(expression, zone, after):
    if len(expression.split()) != 5:
        raise ValueError("cron must contain five fields (minute to weekday)")
    base = datetime.fromtimestamp(after, timezone_for(zone))
    return stamp(croniter(expression, base).get_next(datetime).timestamp())


def timing(at, cron, zone, now):
    if bool(at) == bool(cron):
        raise ValueError("specify exactly one of --at and --cron")
    zone = zone or get_localzone().key
    tz = timezone_for(zone)
    if cron:
        due = next_cron(cron, zone, now)
    else:
        value = datetime.fromisoformat(at)
        if value.tzinfo is None:
            value = value.replace(tzinfo=tz)
            restored = datetime.fromtimestamp(value.timestamp(), tz)
            if restored.replace(tzinfo=None) != value.replace(tzinfo=None):
                raise ValueError("--at is a nonexistent local time")
            if value.utcoffset() != value.replace(fold=1).utcoffset():
                raise ValueError("ambiguous --at requires a UTC offset")
        if value.timestamp() <= now:
            raise ValueError("--at must be in the future")
        due = stamp(value.timestamp())
    return {"at": due if at else "", "cron": cron or "",
            "timezone": zone, "next_at": due}
