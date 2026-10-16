"""Resolve completed and actually observed XNYS daily sessions without live calendar calls."""

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo('America/New_York')


@dataclass(frozen=True)
class MarketDateResolution:
    selected_date: str
    latest_completed_date: str | None
    expected_date: str | None
    source: str
    confidence: str
    reason: str
    generated_at: str
    calendar_version: str
    publication_eligible: bool = False
    calendar_name: str = 'XNYS'


def _utc_now(now):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    return now.astimezone(timezone.utc)


def _get_calendar():
    import exchange_calendars

    return exchange_calendars.get_calendar('XNYS')


def _calendar_version():
    try:
        return version('exchange-calendars')
    except PackageNotFoundError:
        return 'unavailable'


def _iso_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('date must use YYYY-MM-DD format')
    return date.fromisoformat(value)


def explicit_resolution(value, now=None):
    """Record an authoritative user bound without claiming observed availability."""
    _iso_date(value)
    return MarketDateResolution(
        value, None, None, 'explicit', 'explicit', 'user_requested', _utc_now(now).isoformat(), _calendar_version()
    )


def resolve_market_date(fetch_observations=None, *, now=None, source='provider', grace_minutes=30):
    """Select max observed completed session and record confidence/freshness separately.

    ``fetch_observations(start, end)`` returns daily ``date``/``close`` dicts.
    It can be a provider wrapper or a cache-only reader. Failed/malformed data
    returns an explicitly unverified candidate, never fabricated availability.
    """
    now = _utc_now(now)
    if not math.isfinite(grace_minutes) or grace_minutes < 0:
        raise ValueError('delivery grace must be finite and nonnegative')
    local = now.astimezone(NY)
    today = local.date()
    cal_version = _calendar_version()
    try:
        calendar = _get_calendar()
        start = today - timedelta(days=45)
        if start < calendar.first_session.date() or today > calendar.last_session.date():
            raise ValueError('date outside supported calendar horizon')
        schedule = calendar.schedule.loc[start.isoformat() : today.isoformat()]
        closes = pd.to_datetime(schedule['close'], utc=True)
        completed = closes[closes <= now]
        delivered = closes[(closes + pd.Timedelta(minutes=grace_minutes)) <= now]
        latest = completed.index[-1].date()
        expected = delivered.index[-1].date() if len(delivered) else None
    except Exception:
        # Explicitly only a weekday estimate: it may be a holiday/early-close day.
        candidate = today if local.hour >= 16 else today - timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return MarketDateResolution(
            candidate.isoformat(),
            None,
            None,
            'weekday_estimate',
            'weekday_only',
            'calendar_unavailable',
            now.isoformat(),
            cal_version,
        )

    def fallback(reason):
        return MarketDateResolution(
            latest.isoformat(),
            latest.isoformat(),
            expected.isoformat() if expected else None,
            'exchange_calendar',
            'calendar_only',
            reason,
            now.isoformat(),
            cal_version,
        )

    lookup_start = latest - timedelta(days=14)
    try:
        rows = fetch_observations(lookup_start.isoformat(), latest.isoformat()) if fetch_observations else None
    except Exception:
        return fallback('lookup_failed')
    try:
        if not isinstance(rows, list):
            raise ValueError('observation payload must be a list')
        observed = []
        completed_dates = set(completed.index.date)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('observation must be a dict')
            day = _iso_date(row.get('date'))
            close = row.get('close')
            if isinstance(close, bool) or not math.isfinite(float(close)) or float(close) <= 0:
                raise ValueError('observation close must be finite and positive')
            if lookup_start <= day <= latest and day in completed_dates:
                observed.append(day)
    except (ValueError, TypeError, OverflowError):
        return fallback('invalid_observations')
    if not observed:
        return fallback('no_completed_observation')
    selected = max(observed)
    eligible = expected is not None and selected >= expected
    return MarketDateResolution(
        selected.isoformat(),
        latest.isoformat(),
        expected.isoformat() if expected else None,
        source,
        'observed',
        'observed' if eligible else 'observation_stale',
        now.isoformat(),
        cal_version,
        publication_eligible=eligible,
    )
