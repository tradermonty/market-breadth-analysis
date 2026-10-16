# Analysis end-date resolution

Without `--end_date`, analysis selects the maximum actually observed SPY daily-price date whose XNYS session has closed. Row order does not matter. Current partial bars, future dates, weekends and non-session dates cannot be promoted to a fabricated observation. SPY lookup uses the existing FMP wrapper, bounded to the 14 days ending at the latest completed session; malformed or unavailable observations remain unverified.

The offline [`exchange_calendars` XNYS schedule](https://github.com/gerrymanoim/exchange_calendars) supplies session dates and UTC closes, including exchange holidays, early closes and DST. It is a maintained library snapshot, not a real-time closure/halt service. A clock passed to the helpers must be timezone-aware; runtime defaults to UTC, independent of the host timezone.

## Availability and expected freshness

`--data-arrival-grace-minutes` (default 30; finite and nonnegative) controls when a completed session is *expected* to have been delivered. It never delays use of an already supplied completed close.

- Friday at 15:59 New York: Friday is not completed; choose the latest observed earlier session.
- Friday at 16:00: use Friday immediately if the provider supplies it.
- Friday at 16:10, provider still only has Thursday: Thursday is acceptable during the default delivery grace.
- Friday at 16:30, provider still only has Thursday: Thursday remains the actual available research bound, but is marked stale and not publication-eligible.
- On holidays/weekends, expected freshness follows scheduled sessions rather than calendar age; early closes use their scheduled close time.

The helpers retain `get_latest_market_date()` / `resolve_default_end_date(use_saved_data, now=None)` as string-returning convenience APIs. Call `resolve_market_end_date()` to inspect confidence. When data lookup fails, the string can be an unverified calendar candidate; it is never proof of provider availability. Calendar failures/out-of-range yield a clearly labelled weekday estimate, which can fall on a holiday. The previous-weekday helper `get_last_trading_day()` retains its old behavior and is not an exchange calendar.

## Saved and explicit bounds

Saved-data *date resolution* reads only `data/sp500_price_data.csv`, requiring a valid unique daily date index and one positive finite price column. It makes no FMP request. Missing/corrupt cache is unverified. This does not make the rest of `--use_saved_data` strictly offline: constituents/acquisition can still use their existing API fallback paths, tracked in Issue #14.

`--end_date YYYY-MM-DD` is authoritative and bypasses default date lookup. It is validated but never silently clamped to another day (including weekends). `end_date < start_date` is an error. Explicit dates record user intent, not observed availability; metadata does not automatically certify them for publication.

## Metadata and failure contract

`reports/market_date_resolution.json` records `selected_date`, `latest_completed_date`, `expected_date`, `source`, `confidence`, `reason`, `generated_at` (UTC), `calendar_name`, `calendar_version`, and `publication_eligible`.

Sources distinguish FMP/cache observations, exchange-calendar candidates, weekday estimates and explicit requests. Only observed dates meeting expected freshness are publication-eligible. A default run with unverified fallback records the reason and fails with nonzero status before constituent acquisition or output generation. Verified stale data can still be analyzed for research, with `publication_eligible=false`.

These date checks do not validate breadth coverage or the produced CSV/HTML content. Issue #9 owns the deployment gate consuming this metadata; this change alone does not prove a valid publication or update the deployed site. Do not deploy tracked/leftover reports after a failed analysis. Reason codes contain no raw provider exceptions or credentials. No current-main inclusion is claimed until the implementation PR merges.
