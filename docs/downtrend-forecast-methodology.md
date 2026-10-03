# Long-history breadth episode research

The English publication pages are `reports/downtrend_forecast.html` and
`reports/downtrend_forecast_high_price.html`. Both are dated
research snapshot through 2026-10-02, not an automatically refreshed forecast.
The daily workflow publishes this tracked file but does not rerun the study.

## Scope and evidence

The research cache contains fresh FMP price responses for 1,256 candidate current
and former constituents. 983 have usable positive adjusted prices, including all
504 current constituents. SPY has 8,477 observations from 1993-01-29 through
2026-10-02. Individual responses/cache files have hashes in the acquisition
manifest. No API credential is saved in report artifacts or request diagnostics.

Three models are kept separate:

1. `current_fixed`: current constituent cohort with the project's fixed
   denominator; unavailable prices/MAs count as not above the MA. This is the
   reference for compatibility, **not** a verified historical index cohort.
2. `current_eligible`: the same survivor cohort, using only observations with
   available prices and a complete 200-session stock MA in the denominator.
3. `historical_reconstructed`: reverse membership from the FMP change history,
   using available MA-eligible members. Missing delisted prices and 43 reverse
   history inconsistencies make this a sensitivity analysis, not reliable PIT
   evidence for older periods. Median eligibility is approximately 53% in 2000
   and 76% in 2008.

Historical breadth is recomputed entirely within each model; vintages are never
silently spliced. All stock prices and SPY use FMP `adjClose`, which FMP describes
as adjusted for splits and dividends. The pre-SPY period supplies stock-MA/EMA
warmup; returns use actual SPY dates from 1993. The first 200 SPY observations are
excluded as signal onsets. The first reference episode is 1994-03-30.

## Definitions

- Pink: the EMA200 breadth hysteresis trend is -1 and EMA8 breadth is below EMA200.
  Hysteresis changes only when the EMA200 daily difference exceeds +0.001 or
  falls below -0.001.
- Price low: the first minimum adjusted close in the pink interval, including
  onset and excluding the first non-pink observation. It is known retrospectively.
- Warning end: first non-pink observation, including that day's return.
- Durations: index difference in actual SPY sessions, and date difference for
  calendar days; onset is day zero. Onset-to-low plus low-to-end equals total
  duration for every episode. Means add; medians need not.
- Returns: `(endpoint_adjusted_close / onset_adjusted_close - 1) * 100`.
  Zero-decline episodes contribute zero days and zero price-low return.
- Bottom detection: replay the chart algorithm on daily prefixes. Never use a
  backdated marker as its detectable date. First qualifying in-episode pivot is
  followed until the next pink onset; later revised candidates still count.
- Paired differences: detection return minus price-low return within each matched
  episode, measured in percentage points. The difference of marginal medians is
  not the median paired difference. Missing detections are excluded with counts.

The publication removes rapid switching with the same fixed rule on both pages:
at least three pink segments, non-pink gaps no longer than five sessions, and a
total first-onset-to-final-exit span no longer than 40 sessions. Original episode
dates and outcomes remain unchanged; episodes are not merged. In the reference
model, 2015-03-11, 2015-03-26 and 2015-04-17 are excluded. The 2018 episode remains.
This rule uses subsequent regime changes and is a retrospective sensitivity
analysis, not a filter necessarily available at onset or session 11. Apply it
separately to each alternative universe model. Bottom-detection searches still
stop at the next original pink onset, including excluded episodes.

The publication distinguishes retained completed episodes (reference n=53,
original n=56) from those that remained pink through the current equivalent age
(retained n=44, original n=45). It also
shows 5/21/63/126/252-session forward windows; these are convenience horizons,
not optimized prediction intervals or the duration of an entire correction.
Only forward windows ending before the current episode's onset are included.
The separate fixed-126-session definition of the price low is a sensitivity check.

## Offline reproduction

Keep private raw data in `data/long_history_20261002/` (ignored by Git). The cache
must include current/historical constituent snapshots, `universe.json`,
`membership_audit.json`, SPY responses, compressed constituent prices, and the
acquisition manifest. The initial metadata snapshots were fetched on 2026-10-03;
current membership was anchored after the latest effective change of 2026-10-01.

```bash
python -m scripts.prepare_long_history --directory data/long_history_20261002
python -m scripts.publish_long_history_report \
  --directory data/long_history_20261002 --output-dir reports --exclude-chattering
python -m scripts.analyze_high_price_pink \
  --directory reports/long_current_fixed \
  --output reports/downtrend_forecast_high_price.html --exclude-chattering
```

For price reacquisition from the same archived metadata/universe:

```bash
python -m scripts.fetch_long_history \
  --directory data/long_history_20261002 --env-file .env
```

This fetcher is intentionally bounded to 1990-01-01 through 2026-10-02, resumes
successful caches, checks 5,000-row caps, limits request starts to approximately
5.5 per second, and retries transient failures once. It rejects invalid adjusted
prices rather than imputing them. Run new acquisitions in a new dated cache;
do not replace the original evidence snapshot.

API-free validation:

```bash
SKIP_API_TESTS=true FMP_API_KEY='' python -m pytest \
  tests/test_downtrend_forecast.py tests/test_long_history_preparation.py
```

The generated page is ready for review/publication via the normal repository
workflow. Creating it does not merge, deploy, or change live trading.
The statistics are exploratory event-study results, not execution backtests or
evidence that any proposed entry rule is profitable.
