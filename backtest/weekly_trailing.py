"""Weekly trailing stop utilities for the backtest engine.

Provides functions to aggregate daily OHLC data to weekly bars,
detect week boundaries, and check weekly trailing stop conditions.
"""

import pandas as pd


def aggregate_to_weekly(price_df):
    """Aggregate daily OHLC data to weekly bars using W-FRI frequency.

    Trading week runs Saturday through Friday. Uses adjusted OHLC columns
    when available, falls back to adjusted_close only.

    Parameters
    ----------
    price_df : pd.DataFrame
        Daily price data with DatetimeIndex. Expected columns:
        adjusted_open, adjusted_high, adjusted_low, adjusted_close.
        If only adjusted_close is present, derives OHLC from it.

    Returns
    -------
    pd.DataFrame
        Weekly OHLC with columns: open, high, low, close.
        Index is the Friday ending date for each week.
    """
    has_full_ohlc = all(
        col in price_df.columns for col in ('adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close')
    )

    if has_full_ohlc:
        weekly = price_df.resample('W-FRI').agg(
            {
                'adjusted_open': 'first',
                'adjusted_high': 'max',
                'adjusted_low': 'min',
                'adjusted_close': 'last',
            }
        )
        weekly.columns = ['open', 'high', 'low', 'close']
    else:
        weekly = (
            price_df['adjusted_close']
            .resample('W-FRI')
            .agg(
                open='first',
                high='max',
                low='min',
                close='last',
            )
        )

    weekly = weekly.dropna(how='all')
    return weekly


def is_week_end(index, i):
    """Check whether bar *i* is the last bar of its trading week.

    A bar is a week-end if it is the final bar in the index, or if the
    next bar belongs to a different W-FRI period.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The daily bar index.
    i : int
        Position in the index to check.

    Returns
    -------
    bool
    """
    if i >= len(index) - 1:
        return True
    current_period = pd.Timestamp(index[i]).to_period('W-FRI')
    next_period = pd.Timestamp(index[i + 1]).to_period('W-FRI')
    return current_period != next_period


def check_weekly_trailing_stop(
    current_close,
    weekly_df,
    entry_date,
    current_date,
    trailing_type='weekly_ema',
    ema_period=10,
    nweek_low_period=4,
    transition_weeks=3,
):
    """Check whether the weekly trailing stop condition is met.

    Parameters
    ----------
    current_close : float
        Current daily closing price.
    weekly_df : pd.DataFrame
        Weekly OHLC data (output of ``aggregate_to_weekly``).
    entry_date : pd.Timestamp or str
        Date of the earliest open position entry.
    current_date : pd.Timestamp or str
        Current bar date.
    trailing_type : str
        'weekly_ema' or 'weekly_nweek_low'.
    ema_period : int
        EMA span for weekly_ema type.
    nweek_low_period : int
        Number of prior weeks for nweek_low type.
    transition_weeks : int
        Minimum weeks after entry before trailing stop activates.

    Returns
    -------
    bool
        True if the trailing stop condition is triggered.
    """
    # Transition weeks guard
    current_p = pd.Timestamp(current_date).to_period('W-FRI')
    entry_p = pd.Timestamp(entry_date).to_period('W-FRI')
    weeks_elapsed = current_p.ordinal - entry_p.ordinal
    if weeks_elapsed < transition_weeks:
        return False

    # Slice weekly data up to (and including) current week.
    # Use the W-FRI period end date so that holiday-shortened weeks
    # (e.g., Thursday close when Friday is a holiday) still include the
    # current week's bar whose label is the Friday date.
    current_week_end = current_p.end_time.normalize()
    relevant_weekly = weekly_df[weekly_df.index <= current_week_end]
    if relevant_weekly.empty:
        return False

    if trailing_type == 'weekly_ema':
        ema = relevant_weekly['close'].ewm(span=ema_period, adjust=False).mean()
        return current_close < ema.iloc[-1]

    elif trailing_type == 'weekly_nweek_low':
        # Exclude current week (close >= low would make this a dead condition)
        prior_weekly = relevant_weekly.iloc[:-1]
        if len(prior_weekly) < 1:
            return False
        tail = prior_weekly.tail(nweek_low_period)
        nweek_low = tail['low'].min()
        return current_close < nweek_low

    return False
