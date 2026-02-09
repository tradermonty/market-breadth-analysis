import requests
import pandas as pd
import io
from scipy.signal import find_peaks
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pathlib
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fmp_data_fetcher import FMPDataFetcher  # NEW: FMP API client

# Load environment variables
load_dotenv()

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)
data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)

# Instantiate global FMP data fetcher (use 'demo' key if environment variable is not set to allow tests).
fmp_fetcher = FMPDataFetcher(api_key=os.getenv('FMP_API_KEY', 'demo'))

# Configure retry strategy for API calls
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def save_stock_data(data, filename):
    """Save stock data to CSV file"""
    print(f"\nSaving stock data:")
    print(f"Data type: {type(data)}")
    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print(f"Sample columns: {list(data.columns[:5])}")
    elif isinstance(data, pd.Series):
        print(f"Length: {len(data)}")
        print(f"Name: {data.name}")
    
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame with a single column
        df = pd.DataFrame(data, columns=['adjusted_close'])
        print(f"Converted to DataFrame with shape: {df.shape}")
        df.to_csv(data_dir / filename, index=True)
    else:
        # Save DataFrame as is
        data.to_csv(data_dir / filename, index=True)

def load_stock_data(filename):
    """Load stock data from CSV file"""
    try:
        print(f"\nLoading stock data from {filename}")
        data = pd.read_csv(data_dir / filename, index_col=0, parse_dates=True)
        print(f"Loaded data shape: {data.shape}")
        print(f"Sample columns: {list(data.columns[:5])}")
        
        # If it's a single column DataFrame, convert to Series
        if isinstance(data, pd.DataFrame) and len(data.columns) == 1:
            print("Converting single column DataFrame to Series")
            return data.iloc[:, 0]
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

def get_sp500_tickers_from_fmp():
    """Get S&P500 ticker list from FMP API"""
    try:
        tickers = fmp_fetcher.get_sp500_constituents()
        if tickers:
            print(f"Successfully fetched {len(tickers)} S&P500 tickers from FMP")
            return tickers
        else:
            print("Error: No tickers returned from FMP API")
            return []
    except Exception as e:
        print(f"Error fetching S&P500 tickers from FMP: {e}")
        return []

def convert_ticker_symbol(ticker):
    """Return ticker symbol unchanged (no conversion needed for FMP)"""
    return ticker  # FMP uses the standard ticker format

# NEW: Helper to fetch price series from FMP and return as Series named 'adjusted_close'

def fetch_price_data_fmp(symbol: str, from_date: str, to_date: str) -> pd.Series:
    """Fetch historical price data for a symbol from FMP and return a Series.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. 'AAPL')
    from_date : str
        Start date (YYYY-MM-DD)
    to_date : str
        End date (YYYY-MM-DD)

    Returns
    -------
    pd.Series
        Series indexed by date containing the adjusted closing prices.
    """
    data = fmp_fetcher.get_historical_price_data(symbol, from_date, to_date)
    if not data:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(data)
    if df.empty or 'date' not in df.columns:
        return pd.Series(dtype="float64")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Prefer adjusted close, then close
    price_col = None
    for col in ['adjClose', 'adjusted_close', 'close']:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        return pd.Series(dtype="float64")

    series = df[price_col].astype(float)
    series.name = 'adjusted_close'
    return series

def get_sp500_price_data(start_date, end_date, use_saved_data=False):
    """Get S&P 500 price data using FMP or saved data"""
    filename = 'sp500_price_data.csv'
    file_path = data_dir / filename

    # Calculate the actual start date (1 year before the specified start date)
    actual_start_date = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    # Optionally use cached data if it fully covers the required period
    if use_saved_data and file_path.exists() and file_path.stat().st_size > 0:
        saved_data = load_stock_data(filename)
        if saved_data is not None and not saved_data.empty:
            if pd.to_datetime(actual_start_date) >= saved_data.index.min() and pd.to_datetime(end_date) <= saved_data.index.max():
                return saved_data

    # Fetch fresh data from FMP
    price_series = fetch_price_data_fmp('SPY', actual_start_date, end_date)
    if not price_series.empty:
        save_stock_data(price_series, filename)
    return price_series

def get_multiple_stock_data(tickers, start_date, end_date, use_saved_data=False):
    """Get data for multiple stocks using FMP or saved data"""
    filename = 'sp500_all_stocks.csv'

    # Attempt to load cached data if requested
    if use_saved_data:
        saved_data = load_stock_data(filename)
        if saved_data is not None and not saved_data.empty:
            if pd.to_datetime(start_date) >= saved_data.index.min() and pd.to_datetime(end_date) <= saved_data.index.max():
                return saved_data

    actual_start_date = (pd.to_datetime(start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    all_series = []
    print("Fetching stock price data from FMP ...")
    for ticker in tqdm(tickers, desc="Stock data retrieval progress"):
        try:
            series = fetch_price_data_fmp(ticker, actual_start_date, end_date)
            if len(series) > 200:  # Require reasonable history length
                series.name = ticker
                all_series.append(series)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if all_series:
        combined_data = pd.concat(all_series, axis=1)
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()
        save_stock_data(combined_data, filename)
        return combined_data

    return pd.DataFrame()

# Calculate whether each stock is above the specified moving average
def calculate_above_ma(stock_data, window=200):
    """Calculate whether each stock is above the specified moving average"""
    print(f"\nCalculating moving averages:")
    print(f"Input data shape: {stock_data.shape}")
    print(f"Sample column names: {list(stock_data.columns[:5])}")
    print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
    
    # Calculate moving average
    ma_data = stock_data.rolling(window=window).mean()
    
    # Check if price is above moving average
    above_ma = stock_data > ma_data
    
    # Calculate and print statistics
    daily_percentages = above_ma.mean(axis=1)
    print(f"\nBreadth Index Statistics:")
    print(f"Mean: {daily_percentages.mean():.3f}")
    print(f"Max: {daily_percentages.max():.3f}")
    print(f"Min: {daily_percentages.min():.3f}")
    print(f"Number of stocks per day: {above_ma.sum(axis=1).mean():.1f}")
    
    return above_ma

# Calculate trend with hysteresis for slope
def calculate_trend_with_hysteresis(ma_series, threshold=0.001):
    trend = [0] * len(ma_series)  # List to store trends (1: uptrend, -1: downtrend, 0: flat)
    current_trend = 0  # Current trend (initial is flat)

    for i in range(1, len(ma_series)):
        diff = ma_series.iloc[i] - ma_series.iloc[i-1]  # Calculate slope of moving average

        if current_trend <= 0 and diff > threshold:  # Start uptrend (previous trend was down or flat)
            current_trend = 1
        elif current_trend >= 0 and diff < -threshold:  # Start downtrend (previous trend was up or flat)
            current_trend = -1

        trend[i] = current_trend  # Record current trend

    return trend

def get_last_trading_day(date):
    """Return the previous trading day for the given date."""

    # Start from the previous day
    last_day = date - timedelta(days=1)

    # Move back to the most recent weekday
    while last_day.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        last_day -= timedelta(days=1)

    return last_day.strftime('%Y-%m-%d')

def get_latest_market_date():
    """Get the latest available market date using FMP data"""
    try:
        today_dt = datetime.today()
        start_lookup = (today_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        end_lookup = today_dt.strftime('%Y-%m-%d')
        data = fmp_fetcher.get_historical_price_data('SPY', start_lookup, end_lookup)
        if not data:
            raise ValueError('No data retrieved from FMP')

        df = pd.DataFrame(data)
        if df.empty or 'date' not in df.columns:
            raise ValueError('Invalid data format from FMP')

        latest_date = pd.to_datetime(df['date'].iloc[0])  # FMP returns latest first

        # Adjust for weekends
        if latest_date.weekday() >= 5:
            latest_date = pd.to_datetime(get_last_trading_day(latest_date))

        return latest_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error getting latest market date from FMP: {e}")
        return datetime.today().strftime('%Y-%m-%d')

def extract_chart_data(above_ma_200, sp500_data, short_ma_period=10, start_date=None, end_date=None):
    """Extract chart data for CSV export"""
    # Ensure both datasets have the same date range
    common_dates = above_ma_200.index.intersection(sp500_data.index)
    if len(common_dates) == 0:
        raise ValueError("No common dates found between breadth data and S&P500 data")
    
    # Align both datasets to common dates
    above_ma_200 = above_ma_200.loc[common_dates]
    sp500_data = sp500_data.loc[common_dates]
    
    # Filter data by date range if specified
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (above_ma_200.index >= start_date) & (above_ma_200.index <= end_date)
        above_ma_200 = above_ma_200.loc[mask]
        sp500_data = sp500_data.loc[mask]
    
    # Calculate percentage of stocks above 200-day moving average
    breadth_index_200 = above_ma_200.mean(axis=1)

    # Calculate 200-day and short-term moving averages for Breadth Index
    breadth_ma_200 = breadth_index_200.ewm(span=200, adjust=False).mean()
    breadth_ma_short = breadth_index_200.ewm(span=short_ma_period, adjust=False).mean()

    # Calculate 200MA slope using hysteresis
    breadth_ma_200_trend = calculate_trend_with_hysteresis(breadth_ma_200, threshold=0.001)
    breadth_ma_200_trend = pd.Series(breadth_ma_200_trend, index=breadth_ma_200.index)

    # Detect peaks (tops) and troughs (bottoms)
    peaks, _ = find_peaks(breadth_ma_200, distance=50, prominence=0.015)
    troughs, _ = find_peaks(-breadth_ma_200, distance=50, prominence=0.015)

    # Extract data where short-term moving average is below 0.4
    below_04 = breadth_ma_short[breadth_ma_short < 0.4]
    
    # Detect troughs for data below 0.4 using find_peaks
    troughs_below_04, _ = find_peaks(-below_04, prominence=0.02)

    # Calculate average values for peaks and troughs
    peaks_avg = breadth_ma_200.iloc[peaks].mean()
    troughs_avg_below_04 = below_04.iloc[troughs_below_04].mean()
    
    # Return all calculated data
    return {
        'breadth_index_200': breadth_index_200,
        'breadth_ma_200': breadth_ma_200,
        'breadth_ma_short': breadth_ma_short,
        'breadth_ma_200_trend': breadth_ma_200_trend,
        'sp500_data': sp500_data,
        'peaks': peaks,
        'troughs': troughs,
        'troughs_below_04': troughs_below_04,
        'below_04': below_04,
        'peaks_avg': peaks_avg,
        'troughs_avg_below_04': troughs_avg_below_04
    }

def detect_bearish_regions(breadth_ma_200_trend, breadth_ma_short, breadth_ma_200):
    """Convert day-level bearish mask into a list of (start, end) continuous intervals.

    A day is bearish when breadth_ma_200_trend == -1 AND breadth_ma_short < breadth_ma_200.

    Returns
    -------
    list[tuple[Timestamp, Timestamp]]
        Each tuple is (start_date, end_date) of a contiguous bearish region.
    """
    bearish_mask = (breadth_ma_200_trend == -1) & (breadth_ma_short < breadth_ma_200)

    regions = []
    in_bearish = False
    start = None

    for i in range(len(bearish_mask)):
        if bearish_mask.iloc[i] and not in_bearish:
            start = breadth_ma_short.index[i]
            in_bearish = True
        elif not bearish_mask.iloc[i] and in_bearish:
            regions.append((start, breadth_ma_short.index[i - 1]))
            in_bearish = False

    if in_bearish:
        regions.append((start, breadth_ma_short.index[-1]))

    return regions

def plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, short_ma_period=10,
                                       start_date=None, end_date=None, output_dir='reports'):
    """Visualize Breadth Index and S&P 500 price using Plotly.

    Returns the Plotly Figure object for programmatic inspection / testing.
    """
    # Extract chart data
    chart_data = extract_chart_data(above_ma_200, sp500_data, short_ma_period, start_date, end_date)

    breadth_ma_200 = chart_data['breadth_ma_200']
    breadth_ma_short = chart_data['breadth_ma_short']
    breadth_ma_200_trend = chart_data['breadth_ma_200_trend']
    sp500_data = chart_data['sp500_data']
    peaks = chart_data['peaks']
    troughs = chart_data['troughs']
    troughs_below_04 = chart_data['troughs_below_04']
    below_04 = chart_data['below_04']
    peaks_avg = chart_data['peaks_avg']
    troughs_avg_below_04 = chart_data['troughs_avg_below_04']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=['S&P 500 Price',
                        f'S&P 500 Breadth Index with 200-Day MA and {short_ma_period}-Day MA'],
    )

    # --- Panel 1: S&P 500 Price ---
    fig.add_trace(
        go.Scatter(
            x=sp500_data.index,
            y=sp500_data.values,
            name='S&P 500 Price',
            line=dict(color='#00FFFF', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Trough markers on S&P 500 (panel 1)
    if len(troughs_below_04) > 0:
        s_and_p_troughs = sp500_data.loc[below_04.index[troughs_below_04]]
        fig.add_trace(
            go.Scatter(
                x=s_and_p_troughs.index,
                y=s_and_p_troughs.values,
                name=f'Troughs ({short_ma_period}MA < 0.4) on S&P 500',
                mode='markers',
                marker=dict(color='#800080', size=12, symbol='triangle-down'),
            ),
            row=1, col=1,
        )

    # Y axis: log scale
    fig.update_yaxes(type='log', row=1, col=1, title_text='Price')

    # --- Panel 2: Breadth Index ---
    fig.add_trace(
        go.Scatter(
            x=breadth_ma_200.index,
            y=breadth_ma_200.values,
            name='Breadth Index (200-Day MA)',
            line=dict(color='#008000', width=2),
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=breadth_ma_short.index,
            y=breadth_ma_short.values,
            name=f'Breadth Index ({short_ma_period}-Day MA)',
            line=dict(color='#FFA500', width=2),
        ),
        row=2, col=1,
    )

    # Peak markers
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=breadth_ma_200.index[peaks],
                y=breadth_ma_200.iloc[peaks].values,
                name='Peaks (Tops)',
                mode='markers',
                marker=dict(color='#FF0000', size=10, symbol='triangle-up'),
            ),
            row=2, col=1,
        )

    # Trough markers
    if len(troughs) > 0:
        fig.add_trace(
            go.Scatter(
                x=breadth_ma_200.index[troughs],
                y=breadth_ma_200.iloc[troughs].values,
                name='Troughs (Bottoms)',
                mode='markers',
                marker=dict(color='#0000FF', size=10, symbol='triangle-down'),
            ),
            row=2, col=1,
        )

    # Troughs below 0.4 on breadth panel
    if len(troughs_below_04) > 0:
        fig.add_trace(
            go.Scatter(
                x=below_04.index[troughs_below_04],
                y=below_04.iloc[troughs_below_04].values,
                name=f'Troughs ({short_ma_period}MA < 0.4)',
                mode='markers',
                marker=dict(color='#800080', size=12, symbol='triangle-down'),
            ),
            row=2, col=1,
        )

    # Average Peaks horizontal line
    fig.add_hline(
        y=peaks_avg, row=2, col=1,
        line=dict(color='#FF0000', dash='dash', width=2),
        annotation_text=f'Avg Peaks = {peaks_avg:.2f}',
        annotation_position='bottom right',
    )

    # Average Troughs horizontal line
    fig.add_hline(
        y=troughs_avg_below_04, row=2, col=1,
        line=dict(color='#0000FF', dash='dash', width=2),
        annotation_text=f'Avg Troughs = {troughs_avg_below_04:.2f}',
        annotation_position='top right',
    )

    # Y axis range for breadth panel
    fig.update_yaxes(range=[0, 1], row=2, col=1, title_text='Breadth Index Percentage')

    # --- Bearish background (both panels) ---
    bearish_regions = detect_bearish_regions(breadth_ma_200_trend, breadth_ma_short, breadth_ma_200)
    for start, end in bearish_regions:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='rgba(255, 210, 240, 0.35)',
            line_width=0,
            row='all', col=1,
        )

    # --- Layout ---
    fig.update_layout(
        height=900,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=60, b=80),
    )

    # Grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')

    # Range selector on panel 1
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(count=3, label='3Y', step='year', stepmode='backward'),
                dict(count=5, label='5Y', step='year', stepmode='backward'),
                dict(step='all', label='ALL'),
            ],
            bgcolor='#f0f0f0',
            activecolor='#d0d0d0',
        ),
        row=1, col=1,
    )

    # --- Output ---
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    html_file = output_path / 'market_breadth.html'
    fig.write_html(
        str(html_file),
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'displaylogo': False,
        },
    )
    print(f"Interactive chart saved to {html_file}")

    # PNG output (backward compatibility)
    png_file = output_path / 'market_breadth.png'
    try:
        fig.write_image(str(png_file), width=1200, height=900, scale=2)
        print(f"PNG chart saved to {png_file}")
    except Exception as e:
        print(f"PNG export skipped (kaleido not installed): {e}")

    return fig

def get_stock_price_data(symbol, start_date, end_date, use_saved_data=False):
    """
    Get stock price data for a given symbol using FMP
    """
    # Actual start date (get data from 2 years before the specified start date)
    actual_start_date = (pd.to_datetime(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    filename = f'{symbol}_price_data.csv'

    # Use cached data if fully covers period
    if use_saved_data:
        saved_data = load_stock_data(filename)
        if saved_data is not None and not saved_data.empty:
            if pd.to_datetime(actual_start_date) >= saved_data.index.min() and pd.to_datetime(end_date) <= saved_data.index.max():
                mask = (saved_data.index >= pd.to_datetime(actual_start_date)) & (saved_data.index <= pd.to_datetime(end_date))
                return saved_data.loc[mask]

    # Fetch data from FMP
    series = fetch_price_data_fmp(symbol, actual_start_date, end_date)
    if not series.empty:
        save_stock_data(series, filename)
    return series

def export_chart_data_to_csv(chart_data, short_ma_period, filename=None):
    """Export chart data to CSV file"""
    if filename is None:
        current_date = datetime.now().strftime('%Y%m%d')
        filename = f'market_breadth_data_{current_date}_ma{short_ma_period}.csv'
    
    csv_path = reports_dir / filename
    
    # Create main DataFrame with time series data
    df = pd.DataFrame(index=chart_data['breadth_index_200'].index)
    df['Date'] = df.index
    df['S&P500_Price'] = chart_data['sp500_data']
    df['Breadth_Index_Raw'] = chart_data['breadth_index_200']
    df['Breadth_Index_200MA'] = chart_data['breadth_ma_200']
    df[f'Breadth_Index_{short_ma_period}MA'] = chart_data['breadth_ma_short']
    df['Breadth_200MA_Trend'] = chart_data['breadth_ma_200_trend']
    
    # Add peak and trough markers
    df['Is_Peak'] = False
    df['Is_Trough'] = False
    df[f'Is_Trough_{short_ma_period}MA_Below_04'] = False
    
    if len(chart_data['peaks']) > 0:
        df.iloc[chart_data['peaks'], df.columns.get_loc('Is_Peak')] = True
    
    if len(chart_data['troughs']) > 0:
        df.iloc[chart_data['troughs'], df.columns.get_loc('Is_Trough')] = True
    
    if len(chart_data['troughs_below_04']) > 0:
        below_04_indices = chart_data['below_04'].index
        for trough_idx in chart_data['troughs_below_04']:
            trough_date = below_04_indices[trough_idx]
            if trough_date in df.index:
                df.loc[trough_date, f'Is_Trough_{short_ma_period}MA_Below_04'] = True
    
    # Add bearish signal indicator
    df['Bearish_Signal'] = ((chart_data['breadth_ma_200_trend'] == -1) & 
                           (chart_data['breadth_ma_short'] < chart_data['breadth_ma_200']))
    
    # Reorder columns
    column_order = [
        'Date', 'S&P500_Price', 'Breadth_Index_Raw', 'Breadth_Index_200MA', 
        f'Breadth_Index_{short_ma_period}MA', 'Breadth_200MA_Trend', 'Bearish_Signal',
        'Is_Peak', 'Is_Trough', f'Is_Trough_{short_ma_period}MA_Below_04'
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Chart data exported to {csv_path}")
    
    # Create summary data
    summary_data = {
        'Metric': [
            f'Average Peaks (200MA)',
            f'Average Troughs ({short_ma_period}MA < 0.4)',
            'Total Peaks Count',
            'Total Troughs Count',
            f'Total Troughs ({short_ma_period}MA < 0.4) Count',
            'Analysis Period Start',
            'Analysis Period End',
            'Total Trading Days'
        ],
        'Value': [
            f"{chart_data['peaks_avg']:.3f}",
            f"{chart_data['troughs_avg_below_04']:.3f}",
            len(chart_data['peaks']),
            len(chart_data['troughs']),
            len(chart_data['troughs_below_04']),
            chart_data['breadth_index_200'].index.min().strftime('%Y-%m-%d'),
            chart_data['breadth_index_200'].index.max().strftime('%Y-%m-%d'),
            len(chart_data['breadth_index_200'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'market_breadth_summary_{datetime.now().strftime("%Y%m%d")}_ma{short_ma_period}.csv'
    summary_path = reports_dir / summary_filename
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary data exported to {summary_path}")
    
    return csv_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Market Breadth Analysis')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD format). If not specified, today\'s date will be used.')
    parser.add_argument('--short_ma', type=int, default=8, choices=[5, 8, 10, 20], help='Short-term moving average period (5, 8, 10, or 20)')
    parser.add_argument('--use_saved_data', action='store_true', help='Use saved data instead of fetching from FMP')
    parser.add_argument('--export_csv', action='store_true', help='Export chart data to CSV files')

    # Set up command line arguments
    args = parser.parse_args()

    if args.debug:
        print("Debug mode enabled")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Environment variables:")
        print(f"FMP_API_KEY set: {'FMP_API_KEY' in os.environ}")
        print(f"Available directories:")
        print(f"reports/ exists: {os.path.exists('reports')}")
        print(f"data/ exists: {os.path.exists('data')}")


    # Set start date
    if args.start_date:
        start_date = args.start_date
    else:
        # Default is 10 years ago from today
        start_date = (datetime.today() - timedelta(days=3650)).strftime("%Y-%m-%d")
    
    # Set end date
    if args.end_date:
        end_date = args.end_date
    else:
        # Default is today
        end_date = get_last_trading_day(datetime.today())  # Use the previous business day from today
    
    print(f"\nInitial start date: {start_date}")
    print(f"Initial end date: {end_date}")
    
    # Convert dates to datetime for comparison
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Ensure end_date is not before start_date
    if end_date_dt < start_date_dt:
        print(f"Warning: Latest market date ({end_date}) is before start date ({start_date})")
        print("Using start date as end date")
        end_date = start_date
    
    print(f"Final analysis period: {start_date} to {end_date}")
    print(f"Short-term moving average period: {args.short_ma} days")

    try:
        # Get S&P500 list from FMP API
        ticker_list = get_sp500_tickers_from_fmp()

        if ticker_list:
            # Get S&P 500 price data
            sp500_data = get_sp500_price_data(start_date, end_date, args.use_saved_data)
            
            if sp500_data.empty:
                raise ValueError("Failed to retrieve S&P500 price data")

            # Get and process data
            stock_data = get_multiple_stock_data(ticker_list, start_date, end_date, args.use_saved_data)
            
            if stock_data.empty:
                raise ValueError("Failed to retrieve stock data")

            # Calculate 200-day moving average
            above_ma_200 = calculate_above_ma(stock_data, window=200)
            
            if above_ma_200.empty:
                raise ValueError("Failed to calculate moving averages")

            # Ensure both datasets have the same date range
            common_dates = above_ma_200.index.intersection(sp500_data.index)
            if len(common_dates) == 0:
                raise ValueError("No common dates found between breadth data and S&P500 data")
            
            # Align both datasets to common dates
            above_ma_200 = above_ma_200.loc[common_dates]
            sp500_data = sp500_data.loc[common_dates]

            # Print data shapes for debugging
            print(f"\nData shapes after alignment:")
            print(f"above_ma_200 shape: {above_ma_200.shape}")
            print(f"sp500_data shape: {sp500_data.shape}")
            print(f"Number of common dates: {len(common_dates)}")

            # Visualize Breadth Index and S&P 500 price with specified date range
            plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, args.short_ma, start_date, end_date)
            
            # Export to CSV if requested
            if args.export_csv:
                chart_data = extract_chart_data(above_ma_200, sp500_data, args.short_ma, start_date, end_date)
                export_chart_data_to_csv(chart_data, args.short_ma)
        else:
            print("Error: Ticker list could not be retrieved.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --use_saved_data option to use previously saved data.")

if __name__ == '__main__':
    main()
