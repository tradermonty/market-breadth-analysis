import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pathlib
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import platform
from matplotlib.patches import Patch
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

def setup_matplotlib_backend():
    """Set up matplotlib backend based on the operating system"""
    system = platform.system().lower()
    if system == 'darwin':  # macOS
        matplotlib.use('TkAgg')  # Use TkAgg backend for macOS
    elif system == 'windows':
        matplotlib.use('TkAgg')  # Use TkAgg backend for Windows
    else:  # Linux and others
        matplotlib.use('Agg')  # Use Agg backend for other systems

# Set up backend first
setup_matplotlib_backend()

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

def get_sp500_tickers_from_wikipedia():
    """Get S&P500 ticker list from Wikipedia"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P500 tickers from Wikipedia: {e}")
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

def plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, short_ma_period=10, start_date=None, end_date=None):
    """Visualize Breadth Index and S&P 500 price"""
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

    # Create plot with larger figure size and adjusted font sizes
    plt.rcParams['font.size'] = 12  # Increase base font size
    plt.rcParams['axes.titlesize'] = 16  # Increase title font size
    plt.rcParams['axes.labelsize'] = 14  # Increase axis label font size
    plt.rcParams['xtick.labelsize'] = 10  # Set x-axis tick label font size
    plt.rcParams['ytick.labelsize'] = 10  # Set y-axis tick label font size
    plt.rcParams['legend.fontsize'] = 10  # Set legend font size

    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)  # Create larger figure size

    # Plot S&P 500 price first
    axs[0].plot(sp500_data.index, sp500_data, label='S&P 500 Price', color='cyan', zorder=2, linewidth=2)
    
    # Create a custom patch for the background color legend
    background_patch = Patch(facecolor=(1.0, 0.9, 0.96), alpha=0.5, 
                           label='Bearish Signal (MA200 Down & Short MA < MA200)')
    
    # Add background color
    for i in range(len(breadth_ma_short) - 1):
        if (breadth_ma_200_trend.iloc[i] == -1) and (breadth_ma_short.iloc[i] < breadth_ma_200.iloc[i]):
            axs[0].axvspan(breadth_ma_short.index[i], breadth_ma_short.index[i + 1], 
                          color=(1.0, 0.9, 0.96), alpha=0.3, zorder=1)
            axs[1].axvspan(breadth_ma_short.index[i], breadth_ma_short.index[i + 1], 
                          color=(1.0, 0.9, 0.96), alpha=0.3, zorder=1)

    axs[0].set_title('S&P 500 Price', pad=20, fontsize=16)  # Set title font size directly
    axs[0].set_xlabel('Date', fontsize=14)  # Set x-axis label font size directly
    axs[0].set_ylabel('Price', fontsize=14)  # Set y-axis label font size directly
    axs[0].set_yscale('log')
    
    # Configure logarithmic axis format with larger font sizes
    major_ticks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    axs[0].yaxis.set_major_locator(FixedLocator(major_ticks))
    axs[0].yaxis.set_major_formatter(FixedFormatter([str(x) for x in major_ticks]))
    
    # Configure minor ticks
    minor_ticks = []
    for major in major_ticks[:-1]:
        next_major = major_ticks[major_ticks.index(major) + 1]
        step = (next_major - major) / 5
        for i in range(1, 5):
            minor_ticks.append(major + step * i)
    
    axs[0].yaxis.set_minor_locator(FixedLocator(minor_ticks))
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    
    # Add marks on S&P500 at the same timing with larger markers
    if len(troughs_below_04) > 0:
        s_and_p_troughs = sp500_data.loc[below_04.index[troughs_below_04]]
        axs[0].scatter(s_and_p_troughs.index, s_and_p_troughs, color='purple', marker='v', s=150, 
                      label=f'Troughs ({short_ma_period}MA < 0.4) on S&P 500', zorder=3)
    
    # Add custom legend with the background patch and larger markers
    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(1, background_patch)
    axs[0].legend(handles=handles, loc='center left', bbox_to_anchor=(0.02, 0.5))
    
    axs[0].grid(True)

    # Plot Breadth Index with thicker lines
    axs[1].plot(breadth_ma_200.index, breadth_ma_200, label='Breadth Index (200-Day MA)', color='green', zorder=2, linewidth=2)
    axs[1].plot(breadth_ma_short.index, breadth_ma_short, label=f'Breadth Index ({short_ma_period}-Day MA)', color='orange', zorder=2, linewidth=2)
    
    # Add all markers with higher zorder and larger size
    if len(peaks) > 0:
        axs[1].plot(breadth_ma_200.index[peaks], breadth_ma_200.iloc[peaks], 'r^', label='Peaks (Tops)', zorder=3, markersize=10)
    if len(troughs) > 0:
        axs[1].plot(breadth_ma_200.index[troughs], breadth_ma_200.iloc[troughs], 'bv', label='Troughs (Bottoms)', zorder=3, markersize=10)
    if len(troughs_below_04) > 0:
        axs[1].scatter(below_04.index[troughs_below_04], below_04.iloc[troughs_below_04], color='purple', marker='v', s=150,
                      label=f'Troughs ({short_ma_period}MA < 0.4)', zorder=3)

    # Draw horizontal lines for peak and trough averages with thicker lines
    axs[1].axhline(peaks_avg, color='red', linestyle='--', label=f'Average Peaks (200MA) = {peaks_avg:.2f}', zorder=2, linewidth=2)
    axs[1].axhline(troughs_avg_below_04, color='blue', linestyle='--', label=f'Average Troughs ({short_ma_period}MA < 0.4) = {troughs_avg_below_04:.2f}', zorder=2, linewidth=2)

    axs[1].set_title(f'S&P 500 Breadth Index with 200-Day MA and {short_ma_period}-Day MA', pad=20, fontsize=16)  # Set title font size directly
    axs[1].set_ylabel('Breadth Index Percentage', fontsize=14)  # Set y-axis label font size directly
    axs[1].legend(loc='center left', bbox_to_anchor=(0.02, 0.5))
    axs[1].grid(True)

    plt.tight_layout()
    
    # Get current date and add to filename
    filename = 'reports/market_breadth.png'
    
    # Save image with higher DPI and better quality settings
    try:
        plt.savefig(filename, dpi=400, bbox_inches='tight', format='png', facecolor='white', edgecolor='none')
        print(f"Chart saved to {filename}")
    except Exception as e:
        print(f"Error saving chart: {e}")
        # Try alternative save method
        try:
            plt.savefig(filename, dpi=400, bbox_inches='tight', format='png')
            print(f"Chart saved to {filename} (alternative method)")
        except Exception as e:
            print(f"Failed to save chart: {e}")
    
    # Display the graph
    plt.show()
    
    plt.close()  # Close the figure to free memory

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

def main():
    parser = argparse.ArgumentParser(description='Market Breadth Analysis')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD format). If not specified, today\'s date will be used.')
    parser.add_argument('--short_ma', type=int, default=8, choices=[5, 8, 10, 20], help='Short-term moving average period (5, 8, 10, or 20)')
    parser.add_argument('--use_saved_data', action='store_true', help='Use saved data instead of fetching from FMP')

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
        # Get S&P500 list from Wikipedia
        ticker_list = get_sp500_tickers_from_wikipedia()

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
        else:
            print("Error: Ticker list could not be retrieved.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --use_saved_data option to use previously saved data.")

if __name__ == '__main__':
    main()
