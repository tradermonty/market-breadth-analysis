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

# Load environment variables
load_dotenv()

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)
data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)

def setup_matplotlib_backend():
    """Set up matplotlib backend based on the operating system"""
    system = platform.system().lower()
    if system == 'darwin':  # macOS
        matplotlib.use('TkAgg')
    elif system == 'windows':
        matplotlib.use('Qt5Agg')  # Qt5Agg is generally stable on Windows
    else:  # Linux and others
        matplotlib.use('Agg')  # Non-interactive backend for server environments

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
    data.to_csv(data_dir / filename)

def load_stock_data(filename):
    """Load stock data from CSV file"""
    try:
        return pd.read_csv(data_dir / filename, index_col=0, parse_dates=True)
    except FileNotFoundError:
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
    """Convert special ticker symbols for EODHD API"""
    # Convert any ticker containing dots to use hyphens instead
    if '.' in ticker:
        return ticker.replace('.', '-')
    return ticker

def get_sp500_price_data(start_date, end_date, use_saved_data=False):
    """Get S&P 500 price data using EODHD or saved data"""
    filename = 'sp500_price_data.csv'
    file_path = data_dir / filename
    
    # Check if file exists and is not empty
    if use_saved_data and file_path.exists() and file_path.stat().st_size > 0:
        try:
            saved_data = load_stock_data(filename)
            if saved_data is not None and not saved_data.empty:
                return saved_data
        except Exception as e:
            print(f"Error loading saved data: {e}")
    
    # Try to load local data
    saved_data = load_stock_data(filename)
    
    if saved_data is not None and not saved_data.empty:
        # Check the date range of saved data
        saved_start = saved_data.index.min()
        saved_end = saved_data.index.max()
        
        # Verify if the required period is covered by saved data
        if pd.to_datetime(start_date) >= saved_start and pd.to_datetime(end_date) <= saved_end:
            return saved_data
        
        # Identify missing periods
        missing_periods = []
        if pd.to_datetime(start_date) < saved_start:
            missing_periods.append((start_date, saved_start.strftime('%Y-%m-%d')))
        if pd.to_datetime(end_date) > saved_end:
            missing_periods.append((saved_end.strftime('%Y-%m-%d'), end_date))
        
        # Fetch data for missing periods
        new_data = pd.Series()
        for period_start, period_end in missing_periods:
            try:
                url = f'https://eodhd.com/api/eod/SPY.US'
                params = {
                    'from': period_start,
                    'to': period_end,
                    'api_token': os.getenv('EODHD_API_KEY'),
                    'fmt': 'json'
                }
                
                response = session.get(url, params=params)
                if response.status_code == 200:
                    period_data = pd.DataFrame(response.json())
                    if not period_data.empty:
                        period_data['date'] = pd.to_datetime(period_data['date'])
                        period_data.set_index('date', inplace=True)
                        if 'adjusted_close' in period_data.columns:
                            new_data = pd.concat([new_data, period_data['adjusted_close']])
            except Exception as e:
                print(f"Error fetching data for period {period_start} to {period_end}: {e}")
        
        # Combine new data with saved data
        if not new_data.empty:
            combined_data = pd.concat([saved_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            save_stock_data(combined_data, filename)
            return combined_data
        
        return saved_data
    
    # Fetch new data if no saved data exists
    try:
        url = f'https://eodhd.com/api/eod/SPY.US'
        params = {
            'from': start_date,
            'to': end_date,
            'api_token': os.getenv('EODHD_API_KEY'),
            'fmt': 'json'
        }
        
        response = session.get(url, params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                
                if 'adjusted_close' in data.columns:
                    save_stock_data(data['adjusted_close'], filename)
                    return data['adjusted_close']
    except Exception as e:
        print(f"Error fetching S&P500 data: {e}")
    
    return pd.Series()

def get_multiple_stock_data(tickers, start_date, end_date, use_saved_data=False):
    """Get data for multiple stocks using EODHD or saved data"""
    filename = 'stock_data.csv'
    file_path = data_dir / filename
    
    # Check if file exists and is not empty
    if use_saved_data and file_path.exists() and file_path.stat().st_size > 0:
        try:
            saved_data = load_stock_data(filename)
            if saved_data is not None and not saved_data.empty:
                return saved_data
        except Exception as e:
            print(f"Error loading saved data: {e}")
    
    # Try to load local data
    saved_data = load_stock_data(filename)
    
    if saved_data is not None and not saved_data.empty:
        # Check the date range of saved data
        saved_start = saved_data.index.min()
        saved_end = saved_data.index.max()
        
        # Convert dates to datetime for comparison
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Verify if the required period is covered by saved data
        if start_date_dt >= saved_start and end_date_dt <= saved_end:
            return saved_data
        
        # Identify missing periods
        missing_periods = []
        if start_date_dt < saved_start:
            missing_periods.append((start_date, saved_start.strftime('%Y-%m-%d')))
        if end_date_dt > saved_end:
            # Check if the gap is small enough to ignore (less than 2 days)
            gap_days = (end_date_dt - saved_end).days
            if gap_days > 2:
                missing_periods.append((saved_end.strftime('%Y-%m-%d'), end_date))
            else:
                print(f"Ignoring small gap of {gap_days} days after saved data")
        
        if not missing_periods:
            return saved_data
            
        # Fetch data for missing periods
        new_data_list = []
        print("Fetching missing stock price data...")
        
        # Get list of existing tickers in saved data
        existing_tickers = saved_data.columns.tolist()
        print(f"\nExisting tickers in saved data: {len(existing_tickers)}")
        print(f"Sample of existing tickers: {existing_tickers[:5]}")
        
        for ticker in tqdm(tickers, desc="Stock data retrieval progress"):
            try:
                eodhd_ticker = convert_ticker_symbol(ticker)
                
                for period_start, period_end in missing_periods:
                    url = f'https://eodhd.com/api/eod/{eodhd_ticker}.US'
                    params = {
                        'from': period_start,
                        'to': period_end,
                        'api_token': os.getenv('EODHD_API_KEY'),
                        'fmt': 'json'
                    }
                    
                    response = session.get(url, params=params)
                    if response.status_code == 200 and response.text.strip():
                        try:
                            data = pd.DataFrame(response.json())
                            if not data.empty:
                                data['date'] = pd.to_datetime(data['date'])
                                data.set_index('date', inplace=True)
                                
                                if 'adjusted_close' in data.columns and len(data) > 0:
                                    # Remove any duplicate indices before adding to list
                                    data = data[~data.index.duplicated(keep='last')]
                                    new_data_list.append(data['adjusted_close'])
                        except ValueError as e:
                            print(f"\nError parsing data for {ticker}: {str(e)}")
            except Exception as e:
                print(f"\nError processing {ticker}: {str(e)}")
                continue
        
        # Combine new data with saved data
        if new_data_list:
            print(f"\nNumber of new data series: {len(new_data_list)}")
            new_data = pd.concat(new_data_list, axis=1)
            print(f"Shape of new_data: {new_data.shape}")
            print(f"Sample of new_data columns: {new_data.columns[:5]}")
            
            # Remove any duplicate indices from saved_data
            saved_data = saved_data[~saved_data.index.duplicated(keep='last')]
            print(f"Shape of saved_data: {saved_data.shape}")
            
            # Get common tickers between saved_data and new_data
            common_tickers = list(set(saved_data.columns) & set(new_data.columns))
            print(f"\nNumber of common tickers: {len(common_tickers)}")
            print(f"Sample of common tickers: {common_tickers[:5]}")
            
            if common_tickers:
                # Combine data only for common tickers
                print("\nAttempting to combine data...")
                print(f"Saved data shape before combination: {saved_data[common_tickers].shape}")
                print(f"New data shape before combination: {new_data[common_tickers].shape}")
                
                combined_data = pd.concat([
                    saved_data[common_tickers],
                    new_data[common_tickers]
                ])
                print(f"Combined data shape: {combined_data.shape}")
                
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                save_stock_data(combined_data, filename)
                return combined_data
            else:
                print("Warning: No common tickers found between saved and new data")
                return saved_data
        
        return saved_data
    
    # Fetch new data if no saved data exists
    all_data = []
    print("Fetching stock price data...")
    
    for ticker in tqdm(tickers, desc="Stock data retrieval progress"):
        try:
            eodhd_ticker = convert_ticker_symbol(ticker)
            
            url = f'https://eodhd.com/api/eod/{eodhd_ticker}.US'
            params = {
                'from': start_date,
                'to': end_date,
                'api_token': os.getenv('EODHD_API_KEY'),
                'fmt': 'json'
            }
            
            response = session.get(url, params=params)
            if response.status_code == 200 and response.text.strip():
                try:
                    data = pd.DataFrame(response.json())
                    if not data.empty:
                        data['date'] = pd.to_datetime(data['date'])
                        data.set_index('date', inplace=True)
                        
                        if 'adjusted_close' in data.columns and len(data) > 200:
                            # Remove any duplicate indices before adding to list
                            data = data[~data.index.duplicated(keep='last')]
                            all_data.append(data['adjusted_close'])
                        else:
                            print(f"\nSkipping {ticker}: Insufficient data")
                except ValueError as e:
                    print(f"\nError parsing data for {ticker}: {str(e)}")
            else:
                print(f"\nSkipping {ticker}: No data available")
        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            continue
    
    if all_data:
        print(f"\nRetrieved data for {len(all_data)} stocks.")
        combined_data = pd.concat(all_data, axis=1)
        # Remove any duplicate indices
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        save_stock_data(combined_data, filename)
        return combined_data
    
    return pd.DataFrame()

# Calculate whether each stock is above the specified moving average
def calculate_above_ma(stock_data, window=200):
    ma_data = stock_data.rolling(window=window).mean()  # Calculate moving average
    above_ma = stock_data > ma_data  # Check if price is above moving average
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

def get_latest_market_date():
    """Get the latest available market date"""
    try:
        # Use SPY as a reference to get the latest market date
        url = f'https://eodhd.com/api/eod/SPY.US'
        params = {
            'api_token': os.getenv('EODHD_API_KEY'),
            'fmt': 'json',
            'limit': 1
        }
        
        response = session.get(url, params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if not data.empty:
                latest_date = pd.to_datetime(data['date'].iloc[0])
                # Verify that the date is not in the future
                if latest_date > datetime.today():
                    print("Warning: API returned a future date, using yesterday's date instead")
                    yesterday = datetime.today() - timedelta(days=1)
                    if yesterday.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                        yesterday = yesterday - timedelta(days=yesterday.weekday() - 4)
                    return yesterday.strftime('%Y-%m-%d')
                return latest_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error getting latest market date: {e}")
    
    # If API call fails or returns invalid date, use yesterday's date
    yesterday = datetime.today() - timedelta(days=1)
    # If yesterday was a weekend, go back to Friday
    if yesterday.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        yesterday = yesterday - timedelta(days=yesterday.weekday() - 4)
    return yesterday.strftime('%Y-%m-%d')

def plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, short_ma_period=20):
    """Visualize Breadth Index and S&P 500 price"""
    # Ensure both datasets have the same date range
    common_dates = above_ma_200.index.intersection(sp500_data.index)
    if len(common_dates) == 0:
        raise ValueError("No common dates found between breadth data and S&P500 data")
    
    above_ma_200 = above_ma_200.loc[common_dates]
    sp500_data = sp500_data.loc[common_dates]
    
    # Calculate percentage of stocks above 200-day moving average
    breadth_index_200 = above_ma_200.mean(axis=1)

    # Calculate 200-day and short-term moving averages for Breadth Index
    breadth_ma_200 = breadth_index_200.rolling(window=200).mean()
    breadth_ma_short = breadth_index_200.rolling(window=short_ma_period).mean()

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

    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot S&P 500 price first
    axs[0].plot(sp500_data.index, sp500_data['adjusted_close'], label='S&P 500 Price', color='cyan', zorder=2)
    
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

    axs[0].set_title('S&P 500 Price')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].set_yscale('log')
    
    # Configure logarithmic axis format
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
    
    # Add marks on S&P500 at the same timing
    if len(troughs_below_04) > 0:
        s_and_p_troughs = sp500_data['adjusted_close'].loc[below_04.index[troughs_below_04]]
        axs[0].scatter(s_and_p_troughs.index, s_and_p_troughs, color='purple', marker='v', s=100, 
                      label=f'Troughs ({short_ma_period}MA < 0.4) on S&P 500', zorder=3)
    
    # Add custom legend with the background patch
    handles, labels = axs[0].get_legend_handles_labels()
    handles.insert(1, background_patch)
    axs[0].legend(handles=handles, loc='upper left')
    
    axs[0].grid(True)

    # Plot Breadth Index
    axs[1].plot(breadth_ma_200.index, breadth_ma_200, label='Breadth Index (200-Day MA)', color='green', zorder=2)
    axs[1].plot(breadth_ma_short.index, breadth_ma_short, label=f'Breadth Index ({short_ma_period}-Day MA)', color='orange', zorder=2)
    
    # Add all markers with higher zorder
    if len(peaks) > 0:
        axs[1].plot(breadth_ma_200.index[peaks], breadth_ma_200.iloc[peaks], 'r^', label='Peaks (Tops)', zorder=3)
    if len(troughs) > 0:
        axs[1].plot(breadth_ma_200.index[troughs], breadth_ma_200.iloc[troughs], 'bv', label='Troughs (Bottoms)', zorder=3)
    if len(troughs_below_04) > 0:
        axs[1].scatter(below_04.index[troughs_below_04], below_04.iloc[troughs_below_04], color='purple', marker='v', s=100,
                      label=f'Troughs ({short_ma_period}MA < 0.4)', zorder=3)

    # Draw horizontal lines for peak and trough averages
    axs[1].axhline(peaks_avg, color='red', linestyle='--', label=f'Average Peaks (200MA) = {peaks_avg:.2f}', zorder=2)
    axs[1].axhline(troughs_avg_below_04, color='blue', linestyle='--', label=f'Average Troughs ({short_ma_period}MA < 0.4) = {troughs_avg_below_04:.2f}', zorder=2)

    axs[1].set_title(f'S&P 500 Breadth Index with 200-Day MA and {short_ma_period}-Day MA')
    axs[1].set_ylabel('Breadth Index Percentage')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    
    # Get current date and add to filename
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f'reports/market_breadth_{current_date}_ma{short_ma_period}.png'
    
    # Save image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {filename}")
    
    plt.show()

# Main process
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze S&P500 market breadth')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--short_ma', type=int, default=20, choices=[10, 20], help='Short-term moving average period (10 or 20)')
    parser.add_argument('--use_saved_data', action='store_true', help='Use saved data instead of fetching from EODHD')
    args = parser.parse_args()

    # Set start date
    if args.start_date:
        start_date = args.start_date
    else:
        # Default is 10 years ago from today
        start_date = (datetime.today() - timedelta(days=3650)).strftime("%Y-%m-%d")
    
    print(f"\nInitial start date: {start_date}")
    
    # Get the latest available market date
    end_date = get_latest_market_date()
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

            # Get and process data
            stock_data = get_multiple_stock_data(ticker_list, start_date, end_date, args.use_saved_data)

            # Calculate 200-day moving average
            above_ma_200 = calculate_above_ma(stock_data, window=200)

            # Visualize Breadth Index and S&P 500 price
            plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, args.short_ma)
        else:
            print("Error: Ticker list could not be retrieved.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --use_saved_data option to use previously saved data.")
