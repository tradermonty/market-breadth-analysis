# Market Breadth Analysis Tool

A tool for analyzing and visualizing the market breadth of S&P500 stocks.

## Features

- Fetch price data for all S&P500 stocks
- Calculate breadth indicators based on 200-day moving average
- Visualize breadth indicators and S&P500 price movements
- Identify trend reversal points through peak detection
- Save and reuse historical data without requiring an API key
- Backtest trading strategies based on market breadth signals
- Multi-ETF backtesting capabilities

## Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)
- API Key (Optional):
  - EODHD API Key (EOD Historical Data — All World plan or higher required)
    - Pricing details: https://eodhd.com/pricing
  - Not required if using saved data

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd market_breadth
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (Optional - only needed for fetching new data):
Create a `.env` file and set your API key:
```
EODHD_API_KEY=your_eodhd_api_key
```
Or copy the `.env.sample` file to `.env` and edit it:
```bash
cp .env.sample .env
```

## Usage

### Market Breadth Analysis

Basic usage (requires API key):
```bash
python market_breadth.py
```

Using saved data (no API key required):
```bash
python market_breadth.py --use_saved_data
```

With additional options:
```bash
python market_breadth.py --start_date 2020-01-01 --short_ma 20 --use_saved_data
```

### Backtesting

Single strategy backtesting:
```bash
python backtest/backtest.py
```

Multi-ETF backtesting:
```bash
python backtest/run_multi_etf_backtest.py
```

### Command Line Arguments

#### Market Breadth Analysis
- `--start_date`: Start date for analysis (YYYY-MM-DD format)
  - Default: 10 years ago from today
- `--short_ma`: Short-term moving average period (10 or 20)
- `--use_saved_data`: Use previously saved data instead of fetching from EODHD

#### Backtesting
- `--start_date`: Start date for backtesting
- `--end_date`: End date for backtesting
- `--initial_capital`: Initial capital for backtesting
- `--position_size`: Position size as percentage of capital
- `--stop_loss`: Stop loss percentage
- `--take_profit`: Take profit percentage

### Data Storage and Reuse

The tool now supports saving and reusing historical data:

- Data is stored in the `data/` directory:
  - `sp500_price_data.csv`: S&P 500 index price data
  - `stock_data.csv`: Individual stock price data for all S&P 500 companies

Workflow:
1. First-time use (requires API key):
   - Run without `--use_saved_data` to fetch and save data
   - Data is automatically saved for future use
2. Subsequent use:
   - Run with `--use_saved_data` to use previously saved data
   - No API key required
   - Faster execution as no API calls are made

Note: To update the saved data with fresh market data, run without `--use_saved_data` (requires API key).

### Data Source

EODHD (End of Day Historical Data)
- Provides high-quality stock price data
- Requires API key
- S&P500 ticker list is fetched from Wikipedia
- Required plan: EOD Historical Data — All World
  - Price: $19.99/month ($199.00/year with annual contract)
  - 100,000 API calls per day
  - 1,000 API requests per minute
  - 30+ years of historical data
  - For personal use
  - Details: https://eodhd.com/pricing
- Special ticker symbol handling
  - Dots (.) in tickers are converted to hyphens (-)
  - Examples: BRK.B → BRK-B, U.S.B → U-S-B

## Output

The following files are generated in the `reports/` directory:
- `market_breadth_YYYYMMDD.png`: Graph showing breadth indicators and S&P500 price movements
- `market_breadth_YYYYMMDD.csv`: Numerical data of breadth indicators
- `backtest_results_YYYYMMDD.csv`: Backtesting results and performance metrics
- `multi_etf_backtest_results_YYYYMMDD.csv`: Multi-ETF backtesting results
- `backtest_results_summary.md`: Detailed results report in Markdown format
- `backtest_results_summary.csv`: Results data in CSV format

### Graph Color Coding

The pink background in the graph indicates the following conditions:
- 200-day moving average trend is declining (breadth_ma_200_trend[i] == -1)
- Short-term moving average is below the 200-day moving average (breadth_ma_short[i] < breadth_ma_200[i])

This color coding helps visually identify market downtrends and weakness.

### Sample Output

```
market_breadth_20240315.png
```
![Market Breadth Sample](reports/sample_output.png)

In the sample graph above:
- Blue line: S&P500 price movement (logarithmic scale)
- Red line: Breadth indicator (percentage of stocks above 200-day moving average)
- Pink background: Area indicating market strength and trend direction
- Black dots: Peak points (market reversal points)

## System Requirements

- macOS: TkAgg backend
- Windows: Qt5Agg backend
- Linux: Agg backend (non-interactive)
- Minimum 8GB RAM recommended for backtesting
- SSD storage recommended for faster data access

## Notes

- Keep your API key in the `.env` file and do not upload it to GitHub
- EODHD is a paid service
- Commercial use requires a separate EODHD commercial license
- Backtesting results are for educational purposes only
- Past performance does not guarantee future results

## License

MIT License

Copyright (c) 2024 Market Breadth Analysis Tool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 