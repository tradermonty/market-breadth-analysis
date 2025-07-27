#!/usr/bin/env python3
"""
Financial Modeling Prep API Data Fetcher
高精度な決算データを提供するFMP APIクライアント
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time
import json

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FMPDataFetcher:
    """Financial Modeling Prep API クライアント"""
    
    def __init__(self, api_key: str = None):
        """
        FMPDataFetcherの初期化
        
        Args:
            api_key: FMP API キー
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP API key is required. Set FMP_API_KEY environment variable.")
        
        self.base_url = "https://financialmodelingprep.com/stable"
        self.alt_base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        
        # Maximum performance rate limiting - 750 calls/minフル活用
        # Starter: 300 calls/min, Premium: 750 calls/min, Ultimate: 3000 calls/min  
        self.rate_limiting_active = False  # 動的制御フラグ
        self.calls_per_minute = 750  # Premium planの最大値（限界まで使用）
        self.calls_per_second = 12.5  # 750/60 = 12.5 calls/sec
        self.call_timestamps = []
        self.last_request_time = datetime(1970, 1, 1)
        self.min_request_interval = 0.08  # 1/12.5 = 0.08秒間隔（理論値）
        self.rate_limit_cooldown_until = datetime(1970, 1, 1)  # 制限解除時刻
        
        # パフォーマンス最適化フラグ
        self.max_performance_mode = True  # 429発生まで制限なし
        
        logger.info("FMP Data Fetcher initialized successfully")
    
    def _rate_limit_check(self):
        """最大パフォーマンス制限チェック - 429発生まで制限を最小限に"""
        now = datetime.now()
        
        # クールダウン期間後の制限解除チェック
        if self.rate_limiting_active and now > self.rate_limit_cooldown_until:
            self.rate_limiting_active = False
            self.max_performance_mode = True
            logger.info("Rate limiting deactivated - returning to maximum performance")
        
        # 429エラー発生時のみ厳格な制限を適用
        if self.rate_limiting_active:
            self.max_performance_mode = False
            # 保守的な制限を適用
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < 0.2:  # 429発生時は0.2秒間隔
                sleep_time = 0.2 - time_since_last
                logger.warning(f"Conservative rate limiting: sleeping {sleep_time:.3f}s")
                time.sleep(sleep_time)
                now = datetime.now()
                
            # 1分以内のコール履歴をフィルター
            self.call_timestamps = [
                ts for ts in self.call_timestamps 
                if (now - ts).total_seconds() < 60
            ]
            
            # 保守的な1分間制限（300 calls/min）
            if len(self.call_timestamps) >= 300:
                sleep_time = 60 - (now - self.call_timestamps[0]).total_seconds() + 1
                logger.warning(f"Conservative per-minute limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                now = datetime.now()
        elif self.max_performance_mode:
            # 最大パフォーマンスモード：429発生まで制限を完全に無効化
            # ネットワーク遅延による自然なレート制限のみ
            pass
        else:
            # 通常モード：理論値まで使用
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
                now = datetime.now()
        
        # コール履歴の記録（429エラー時のみ）
        if self.rate_limiting_active:
            self.call_timestamps.append(now)
        
        self.last_request_time = now
    
    def _activate_rate_limiting(self, duration_minutes: int = 5):
        """429エラー発生時にレート制限を有効化"""
        self.rate_limiting_active = True
        self.max_performance_mode = False
        self.rate_limit_cooldown_until = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"Rate limiting activated for {duration_minutes} minutes due to 429 error")
    
    def _make_request(self, endpoint: str, params: Dict = None, max_retries: int = 3) -> Optional[Dict]:
        """
        FMP APIへのリクエスト実行（リトライと指数バックオフ付き）
        
        Args:
            endpoint: APIエンドポイント
            params: リクエストパラメータ
            max_retries: 最大リトライ回数
        
        Returns:
            APIレスポンス
        """
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries + 1):
            # レート制限チェック（軽微または429エラー後の厳格制限）
            self._rate_limit_check()
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                # Handle different HTTP status codes
                if response.status_code == 404:
                    logger.debug(f"Endpoint not found (404): {endpoint}")
                    return None
                elif response.status_code == 403:
                    logger.warning(f"Access forbidden (403) for {endpoint} - check API plan limits")
                    return None
                elif response.status_code == 429:
                    # 429エラー発生時：動的レート制限を有効化
                    self._activate_rate_limiting(duration_minutes=5)
                    
                    if attempt < max_retries:
                        # 指数バックオフ: 2^attempt * 5秒 + ランダムジッター
                        base_delay = 5 * (2 ** attempt)
                        jitter = base_delay * 0.1 * (0.5 - time.time() % 1)  # ±10%のジッター
                        delay = base_delay + jitter
                        
                        logger.warning(f"Rate limit exceeded (429) for {endpoint}. "
                                     f"Activating rate limiting for 5 minutes. "
                                     f"Attempt {attempt + 1}/{max_retries + 1}. "
                                     f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded (429) for {endpoint}. Max retries exceeded.")
                        return None
                
                response.raise_for_status()
                
                data = response.json()
                
                # Check for empty or invalid responses
                if data is None:
                    logger.debug(f"Empty response from {endpoint}")
                    return None
                elif isinstance(data, dict) and data.get('Error Message'):
                    logger.debug(f"API error for {endpoint}: {data.get('Error Message')}")
                    return None
                elif isinstance(data, list) and len(data) == 0:
                    logger.debug(f"Empty data array from {endpoint}")
                    return None
                
                logger.debug(f"Successfully fetched data from {endpoint}")
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    delay = 2 ** attempt  # 指数バックオフ
                    logger.warning(f"Request failed for {endpoint}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.debug(f"Request failed for {endpoint} after {max_retries} retries: {e}")
                    return None
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error for {endpoint}: {e}")
                return None
        
        return None
    
    def get_earnings_calendar(self, from_date: str, to_date: str, target_symbols: List[str] = None, us_only: bool = True) -> List[Dict]:
        """
        決算カレンダーをBulk取得 (Premium+ plan required)
        90日を超える期間は自動的に分割
        
        Args:
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
            target_symbols: 対象銘柄リスト（省略時は全銘柄）
            us_only: アメリカ市場のみに限定するか（デフォルト: True）
        
        Returns:
            決算データリスト
        """
        logger.info(f"Fetching earnings calendar from {from_date} to {to_date}")
        
        # 日付をdatetimeオブジェクトに変換
        start_dt = datetime.strptime(from_date, '%Y-%m-%d')
        end_dt = datetime.strptime(to_date, '%Y-%m-%d')
        
        # FMP Premium planの制限チェック（2020年8月以前はデータなし）
        fmp_limit_date = datetime(2020, 8, 1)
        if start_dt < fmp_limit_date:
            error_msg = (
                f"\n{'='*60}\n"
                f"FMP データソース制限エラー\n"
                f"{'='*60}\n"
                f"開始日: {from_date}\n"
                f"FMP Premium plan制限: 2020年8月1日以降のデータのみ利用可能\n\n"
                f"解決策:\n"
                f"1. 開始日を2020-08-01以降に変更\n"
                f"   python main.py --start_date 2020-08-01\n\n"
                f"{'='*60}"
            )
            logger.error(error_msg)
            raise ValueError(f"FMP Premium plan does not support data before 2020-08-01. Requested start date: {from_date}")
        
        # 開始日が制限日以降でも、一部が制限範囲に入る場合の警告
        if start_dt < datetime(2020, 9, 1):
            logger.warning(f"Warning: FMP data coverage may be limited for dates close to August 2020. "
                         f"For comprehensive historical analysis, consider using EODHD data source.")
        
        # 期間が90日を超える場合は分割
        max_days = 30  # 30日ごとに分割（安全マージン）
        all_data = []
        
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=max_days), end_dt)
            
            params = {
                'from': current_start.strftime('%Y-%m-%d'),
                'to': current_end.strftime('%Y-%m-%d')
            }
            
            logger.info(f"Fetching chunk: {params['from']} to {params['to']}")
            chunk_data = self._make_request('earnings-calendar', params)
            
            if chunk_data is None:
                logger.warning(f"Failed to fetch data for {params['from']} to {params['to']}")
            elif len(chunk_data) == 0:
                logger.info(f"No data for {params['from']} to {params['to']}")
            else:
                all_data.extend(chunk_data)
                logger.info(f"Retrieved {len(chunk_data)} records for this chunk")
            
            # 次の期間へ
            current_start = current_end + timedelta(days=1)
            
            # レート制限は_rate_limit_check()で動的に管理
            # チャンク間の固定待機は削除し、最大スピードを確保
        
        if len(all_data) == 0:
            logger.warning("earnings-calendar endpoint returned no data, trying alternative method")
            return self._get_earnings_calendar_alternative(from_date, to_date, target_symbols, us_only)
        
        # アメリカ市場のみにフィルタリング
        if us_only:
            us_data = []
            for item in all_data:
                symbol = item.get('symbol', '')
                # US市場の銘柄を識別（通常はexchangeShortNameで判定）
                exchange = item.get('exchangeShortName', '').upper()
                if exchange in ['NASDAQ', 'NYSE', 'AMEX', 'NYSE AMERICAN']:
                    us_data.append(item)
                # exchangeShortName情報がない場合は、通常のUS銘柄パターンで判定
                elif exchange == '' and symbol and not any(x in symbol for x in ['.TO', '.L', '.PA', '.AX', '.DE', '.HK']):
                    us_data.append(item)
            
            logger.info(f"Filtered to {len(us_data)} US market earnings records (from {len(all_data)} total)")
            return us_data
        
        logger.info(f"Retrieved total {len(all_data)} earnings records")
        return all_data
    
    def _get_earnings_calendar_alternative(self, from_date: str, to_date: str, 
                                           target_symbols: List[str] = None, us_only: bool = True) -> List[Dict]:
        """
        代替決算カレンダー取得
        個別銘柄のearnings-surprises APIを使用
        
        Args:
            from_date: 開始日
            to_date: 終了日
            target_symbols: 対象銘柄リスト（Noneの場合はデフォルトリスト使用）
        """
        logger.info("Using alternative earnings data collection method")
        
        # Premiumプラン対応：拡張銘柄リスト（主要S&P 500銘柄）
        major_symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 
            'CRM', 'ADBE', 'NFLX', 'INTC', 'AMD', 'AVGO', 'QCOM', 'TXN', 'CSCO',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'USB', 'PNC',
            'TFC', 'COF', 'SCHW', 'CB', 'MMC', 'AON', 'SPGI', 'ICE',
            
            # Healthcare
            'JNJ', 'PFE', 'ABT', 'MRK', 'TMO', 'DHR', 'BMY', 'ABBV', 'LLY', 'UNH',
            'CVS', 'AMGN', 'GILD', 'MDLZ', 'BSX', 'SYK', 'ZTS', 'ISRG',
            
            # Consumer Discretionary
            'TSLA', 'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
            'CMG', 'ORLY', 'AZO', 'RCL', 'MAR', 'HLT', 'MGM', 'WYNN',
            
            # Consumer Staples
            'KO', 'PEP', 'WMT', 'COST', 'PG', 'CL', 'KMB', 'GIS', 'K', 'SJM',
            'HSY', 'CPB', 'CAG', 'HRL', 'MKC', 'LW', 'CHD',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'FDX',
            'NOC', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'OTIS', 'CARR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'VLO', 'MPC', 'PSX',
            'KMI', 'WMB', 'OKE', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO',
            
            # Materials
            'LIN', 'SHW', 'APD', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'IFF',
            'ALB', 'CE', 'VMC', 'MLM', 'PKG', 'BALL', 'AMCR',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EQR',
            'AVB', 'VTR', 'ESS', 'MAA', 'EXR', 'UDR', 'CPT',
            
            # Utilities
            'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'D', 'EXC', 'XEL', 'WEC', 'AWK',
            'PPL', 'ES', 'FE', 'ETR', 'AES', 'LNT', 'NI',
            
            # Communication Services
            'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
            'CHTR', 'ATVI', 'EA', 'TTWO', 'NWSA', 'NWS', 'FOXA', 'FOX',
            
            # Mid/Small Cap (includes MANH)
            'MANH', 'POOL', 'ODFL', 'WST', 'MPWR', 'ENPH', 'ALGN', 'MKTX', 'CDAY',
            'PAYC', 'FTNT', 'ANSS', 'CDNS', 'SNPS', 'KLAC', 'LRCX', 'AMAT', 'MCHP'
        ]
        
        earnings_data = []
        start_dt = datetime.strptime(from_date, '%Y-%m-%d')
        end_dt = datetime.strptime(to_date, '%Y-%m-%d')
        
        for symbol in major_symbols:
            try:
                # Earnings surprises API (available in Starter)
                symbol_data = self._make_request(f'earnings-surprises/{symbol}')
                
                if symbol_data and isinstance(symbol_data, list):
                    for earning in symbol_data:
                        try:
                            earning_date = datetime.strptime(earning.get('date', ''), '%Y-%m-%d')
                            if start_dt <= earning_date <= end_dt:
                                # Convert to earnings-calendar format
                                converted = {
                                    'symbol': symbol,
                                    'date': earning.get('date'),
                                    'epsActual': earning.get('actualEarningResult'),
                                    'epsEstimate': earning.get('estimatedEarning'),
                                    'time': None,  # Not available in Starter
                                    'revenueActual': None,  # Not available in earnings-surprises
                                    'revenueEstimate': None,  # Not available in earnings-surprises
                                    'fiscalDateEnding': earning.get('date'),
                                    'updatedFromDate': earning.get('date')
                                }
                                earnings_data.append(converted)
                                logger.debug(f"Added {symbol} earnings for {earning.get('date')}")
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Date parsing error for {symbol}: {e}")
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to get earnings for {symbol}: {e}")
                continue
        
        # アメリカ市場のみにフィルタリング（代替メソッド用）
        if us_only:
            us_earnings = []
            for earning in earnings_data:
                symbol = earning.get('symbol', '')
                # アメリカ市場の銘柄（S&P銘柄等）のみを対象
                if symbol and not any(x in symbol for x in ['.TO', '.L', '.PA', '.AX', '.DE', '.HK']):
                    us_earnings.append(earning)
            earnings_data = us_earnings
            logger.info(f"Filtered to {len(earnings_data)} US market earnings records using alternative method")
        
        # Sort by date
        earnings_data.sort(key=lambda x: x.get('date', ''))
        logger.info(f"Retrieved {len(earnings_data)} earnings records using alternative method")
        
        return earnings_data
    
    
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        企業プロファイル取得
        
        Args:
            symbol: 銘柄コード
        
        Returns:
            企業情報
        """
        logger.debug(f"Fetching company profile for {symbol}")
        
        # Try different endpoints - profile data is only available on v3 API
        endpoints_to_try = [
            ('v3', f'profile/{symbol}'),      # v3 endpoint (correct one)
            ('stable', f'profile/{symbol}'),  # stable endpoint (backup)
        ]
        
        data = None
        for api_version, endpoint in endpoints_to_try:
            base_url = self.base_url if api_version == 'stable' else self.alt_base_url
            logger.debug(f"Trying {api_version} endpoint for profile: {endpoint}")
            
            # Temporarily override base URL for this request
            original_base_url = self.base_url
            self.base_url = base_url
            
            data = self._make_request(endpoint)
            
            # Restore original base URL
            self.base_url = original_base_url
            
            if data is not None:
                logger.debug(f"Successfully fetched profile using: {api_version}/{endpoint}")
                break
            else:
                logger.debug(f"Profile endpoint failed: {api_version}/{endpoint}")
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        
        logger.warning(f"Failed to fetch company profile for {symbol} using all available endpoints")
        return None
    
    def process_earnings_data(self, earnings_data: List[Dict]) -> pd.DataFrame:
        """
        FMP決算データを標準形式に変換
        
        Args:
            earnings_data: FMP決算データ
        
        Returns:
            標準化されたDataFrame
        """
        if not earnings_data:
            return pd.DataFrame()
        
        processed_data = []
        
        for earning in earnings_data:
            try:
                # FMPデータ構造に基づく処理
                processed_earning = {
                    'code': earning.get('symbol', '') + '.US',  # .US suffix for compatibility
                    'report_date': earning.get('date', ''),
                    'date': earning.get('date', ''),  # 実際の決算日
                    'before_after_market': self._parse_timing(earning.get('time', '')),
                    'currency': 'USD',  # FMPは主にUSDデータ
                    'actual': self._safe_float(earning.get('epsActual')),
                    'estimate': self._safe_float(earning.get('epsEstimated')),  # FMP uses 'epsEstimated'
                    'difference': 0,  # 後で計算
                    'percent': 0,     # 後で計算
                    'revenue_actual': self._safe_float(earning.get('revenueActual')),
                    'revenue_estimate': self._safe_float(earning.get('revenueEstimate')),
                    'updated_from_date': earning.get('updatedFromDate', ''),
                    'fiscal_date_ending': earning.get('fiscalDateEnding', ''),
                    'data_source': 'FMP'
                }
                
                # サプライズ率計算
                if processed_earning['actual'] is not None and processed_earning['estimate'] is not None:
                    if processed_earning['estimate'] != 0:
                        processed_earning['difference'] = processed_earning['actual'] - processed_earning['estimate']
                        processed_earning['percent'] = (processed_earning['difference'] / abs(processed_earning['estimate'])) * 100
                
                processed_data.append(processed_earning)
                
            except Exception as e:
                logger.warning(f"Error processing earning data: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            # 日付でソート
            df = df.sort_values('report_date')
            logger.info(f"Processed {len(df)} earnings records")
        
        return df
    
    def _parse_timing(self, time_str: str) -> str:
        """
        FMPの時間情報をBefore/AfterMarket形式に変換
        
        Args:
            time_str: FMP時間文字列
        
        Returns:
            Before/AfterMarket
        """
        if not time_str:
            return None
        
        time_lower = time_str.lower()
        
        if any(keyword in time_lower for keyword in ['before', 'pre', 'bmo']):
            return 'BeforeMarket'
        elif any(keyword in time_lower for keyword in ['after', 'post', 'amc']):
            return 'AfterMarket'
        else:
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """
        安全なfloat変換
        
        Args:
            value: 変換対象値
        
        Returns:
            float値またはNone
        """
        if value is None or value == '':
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_historical_price_data(self, symbol: str, from_date: str, to_date: str) -> Optional[List[Dict]]:
        """
        FMPから株価履歴データを取得
        
        Args:
            symbol: 銘柄コード（例: "AAPL"）
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
        
        Returns:
            株価データリスト
        """
        logger.debug(f"Fetching historical price data for {symbol} from {from_date} to {to_date}")
        
        # Try different endpoint formats and base URLs for FMP
        endpoints_to_try = [
            # Stable API endpoints
            ('stable', f'historical-price-full/{symbol}'),
            ('stable', f'historical-chart/1day/{symbol}'),
            ('stable', f'historical/{symbol}'),
            # API v3 endpoints
            ('v3', f'historical-price-full/{symbol}'),
            ('v3', f'historical-chart/1day/{symbol}'),
            ('v3', f'historical-daily-prices/{symbol}'),
        ]
        
        params = {
            'from': from_date,
            'to': to_date
        }
        
        data = None
        successful_endpoint = None
        
        for api_version, endpoint in endpoints_to_try:
            base_url = self.base_url if api_version == 'stable' else self.alt_base_url
            logger.debug(f"Trying {api_version} endpoint: {endpoint}")
            
            # Temporarily override base URL for this request
            original_base_url = self.base_url
            self.base_url = base_url
            
            # 最大パフォーマンスで実行
            data = self._make_request(endpoint, params, max_retries=3)
            
            # Restore original base URL
            self.base_url = original_base_url
            
            if data is not None:
                successful_endpoint = f"{api_version}/{endpoint}"
                logger.debug(f"Successfully fetched data using: {successful_endpoint}")
                break
            else:
                logger.debug(f"Endpoint failed: {api_version}/{endpoint}")
                # エンドポイント間の固定待機を削除（動的制限で管理）
        
        if data is None:
            logger.warning(f"Failed to fetch historical price data for {symbol} using all available endpoints")
            return None
        
        # Handle different response formats
        if isinstance(data, dict):
            # Standard format with 'historical' field
            if 'historical' in data:
                return data['historical']
            # Alternative format with direct data
            elif 'results' in data:
                return data['results']
            # Chart format
            elif isinstance(data, dict) and 'date' in str(data):
                return [data]
        elif isinstance(data, list):
            # Direct list format
            return data
        
        logger.warning(f"Unexpected data format for {symbol}: {type(data)}")
        return None
    
    def get_sp500_constituents(self) -> List[str]:
        """
        S&P 500構成銘柄を取得
        
        Returns:
            銘柄コードリスト
        """
        logger.debug("Fetching S&P 500 constituents")
        
        data = self._make_request('sp500_constituent')
        
        if data is None:
            logger.warning("Failed to fetch S&P 500 constituents")
            return []
        
        # Extract symbols from constituent data
        symbols = []
        if isinstance(data, list):
            symbols = [item.get('symbol', '') for item in data if item.get('symbol')]
        
        logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
        return symbols
    
    
    
    
    def get_mid_small_cap_symbols(self, min_market_cap: float = 1e9, max_market_cap: float = 50e9) -> List[str]:
        """
        時価総額ベースで中小型株を取得
        
        Args:
            min_market_cap: 最小時価総額（デフォルト: $1B）
            max_market_cap: 最大時価総額（デフォルト: $50B）
        
        Returns:
            中小型株の銘柄コードリスト
        """
        logger.info(f"Fetching mid/small cap stocks (${min_market_cap/1e9:.1f}B - ${max_market_cap/1e9:.1f}B)")
        
        # FMPのstock screenerを使用
        params = {
            'marketCapMoreThan': int(min_market_cap),
            'marketCapLowerThan': int(max_market_cap),
            'limit': 3000  # 大きめの制限を設定
        }
        
        # Try different endpoints
        endpoints_to_try = [
            'stock_screener',  # 正しいエンドポイント名
            'screener',        # 代替エンドポイント 
            'stock-screener'   # 元のエンドポイント
        ]
        
        data = None
        for endpoint in endpoints_to_try:
            data = self._make_request(endpoint, params)
            if data is not None:
                logger.debug(f"Successfully used endpoint: {endpoint}")
                break
        
        if data is None:
            logger.warning("Stock screener API not available, using fallback method")
            # Fallback: Use market cap filtering in earnings data processing
            return self._get_mid_small_cap_fallback(min_market_cap, max_market_cap)
        
        # US市場の銘柄のみを抽出
        us_symbols = []
        if isinstance(data, list):
            for stock in data:
                symbol = stock.get('symbol', '')
                exchange = stock.get('exchangeShortName', '')
                country = stock.get('country', '')
                
                # US市場の銘柄のみを選択
                if (exchange in ['NASDAQ', 'NYSE', 'AMEX'] or country == 'US') and symbol:
                    # 一般的でない銘柄タイプを除外
                    if not any(x in symbol for x in ['.', '-', '^', '=']):
                        us_symbols.append(symbol)
        
        logger.info(f"Retrieved {len(us_symbols)} mid/small cap US stocks")
        return us_symbols[:2000]  # 実用的な数に制限
    
    def _get_mid_small_cap_fallback(self, min_market_cap: float, max_market_cap: float) -> List[str]:
        """
        Stock screenerが利用できない場合の代替手段
        人気のある中小型株リストを使用
        """
        logger.info("Using curated mid/small cap stock list as fallback")
        
        # 中小型株として人気の銘柄リスト（時価総額範囲に適合するもの）
        mid_small_cap_stocks = [
            # Regional Banks (typically $2-20B market cap)
            'OZK', 'ZION', 'PNFP', 'FHN', 'SNV', 'FULT', 'CBSH', 'ONB', 'IBKR',
            'BKU', 'OFG', 'FFBC', 'COLB', 'BANC', 'FFIN', 'FBP', 'CUBI', 'ASB',
            'HFWA', 'PPBI', 'SSB', 'TCBI', 'NBHC', 'BANR', 'CVBF', 'UMBF',
            'LKFN', 'NWBI', 'HOPE', 'SBCF', 'WSFS', 'SFBS', 'HAFC', 'FBNC',
            'CFFN', 'ABCB', 'BHLB', 'STBA',
            
            # Mid-cap industrials and tech
            'CALM', 'AIR', 'AZZ', 'JEF', 'ACI', 'MSM', 'SMPL', 'GBX', 'UNF',
            'NEOG', 'WDFC', 'CNXC', 'IIIN', 'WBS', 'HWC', 'PRGS', 'AGYS',
            'AA', 'ALK', 'SLG', 'PLXS', 'SFNC', 'KNX', 'MANH', 'QRVO', 'WRLD',
            'ADNT', 'TRMK', 'NXT', 'AIT', 'VFC', 'SF', 'EXTR', 'WHR', 'GPI',
            'CCS', 'CALX', 'CPF', 'CACI', 'GATX', 'ORI', 'HZO', 'MRTN', 'SANM',
            'ELS', 'HLI', 'RNR', 'RNST', 'CVLT', 'FLEX', 'NFG', 'LBRT', 'VIRT',
            'DLB', 'BHE', 'OSK', 'VIAV', 'ATGE', 'BC', 'SXI', 'OLN', 'PMT',
            'SXC', 'DT', 'CRS', 'ABG', 'NTCT', 'CFR', 'CVCO', 'STEL', 'HTH',
            'SKYW', 'CSWI', 'FHI', 'BOOT', 'BFH', 'ALGM', 'TMP', 'ALV', 'VSTS',
            'RBC', 'JHG', 'ARCB', 'PIPR', 'CR', 'NLY', 'EAT'
        ]
        
        logger.info(f"Using {len(mid_small_cap_stocks)} curated mid/small cap symbols")
        return mid_small_cap_stocks
    
    def get_api_usage_stats(self) -> Dict:
        """
        API使用統計を取得
        
        Returns:
            使用統計情報
        """
        now = datetime.now()
        recent_calls_minute = [
            ts for ts in self.call_timestamps 
            if (now - ts).total_seconds() < 60
        ]
        recent_calls_second = [
            ts for ts in self.call_timestamps 
            if (now - ts).total_seconds() < 1
        ]
        
        return {
            'calls_last_minute': len(recent_calls_minute),
            'calls_last_second': len(recent_calls_second),
            'calls_per_minute_limit': self.calls_per_minute,
            'calls_per_second_limit': self.calls_per_second,
            'remaining_calls_minute': max(0, self.calls_per_minute - len(recent_calls_minute)),
            'remaining_calls_second': max(0, self.calls_per_second - len(recent_calls_second)),
            'api_key_set': bool(self.api_key),
            'base_url': self.base_url,
            'min_request_interval': self.min_request_interval
        }


