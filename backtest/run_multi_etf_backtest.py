import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')  # プロットを表示しないようにバックエンドを設定

# Add parent directory to path to import backtest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest import Backtest

def run_multi_etf_backtest(etfs, start_date=None, end_date=None, 
                          short_ma=20, long_ma=200, initial_capital=50000,
                          slippage=0.001, commission=0.001, use_saved_data=True,
                          debug=False, threshold=0.5, ma_type='ema', stop_loss_pct=0.10,
                          no_show_plot=True):
    """
    複数のETFに対してバックテストを実施し、結果をまとめたMD形式の表を出力する
    
    Parameters:
    -----------
    etfs : list
        バックテスト対象のETFシンボルのリスト
    start_date : str, optional
        バックテスト開始日 (YYYY-MM-DD形式)
    end_date : str, optional
        バックテスト終了日 (YYYY-MM-DD形式)
    short_ma : int, optional
        短期移動平均の期間
    long_ma : int, optional
        長期移動平均の期間
    initial_capital : float, optional
        初期投資額
    slippage : float, optional
        スリッページ率
    commission : float, optional
        取引手数料率
    use_saved_data : bool, optional
        保存済みデータを使用するかどうか
    debug : bool, optional
        デバッグモードを有効にするかどうか
    threshold : float, optional
        ボトム検出のしきい値
    ma_type : str, optional
        移動平均の種類 ('ema' or 'sma')
    stop_loss_pct : float, optional
        損切りパーセンテージ
    no_show_plot : bool, optional
        プロットを表示しないかどうか（デフォルト: True）
    """
    # 日付が指定されていない場合は、10年前から現在までをデフォルトとする
    if end_date is None:
        end_date = (datetime.now()- timedelta(days=1)).strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # 結果を格納するリスト
    results = []
    
    # 各ETFに対してバックテストを実行
    for symbol in etfs:
        print(f"\n{'='*50}")
        print(f"Running backtest for {symbol}...")
        print(f"{'='*50}")
        
        try:
            # バックテストの実行
            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                short_ma=short_ma,
                long_ma=long_ma,
                initial_capital=initial_capital,
                slippage=slippage,
                commission=commission,
                use_saved_data=use_saved_data,
                debug=debug,
                threshold=threshold,
                ma_type=ma_type,
                symbol=symbol,
                stop_loss_pct=stop_loss_pct,
                no_show_plot=no_show_plot
            )
            
            # バックテストの実行
            backtest.run()
            
            # 結果を可視化（no_show_plotパラメータを渡す）
            backtest.visualize_results(show_plot=not no_show_plot)
            
            # 結果を取得
            result = {
                'Symbol': symbol,
                'Total Return': backtest.total_return,
                'Annual Return (CAGR)': backtest.cagr,
                'Sharpe Ratio': backtest.sharpe_ratio,
                'Maximum Drawdown': backtest.max_drawdown,
                'Win Rate': backtest.win_rate,
                'Profit-Loss Ratio': backtest.profit_loss_ratio,
                'Profit Factor': backtest.profit_factor,
                'Calmar Ratio': backtest.calmar_ratio,
                'Expected Value': backtest.expected_value,
                'Avg. PnL per Trade': backtest.avg_pnl_per_trade,
                'Pareto Ratio': backtest.pareto_ratio
            }
            
            results.append(result)
            
            # 少し待機してAPIリクエストを制限
            time.sleep(2)
            
        except Exception as e:
            print(f"Error running backtest for {symbol}: {e}")
            # エラーが発生した場合でも、空の結果を追加して処理を継続
            results.append({
                'Symbol': symbol,
                'Total Return': np.nan,
                'Annual Return (CAGR)': np.nan,
                'Sharpe Ratio': np.nan,
                'Maximum Drawdown': np.nan,
                'Win Rate': np.nan,
                'Profit-Loss Ratio': np.nan,
                'Profit Factor': np.nan,
                'Calmar Ratio': np.nan,
                'Expected Value': np.nan,
                'Avg. PnL per Trade': np.nan,
                'Pareto Ratio': np.nan
            })
            continue
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # 結果を表示
    print("\n" + "="*100)
    print("BACKTEST RESULTS SUMMARY")
    print("="*100)
    
    # 結果を整形して表示
    formatted_results = results_df.copy()
    
    # パーセンテージ形式に変換
    formatted_results['Total Return'] = formatted_results['Total Return'].map('{:.2%}'.format)
    formatted_results['Annual Return (CAGR)'] = formatted_results['Annual Return (CAGR)'].map('{:.2%}'.format)
    formatted_results['Maximum Drawdown'] = formatted_results['Maximum Drawdown'].map('{:.2%}'.format)
    formatted_results['Win Rate'] = formatted_results['Win Rate'].map('{:.2%}'.format)
    
    # 小数点以下2桁に丸める
    formatted_results['Sharpe Ratio'] = formatted_results['Sharpe Ratio'].map('{:.2f}'.format)
    formatted_results['Profit-Loss Ratio'] = formatted_results['Profit-Loss Ratio'].map('{:.2f}'.format)
    formatted_results['Profit Factor'] = formatted_results['Profit Factor'].map('{:.2f}'.format)
    formatted_results['Calmar Ratio'] = formatted_results['Calmar Ratio'].map('{:.2f}'.format)
    formatted_results['Expected Value'] = formatted_results['Expected Value'].map('${:.2f}'.format)
    formatted_results['Avg. PnL per Trade'] = formatted_results['Avg. PnL per Trade'].map('${:.2f}'.format)
    formatted_results['Pareto Ratio'] = formatted_results['Pareto Ratio'].map('{:.2f}'.format)
    
    # MD形式の表を生成
    md_table = formatted_results.to_markdown(index=False)
    
    # 結果を表示
    print(md_table)
    
    # 結果をファイルに保存
    with open('reports/backtest_results_summary.md', 'w') as f:
        f.write("# ETF Backtest Results Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Period: {start_date} to {end_date}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Short MA: {short_ma}\n")
        f.write(f"- Long MA: {long_ma}\n")
        f.write(f"- Initial Capital: ${initial_capital:,.2f}\n")
        f.write(f"- Slippage: {slippage:.3f}\n")
        f.write(f"- Commission: {commission:.3f}\n")
        f.write(f"- MA Type: {ma_type.upper()}\n")
        f.write(f"- Stop Loss: {stop_loss_pct:.1%}\n\n")
        f.write(md_table)
    
    print(f"\nResults saved to reports/backtest_results_summary.md")
    
    # 結果をCSVファイルにも保存
    results_df.to_csv('reports/backtest_results_summary.csv', index=False)
    print(f"Results also saved to reports/backtest_results_summary.csv")
    
    return results_df

if __name__ == "__main__":
    # バックテスト対象のETFリスト（カテゴリ別に整理）
    etfs = [
        # 米国株式市場の多様化のためのETF
        'SPY',  # SPDR S&P 500 ETF
        'VOO',  # Vanguard S&P 500 ETF（S&P500指数）
        'VTI',  # Vanguard Total Stock Market ETF（全米株式市場）
        'QQQ',  # Invesco QQQ Trust
        'VUG',  # Vanguard Growth ETF（成長株）
        'VTV',  # Vanguard Value ETF（バリュー株）
        'VB',   # Vanguard Small-Cap ETF（小型株）
        'VEA',  # Vanguard Developed Markets ETF（先進国株式）
        'VWO',  # Vanguard Emerging Markets ETF（新興国株式）
        
        # セクター別ETF
        'XLF',  # Financial Select Sector SPDR Fund（金融セクター）
        'XLE',  # Energy Select Sector SPDR Fund（エネルギーセクター）
        'XLK',  # Technology Select Sector SPDR Fund（テクノロジーセクター）
        'XLV',  # Health Care Select Sector SPDR Fund（ヘルスケアセクター）
        'XLI',  # Industrial Select Sector SPDR Fund（工業セクター）
        'VGT',  # Vanguard Information Technology ETF
        
        # レバレッジ・ブルETF
        'SSO',  # ProShares Ultra S&P 500
        'TQQQ', # ProShares UltraPro QQQ
        'QLD',  # ProShares Ultra QQQ
        'SPXL', # Direxion Daily S&P 500 Bull 3X Shares
        'SOXL', # Direxion Daily Semiconductor Bull 3X Shares
        
        # 小型・中型株ETF
        'TNA',  # Direxion Daily Small Cap Bull 3X Shares
        'IWR',  # iShares Russell Mid-Cap ETF
        
        # 成長株ETF
        'SCHG', # Schwab U.S. Large-Cap Growth ETF
        'IWF',  # iShares Russell 1000 Growth ETF
        
        # ファクターETF
        'MTUM', # iShares MSCI USA Momentum Factor ETF
        
        # 配当重視ETF
        'VYM',  # Vanguard High Dividend Yield ETF（高配当株）
        'SCHD', # Schwab U.S. Dividend Equity ETF（配当株）
        'NOBL'  # ProShares S&P 500 Dividend Aristocrats ETF（配当貴族株）
    ]
    
    # バックテストの実行
    run_multi_etf_backtest(
        etfs=etfs,
        start_date='2001-01-01',
        end_date='2024-12-31',
        short_ma=8,
        long_ma=200,
        initial_capital=50000,
        slippage=0.001,
        commission=0.001,
        use_saved_data=True,
        debug=False,
        threshold=0.5,
        ma_type='ema',
        stop_loss_pct=0.08,
        no_show_plot=True
    ) 