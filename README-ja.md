# マーケットブレッド分析ツール

[English version is here](README.md)

S&P500銘柄のマーケットブレッドを分析・可視化するツールです。

## 機能

- S&P500全銘柄の価格データ取得
- 200日移動平均線に基づくブレッド指標の計算
- ブレッド指標とS&P500の価格変動の可視化
- ピーク検出によるトレンド転換点の特定
- APIキー不要で履歴データの保存と再利用
- マーケットブレッドシグナルに基づく取引戦略のバックテスト
- 複数ETFのバックテスト機能

## ライブデータ（GitHub Pages）

最新の Market Breadth データは1日2回自動更新され、GitHub Pages で公開されています:

| リソース | URL |
|---|---|
| インタラクティブチャート | [market_breadth.html](https://tradermonty.github.io/market-breadth-analysis/market_breadth.html) |
| ダッシュボード | [index.html](https://tradermonty.github.io/market-breadth-analysis/) |
| データCSV | [market_breadth_data.csv](https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv) |
| サマリーCSV | [market_breadth_summary.csv](https://tradermonty.github.io/market-breadth-analysis/market_breadth_summary.csv) |

## 必要条件

- Python 3.8以上
- 必要なPythonパッケージ（requirements.txtに記載）
- APIキー（オプション）:
  - FMP APIキー（Financial Modeling Prep）
    - 料金・ドキュメント: https://site.financialmodelingprep.com/developer/docs/
    - 保存データを使用する場合は不要
  - GitHub Token（ワークフロー更新トリガー用）
    - Fine-grained PAT、Actions: Read and write 権限
    - `trigger_market_breadth.py` 使用時のみ必要

## インストール

1. リポジトリのクローン:
```bash
git clone [repository-url]
cd market_breadth
```

2. 必要なパッケージのインストール:
```bash
pip install -r requirements.txt
```

3. 環境変数の設定（オプション）:
`.env.sample`ファイルを`.env`にコピーして編集:
```bash
cp .env.sample .env
```
```
FMP_API_KEY=your_fmp_api_key        # FMPからの新規データ取得用
GITHUB_TOKEN=your_github_pat_here   # ワークフロー更新トリガー用（オプション）
```

## 使用方法

### マーケットブレッド分析

基本的な使用方法（APIキー必要）:
```bash
python market_breadth.py
```

保存データの使用:
```bash
python market_breadth.py --use_saved_data
```

追加オプション付き:
```bash
python market_breadth.py --start_date 2020-01-01 --short_ma 20 --use_saved_data
```

### バックテスト

単一戦略のバックテスト:
```bash
python backtest/backtest.py
```

複数ETFのバックテスト:
```bash
python backtest/run_multi_etf_backtest.py
```

### データ取得 / ワークフロートリガー

GitHub Pages から最新データを取得、またはデータが古い場合にワークフローをトリガー:

```bash
# 自動モード: 新鮮ならフェッチ、古ければワークフロー起動（デフォルト: 12時間閾値）
python trigger_market_breadth.py

# CSVデータのフェッチのみ（トリガーなし）
python trigger_market_breadth.py --fetch-only

# GitHub Actions ワークフローを強制トリガー
python trigger_market_breadth.py --trigger-only

# 鮮度閾値を6時間に変更
python trigger_market_breadth.py --max-age 6
```

Python から利用（LLM連携など）:
```python
from trigger_market_breadth import fetch_market_breadth

result = fetch_market_breadth(max_age_hours=12)
if result["status"] == "fresh":
    csv_data = result["csv_text"]  # そのまま分析に使用
elif result["status"] == "triggered":
    print(result["message"])  # "Workflow triggered. Data will be ready in ~5 minutes."
```

### コマンドライン引数

#### マーケットブレッド分析
- `--start_date`: 分析開始日（YYYY-MM-DD形式）
  - デフォルト: 現在から10年前
- `--short_ma`: 短期移動平均期間（10または20）
- `--use_saved_data`: FMPから取得せずに保存済みデータを使用

#### バックテスト
- `--start_date`: バックテスト開始日
- `--end_date`: バックテスト終了日
- `--initial_capital`: バックテストの初期資金
- `--position_size`: 資金に対するポジションサイズの割合
- `--stop_loss`: ストップロス率
- `--take_profit`: 利確率

### データの保存と再利用

このツールは履歴データの保存と再利用をサポートしています:

- データは`data/`ディレクトリに保存:
  - `sp500_price_data.csv`: S&P 500指数の価格データ
  - `stock_data.csv`: S&P 500全銘柄の個別株価データ

ワークフロー:
1. 初回使用（APIキー必要）:
   - `--use_saved_data`なしで実行し、データを取得・保存
   - データは自動的に保存され、将来の使用に備える
2. 以降の使用:
   - `--use_saved_data`付きで実行し、保存済みデータを使用
   - APIキー不要
   - API呼び出しがないため、より高速に実行

注意: 保存データを最新の市場データで更新するには、`--use_saved_data`なしで実行（APIキー必要）。

### データソース

Financial Modeling Prep (FMP)
- 米国株・世界株、ETF、ファンダメンタル等を提供
- 無料 "demo" キーが利用可能（速度・機能制限あり）
- 上位プランでより高いレートリミット・履歴期間を利用可能
- S&P500ティッカーリストはWikipediaから取得
- ドキュメント: https://site.financialmodelingprep.com/developer/docs/
- 特殊ティッカーシンボルの処理
  - ティッカー内のドット(.)はハイフン(-)に変換
  - 例: BRK.B → BRK-B

## 出力

以下のファイルが`reports/`ディレクトリに生成されます:
- `market_breadth_YYYYMMDD.png`: ブレッド指標とS&P500の価格変動を示すグラフ
- `market_breadth_YYYYMMDD.csv`: ブレッド指標の数値データ
- `backtest_results_YYYYMMDD.csv`: バックテスト結果とパフォーマンス指標
- `multi_etf_backtest_results_YYYYMMDD.csv`: 複数ETFバックテスト結果
- `backtest_results_summary.md`: 詳細な結果レポート（Markdown形式）
- `backtest_results_summary.csv`: 結果データ（CSV形式）

### グラフの色分け

グラフのピンク背景は以下の条件を示します:
- 200日移動平均線のトレンドが下降中（breadth_ma_200_trend[i] == -1）
- 短期移動平均線が200日移動平均線を下回っている（breadth_ma_short[i] < breadth_ma_200[i]）

この色分けにより、市場の下降トレンドと弱気相場を視覚的に識別できます。

### サンプル出力

```
market_breadth_20240315.png
```
![マーケットブレッドサンプル](reports/sample_output.png)

上記のサンプルグラフでは:
- 青線: S&P500の価格変動（対数スケール）
- 赤線: ブレッド指標（200日移動平均線を上回る銘柄の割合）
- ピンク背景: 市場の強さとトレンド方向を示す領域
- 黒点: ピークポイント（市場転換点）

## システム要件

- macOS: TkAggバックエンド
- Windows: Qt5Aggバックエンド
- Linux: Aggバックエンド（非対話型）
- バックテストには最低8GBのRAMを推奨
- 高速なデータアクセスのためSSDストレージを推奨

## 注意事項

- APIキーは`.env`ファイルに保存し、GitHubにアップロードしないでください
- 一部のFMPエンドポイントは有料プランが必要です
- 商用利用時はFMPの利用規約を確認してください
- バックテスト結果は教育目的のみです
- 過去の実績は将来の結果を保証するものではありません

## ライセンス

MITライセンス

Copyright (c) 2024 Market Breadth Analysis Tool

このソフトウェアと関連する文書ファイル（以下「ソフトウェア」）の複製を取得したすべての人に対し、ソフトウェアを制限なく扱うことを無償で許可します。これには、ソフトウェアの使用、複製、変更、結合、掲載、頒布、サブライセンス、および/または販売の権利、およびソフトウェアを提供された人にこれを行うことを許可する権利を含みます。

上記の著作権表示と本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 