# Market Breadth チャート Plotly 移行 設計書

## 1. 概要

### 1.1 目的

現在 Matplotlib で PNG 画像として生成している Market Breadth チャートを、
Plotly ベースのインタラクティブ HTML に移行する。

### 1.2 期待される効果

| 項目 | 現状 (Matplotlib + PNG) | 移行後 (Plotly + HTML) |
|---|---|---|
| ユーザー操作 | 静止画像を閲覧するのみ | ズーム、パン、ホバーで日付・値を確認 |
| 期間選択 | 不可（10年分全体を表示） | レンジセレクター（1Y/3Y/5Y/ALL）で切替 |
| データ確認 | 画像を目視で読み取り | ホバーツールチップで正確な数値を表示 |
| ファイル形式 | PNG（約1-2MB） | HTML（約5MB、JS埋め込み自己完結） |
| 配信方式 | GitHub Pages（変更なし） | GitHub Pages（変更なし） |

### 1.3 変更しないもの

- データ取得ロジック（`fmp_data_fetcher.py`）
- データ加工ロジック（`extract_chart_data()` 関数）
- `backtest/` モジュール（別チャート、影響なし）
- `trade/` モジュール（チャートに依存しない）
- GitHub Actions のスケジュール・トリガー設定
- コマンドライン引数の仕様

---

## 2. 現行チャート仕様（色・デザインの完全な定義）

### 2.1 レイアウト

```
┌─────────────────────────────────┐
│  Panel 1: S&P 500 Price         │  高さ比率 1:1
│  (対数スケール Y軸)              │
├─────────────────────────────────┤
│  Panel 2: Breadth Index         │  高さ比率 1:1
│  (リニアスケール Y軸 0.0-1.0)    │
└─────────────────────────────────┘
       共有 X軸（日付）
```

- 全体サイズ: `figsize=(12, 12)`
- X軸共有: `sharex=True`
- 背景色: 白 (`facecolor='white'`)

### 2.2 Panel 1: S&P 500 Price — 要素一覧

| 要素 | Matplotlib コード | 色 (RGB/名前) | スタイル |
|---|---|---|---|
| S&P 500 価格ライン | `color='cyan'` | **cyan** (`#00FFFF`) | linewidth=2, zorder=2 |
| Bearish Signal 背景 | `color=(1.0, 0.9, 0.96)` | **薄ピンク** (`rgba(255, 230, 245, 0.3)`) | alpha=0.3, axvspan |
| Trough マーカー (S&P上) | `color='purple', marker='v'` | **purple** (`#800080`) | 逆三角形, s=150, zorder=3 |
| Y軸 | `set_yscale('log')` | — | 対数スケール, 目盛: 100-1000 |
| グリッド | `grid(True)` | — | デフォルトグレー |
| タイトル | `'S&P 500 Price'` | — | fontsize=16, pad=20 |

**Bearish Signal 背景の条件:**
```python
(breadth_ma_200_trend == -1) and (breadth_ma_short < breadth_ma_200)
```
→ 200MA が下降トレンド かつ 短期MA が 200MA を下回っている期間

### 2.3 Panel 2: Breadth Index — 要素一覧

| 要素 | Matplotlib コード | 色 (RGB/名前) | スタイル |
|---|---|---|---|
| Breadth Index 200-Day MA | `color='green'` | **green** (`#008000`) | linewidth=2, zorder=2 |
| Breadth Index 短期MA | `color='orange'` | **orange** (`#FFA500`) | linewidth=2, zorder=2 |
| Peaks (Tops) マーカー | `'r^'` | **red** (`#FF0000`) | 三角形(上), markersize=10, zorder=3 |
| Troughs (Bottoms) マーカー | `'bv'` | **blue** (`#0000FF`) | 三角形(下), markersize=10, zorder=3 |
| Troughs (短期MA < 0.4) | `color='purple', marker='v'` | **purple** (`#800080`) | 逆三角形, s=150, zorder=3 |
| Average Peaks 水平線 | `color='red', linestyle='--'` | **red** (`#FF0000`) | 破線, linewidth=2 |
| Average Troughs 水平線 | `color='blue', linestyle='--'` | **blue** (`#0000FF`) | 破線, linewidth=2 |
| Bearish Signal 背景 | Panel 1 と同一 | **薄ピンク** | Panel 1 と同期 |
| グリッド | `grid(True)` | — | デフォルトグレー |
| タイトル | 動的（MA期間を含む） | — | fontsize=16, pad=20 |

### 2.4 凡例

**Panel 1 凡例:**
- S&P 500 Price
- Bearish Signal (MA200 Down & Short MA < MA200) — カスタムパッチ（薄ピンク背景）
- Troughs ({short_ma}MA < 0.4) on S&P 500

**Panel 2 凡例:**
- Breadth Index (200-Day MA)
- Breadth Index ({short_ma}-Day MA)
- Peaks (Tops)
- Troughs (Bottoms)
- Troughs ({short_ma}MA < 0.4)
- Average Peaks (200MA) = {値}
- Average Troughs ({short_ma}MA < 0.4) = {値}

---

## 3. Plotly 移行設計

### 3.1 色のマッピング（現行 → Plotly）

| 要素 | Matplotlib | Plotly 指定値 |
|---|---|---|
| S&P 500 ライン | `'cyan'` | `'#00FFFF'` |
| Breadth 200MA | `'green'` | `'#008000'` |
| Breadth 短期MA | `'orange'` | `'#FFA500'` |
| Peaks マーカー | `'red'` | `'#FF0000'` |
| Troughs マーカー | `'blue'` | `'#0000FF'` |
| Trough (< 0.4) | `'purple'` | `'#800080'` |
| Average Peaks 線 | `'red'` | `'#FF0000'` |
| Average Troughs 線 | `'blue'` | `'#0000FF'` |
| Bearish 背景 | `(1.0, 0.9, 0.96), a=0.3` | `'rgba(255, 230, 245, 0.3)'` |
| チャート背景 | `'white'` | `'white'` |

### 3.2 サブプロット構成

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.5, 0.5],
    subplot_titles=['S&P 500 Price', f'S&P 500 Breadth Index with 200-Day MA and {short_ma}-Day MA'],
)
```

### 3.3 Panel 1: S&P 500 Price の Plotly 実装

```python
# S&P 500 価格ライン
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

# Trough マーカー (S&P 上)
fig.add_trace(
    go.Scatter(
        x=s_and_p_troughs.index,
        y=s_and_p_troughs.values,
        name=f'Troughs ({short_ma}MA < 0.4) on S&P 500',
        mode='markers',
        marker=dict(color='#800080', size=12, symbol='triangle-down'),
    ),
    row=1, col=1,
)

# Y軸を対数スケールに
fig.update_yaxes(type='log', row=1, col=1, title_text='Price')
```

### 3.4 Panel 2: Breadth Index の Plotly 実装

```python
# Breadth Index 200-Day MA
fig.add_trace(
    go.Scatter(
        x=breadth_ma_200.index,
        y=breadth_ma_200.values,
        name='Breadth Index (200-Day MA)',
        line=dict(color='#008000', width=2),
    ),
    row=2, col=1,
)

# Breadth Index 短期MA
fig.add_trace(
    go.Scatter(
        x=breadth_ma_short.index,
        y=breadth_ma_short.values,
        name=f'Breadth Index ({short_ma}-Day MA)',
        line=dict(color='#FFA500', width=2),
    ),
    row=2, col=1,
)

# Peaks マーカー
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

# Troughs マーカー
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

# Troughs (短期MA < 0.4) マーカー
fig.add_trace(
    go.Scatter(
        x=below_04.index[troughs_below_04],
        y=below_04.iloc[troughs_below_04].values,
        name=f'Troughs ({short_ma}MA < 0.4)',
        mode='markers',
        marker=dict(color='#800080', size=12, symbol='triangle-down'),
    ),
    row=2, col=1,
)

# Average Peaks 水平線
fig.add_hline(
    y=peaks_avg, row=2, col=1,
    line=dict(color='#FF0000', dash='dash', width=2),
    annotation_text=f'Avg Peaks = {peaks_avg:.2f}',
    annotation_position='bottom right',
)

# Average Troughs 水平線
fig.add_hline(
    y=troughs_avg_below_04, row=2, col=1,
    line=dict(color='#0000FF', dash='dash', width=2),
    annotation_text=f'Avg Troughs = {troughs_avg_below_04:.2f}',
    annotation_position='top right',
)

# Y軸範囲
fig.update_yaxes(range=[0, 1], row=2, col=1, title_text='Breadth Index Percentage')
```

### 3.5 Bearish Signal 背景（両パネル共通）

```python
# Bearish signal の連続区間を検出してまとめる
bearish_mask = (breadth_ma_200_trend == -1) & (breadth_ma_short < breadth_ma_200)
bearish_starts = []
bearish_ends = []

in_bearish = False
for i in range(len(bearish_mask)):
    if bearish_mask.iloc[i] and not in_bearish:
        bearish_starts.append(breadth_ma_short.index[i])
        in_bearish = True
    elif not bearish_mask.iloc[i] and in_bearish:
        bearish_ends.append(breadth_ma_short.index[i])
        in_bearish = False
if in_bearish:
    bearish_ends.append(breadth_ma_short.index[-1])

# vrect で背景を描画（Panel 1 と Panel 2 の両方に適用）
for start, end in zip(bearish_starts, bearish_ends):
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor='rgba(255, 230, 245, 0.3)',
        line_width=0,
        row='all', col=1,
    )
```

**注意:** Matplotlib では1日単位で `axvspan` をループしているが、
Plotly では連続区間をまとめて `vrect` にすることで描画パフォーマンスを大幅に改善する。

### 3.6 レイアウト設定

```python
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

# グリッド線
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
```

### 3.7 レンジセレクター（新機能）

```python
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
```

### 3.8 HTML 出力

```python
fig.write_html(
    'reports/market_breadth.html',
    include_plotlyjs=True,   # JS をファイルに埋め込み（オフライン対応）
    full_html=True,
    config={
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
    },
)
```

---

## 4. ファイル変更一覧

### 4.1 変更対象ファイル

| ファイル | 変更内容 | 影響度 |
|---|---|---|
| `market_breadth.py` | `plot_breadth_and_sp500_with_peaks()` 関数をPlotlyに書き換え | **大** |
| `reports/index.html` | `<img>` → `<iframe>` に変更、フォールバック修正 | **中** |
| `requirements.txt` | `plotly` を追加 | **小** |
| `.github/workflows/daily-market-breadth.yml` | 変更不要（`pip install -r requirements.txt` で plotly も入る） | **なし** |

### 4.2 変更しないファイル

| ファイル | 理由 |
|---|---|
| `fmp_data_fetcher.py` | データ取得に変更なし |
| `extract_chart_data()` | データ加工ロジックはそのまま利用 |
| `export_chart_data_to_csv()` | CSV出力に変更なし |
| `main()` | 引数解析・データフローに変更なし（`plot_*` 呼び出しはそのまま） |
| `backtest/` | 別モジュール、Matplotlib のまま |
| `trade/` | チャートに依存しない |

---

## 5. `market_breadth.py` の変更詳細

### 5.1 import の変更

```python
# 削除
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.patches import Patch

# 追加
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### 5.2 削除する関数・コード

| 対象 | 理由 |
|---|---|
| `setup_matplotlib_backend()` | Matplotlib 専用、Plotly では不要 |
| `setup_matplotlib_backend()` の呼び出し（47行目） | 同上 |
| `plt.rcParams` の設定（367-372行目） | Plotly ではレイアウト設定で制御 |

### 5.3 `plot_breadth_and_sp500_with_peaks()` 関数の書き換え

**関数シグネチャ: 変更なし**
```python
def plot_breadth_and_sp500_with_peaks(above_ma_200, sp500_data, short_ma_period=10, start_date=None, end_date=None):
```

**処理フロー:**

1. `extract_chart_data()` を呼び出し（変更なし）
2. Bearish signal の連続区間を検出（新規ロジック、3.5節参照）
3. `make_subplots()` でFigure生成
4. Panel 1 の要素を追加（3.3節参照）
5. Panel 2 の要素を追加（3.4節参照）
6. Bearish 背景を追加（3.5節参照）
7. レイアウト設定（3.6節参照）
8. レンジセレクター設定（3.7節参照）
9. HTML 出力（3.8節参照）
10. PNG 出力も並行して生成（後方互換性のため、下記参照）

### 5.4 PNG 出力の並行生成（後方互換性）

既存の index.html のフォールバックや、ブログ記事での画像参照に対応するため、
Plotly の `write_image()` で PNG も同時に生成する。

```python
# PNG 出力（後方互換性）
try:
    fig.write_image('reports/market_breadth.png', width=1200, height=900, scale=2)
    print("PNG chart saved to reports/market_breadth.png")
except Exception as e:
    print(f"PNG export skipped (kaleido not installed): {e}")
```

**注意:** `kaleido` パッケージが必要。GitHub Actions 環境では `requirements.txt` に追加して対応。
ローカル環境で `kaleido` がない場合は PNG をスキップし、HTML のみ生成する。

---

## 6. `reports/index.html` の変更詳細

### 6.1 変更方針

- メインコンテンツを `<img>` から `<iframe>` に変更
- HTML ファイルが存在しない場合は PNG にフォールバック
- Last Update の取得ロジックはそのまま維持

### 6.2 変更後の HTML 構造

```html
<div class="chart-container">
    <div class="error-message" id="error-message">
        Failed to load the chart. Please try again later.
    </div>
    <!-- Primary: Interactive HTML chart -->
    <iframe id="chart-iframe"
            style="width:100%; height:950px; border:none; display:none;"
            src="market_breadth.html">
    </iframe>
    <!-- Fallback: Static PNG image -->
    <img id="chart-image" alt="Market Breadth Analysis Chart" style="display:none;">
</div>
```

### 6.3 JavaScript ロード順序

```
1. market_breadth.html を <iframe> で読み込み試行
   ↓ 成功 → iframe を表示、Last Update を更新
   ↓ 失敗
2. market_breadth.png を <img> で読み込み試行
   ↓ 成功 → img を表示、Last Update を更新
   ↓ 失敗
3. エラーメッセージを表示
```

---

## 7. `requirements.txt` の変更

```diff
  requests>=2.31.0
  pandas>=1.5.0
  matplotlib>=3.5.0
  scipy>=1.9.0
  python-dotenv>=1.0.0
  tqdm>=4.64.0
  lxml>=4.9.0
+ plotly>=5.18.0
+ kaleido>=0.2.1
```

---

## 8. テスト計画

### 8.1 ローカルテスト

| # | テスト項目 | 確認内容 |
|---|---|---|
| 1 | HTML ファイル生成 | `reports/market_breadth.html` が生成される |
| 2 | PNG ファイル生成 | `reports/market_breadth.png` が生成される（kaleido がある場合） |
| 3 | 色の一致 | 各要素の色が 2.2, 2.3 節の定義と一致 |
| 4 | Bearish 背景 | ピンク背景が Panel 1, Panel 2 の両方で同期して表示 |
| 5 | マーカー | Peaks, Troughs が正しい位置・色・形状で表示 |
| 6 | 対数スケール | Panel 1 の Y軸が対数スケール |
| 7 | 水平線 | Average Peaks/Troughs の破線が正しい値に表示 |
| 8 | レンジセレクター | 1Y/3Y/5Y/ALL ボタンで期間が切り替わる |
| 9 | ホバー | X unified モードで日付・値が表示される |
| 10 | ズーム/パン | ドラッグでズーム、X軸連動でパン |
| 11 | オフライン | HTML をローカルで開いてインタラクションが動作する |
| 12 | コマンドライン引数 | `--short_ma 5/8/10/20` で正しく動作 |
| 13 | `--use_saved_data` | 保存済みデータでチャートが生成される |
| 14 | `--export_csv` | CSV出力が影響を受けない |

### 8.2 GitHub Actions テスト

| # | テスト項目 | 確認内容 |
|---|---|---|
| 1 | ワークフロー実行 | `workflow_dispatch` で手動実行が成功 |
| 2 | GitHub Pages 表示 | `index.html` から `market_breadth.html` が iframe で表示 |
| 3 | フォールバック | HTML が壊れた場合に PNG にフォールバック |

---

## 9. 移行手順

### Step 1: `requirements.txt` に `plotly` と `kaleido` を追加

### Step 2: `market_breadth.py` の修正
- import 文の変更
- `setup_matplotlib_backend()` の削除
- `plot_breadth_and_sp500_with_peaks()` の Plotly 書き換え

### Step 3: `reports/index.html` の修正
- `<iframe>` + `<img>` フォールバック構成に変更

### Step 4: ローカルテスト
- `python market_breadth.py --use_saved_data` で動作確認
- 生成された HTML をブラウザで開いて色・デザインを確認

### Step 5: GitHub Actions テスト
- `workflow_dispatch` で手動実行
- GitHub Pages の表示を確認

### Step 6: 旧 PNG ファイルのクリーンアップ（任意）
- `reports/` 内の不要な過去 PNG を整理

---

## 10. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| HTML ファイルサイズ（約5MB） | GitHub Pages 表示が遅くなる可能性 | `include_plotlyjs='cdn'` でCDN参照に切り替え可能（オフライン非対応になる） |
| `kaleido` が GitHub Actions で動かない | PNG フォールバックが生成されない | try/except で PNG 生成をオプショナルにする |
| Bearish 背景の `vrect` が多数 | 描画パフォーマンス低下 | 連続区間をまとめて vrect 数を最小化（3.5節の実装） |
| 既存の backtest チャートへの影響 | backtest も Matplotlib 依存 | `backtest/` は変更しない。Matplotlib は requirements に残す |
| Plotly の対数スケールの目盛り | Matplotlib と完全一致しない可能性 | `fig.update_yaxes(dtick=...)` で近似調整。厳密一致は不要 |
