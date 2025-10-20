# トレードログ機能 設計書

## 概要
バックテストシステムに詳細なトレードログ機能を追加し、各トレードのエントリーからイグジットまでの情報を記録・出力できるようにします。

## 目的
- トレードの詳細な分析を可能にする
- エントリー・イグジットのタイミングと理由を記録
- 各トレードの損益を追跡
- CSV形式でエクスポートして外部ツールで分析可能にする

## 記録する情報

### 各トレードごとに記録する項目
1. **Trade ID** - トレード番号（連番）
2. **Entry Date** - エントリー日時
3. **Entry Price** - エントリー価格
4. **Entry Shares** - エントリー数量
5. **Entry Cost** - エントリー総コスト（手数料・スリッページ込み）
6. **Entry Reason** - エントリー理由（short_ma_bottom, long_ma_bottom, background_color_change）
7. **Exit Date** - イグジット日時
8. **Exit Price** - イグジット価格
9. **Exit Shares** - イグジット数量
10. **Exit Proceeds** - イグジット総収益（手数料・スリッページ控除後）
11. **Exit Reason** - イグジット理由（peak exit, stop loss, background color change）
12. **Holding Days** - 保有日数
13. **Profit/Loss ($)** - 損益（ドル）
14. **Profit/Loss (%)** - 損益率（パーセント）
15. **Cumulative P/L** - 累積損益

## 実装設計

### 1. データ構造の追加

#### Backtestクラスに追加するプロパティ
```python
self.trade_log = []  # 完結したトレードの詳細ログ
self.open_positions = []  # 現在オープンしているポジションの詳細
```

#### trade_logの構造
```python
{
    'trade_id': int,
    'entry_date': pd.Timestamp,
    'entry_price': float,
    'entry_shares': int,
    'entry_cost': float,
    'entry_reason': str,
    'exit_date': pd.Timestamp,
    'exit_price': float,
    'exit_shares': int,
    'exit_proceeds': float,
    'exit_reason': str,
    'holding_days': int,
    'pnl_dollar': float,
    'pnl_percent': float,
    'cumulative_pnl': float
}
```

#### open_positionsの構造
```python
{
    'entry_date': pd.Timestamp,
    'entry_price': float,
    'entry_shares': int,
    'entry_cost': float,
    'entry_reason': str
}
```

### 2. メソッドの変更

#### 2.1 _execute_entry()の変更
- 現在: tradesリストにBUY情報を追加
- 変更後: tradesリストに加えて、open_positionsにも詳細情報を追加

#### 2.2 _execute_exit()の変更
- 現在: tradesリストにSELL情報を追加
- 変更後:
  - open_positionsから対応するエントリー情報を取得
  - エントリーとイグジットの情報を組み合わせて完結したトレードとしてtrade_logに記録
  - FIFOロジックで複数エントリーに対応

#### 2.3 新規メソッドの追加

**_record_completed_trade()**
```python
def _record_completed_trade(self, entry_info, exit_date, exit_price,
                            exit_shares, exit_proceeds, exit_reason):
    """完結したトレードをtrade_logに記録"""
```

**save_trade_log()**
```python
def save_trade_log(self, filename=None):
    """トレードログをCSVファイルとして保存"""
```

**print_trade_log()**
```python
def print_trade_log(self, limit=None):
    """トレードログをコンソールに表示"""
```

### 3. 出力形式

#### 3.1 CSV出力
- ファイル名: `reports/trade_log_{symbol}_{start_date}_{end_date}.csv`
- エンコーディング: UTF-8
- 区切り文字: カンマ

#### 3.2 コンソール出力
- 表形式で見やすく表示
- デフォルトで全トレードを表示
- limitパラメータで表示件数を制限可能

### 4. 複数ポジション対応

現在のシステムは段階的な買い増しに対応しているため、以下のロジックで処理:

1. **エントリー時**: open_positionsリストに追加（複数エントリー可能）
2. **イグジット時**: FIFOロジックで古いエントリーから順にマッチング
3. **部分イグジット**:
   - partial_exitが有効な場合、売却した分だけをtrade_logに記録
   - 残ったポジションはopen_positionsに残る

### 5. 既存機能への影響

#### 影響なし
- 既存のtradesリスト - そのまま維持（後方互換性）
- パフォーマンス計算ロジック - 変更なし
- 可視化機能 - 変更なし

#### 追加される機能
- run()メソッドの最後にtrade_logの自動保存
- calculate_performance()でtrade_logの統計情報を表示

### 6. 使用例

```python
# バックテスト実行
backtest = Backtest(
    start_date='2020-01-01',
    end_date='2024-12-31',
    symbol='SSO',
    # ... その他のパラメータ
)
backtest.run()

# トレードログをコンソールに表示（最新10件）
backtest.print_trade_log(limit=10)

# トレードログをCSVに保存
backtest.save_trade_log()

# トレードログをDataFrameとして取得（さらなる分析用）
trade_df = pd.DataFrame(backtest.trade_log)
```

### 7. テスト計画

1. **単一トレードのテスト**
   - 1回のエントリー・イグジットが正しく記録されるか

2. **複数ポジションのテスト**
   - 段階的な買い増しが正しく記録されるか
   - FIFOロジックが正しく動作するか

3. **部分イグジットのテスト**
   - partial_exitが有効な場合の記録が正しいか

4. **各種イグジット理由のテスト**
   - peak exit
   - stop loss
   - trailing stop
   - background color change

5. **CSV出力のテスト**
   - ファイルが正しく作成されるか
   - データが正しくフォーマットされているか

## 実装の優先順位

### Phase 1 (必須)
1. trade_logとopen_positionsのデータ構造追加
2. _record_completed_trade()メソッドの実装
3. _execute_entry()と_execute_exit()の変更
4. save_trade_log()メソッドの実装

### Phase 2 (推奨)
1. print_trade_log()メソッドの実装
2. run()メソッドでの自動保存機能

### Phase 3 (オプション)
1. トレードログの統計サマリー表示
2. 勝ちトレード/負けトレードの詳細分析

## 注意事項

1. **メモリ使用量**: 長期間のバックテストでは大量のトレードログが蓄積される可能性があるが、通常のバックテスト期間（数年〜10年程度）では問題ないと想定

2. **後方互換性**: 既存のtradesリストは維持し、既存コードへの影響を最小限にする

3. **複数ポジション管理**: 現在のシステムが段階的な買い増しに対応しているため、FIFOロジックで適切に処理する必要がある

4. **パフォーマンス**: ログ記録による処理速度への影響は最小限（1トレードあたり数ミリ秒未満）と想定

## 期待される効果

1. **分析の深化**: 各トレードの詳細を把握し、戦略の改善ポイントを特定
2. **外部ツール連携**: CSV出力により、Excel、Python、Rなどでさらなる分析が可能
3. **監査証跡**: 全トレードの記録により、バックテスト結果の検証が容易
4. **パターン発見**: エントリー・イグジット理由別のパフォーマンス分析が可能
