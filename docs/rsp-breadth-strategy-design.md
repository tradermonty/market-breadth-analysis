# RSP主軸 + Market Breadth 戦略 設計書

## 1. 概要

### 1.1 目的
RSP（等ウェイトS&P500）を主軸に、Market Breadth を使ってリスク量と配分を調整し、
以下の両立を目指す。

- メガキャップ集中リスクの抑制
- 相対優位（RSP/SPY, XLE/SPY）の活用
- 株式急変時のドローダウン抑制

### 1.2 対象データ
- Breadth CSV: `market_breadth_data.csv`
- 利用列: `Date`, `Breadth_Index_Raw`, `Breadth_Index_8MA`, `Breadth_Index_200MA`
- 価格データ: `SPY`, `RSP`, `XLE`, `SGOV`（キャッシュ代替）

### 1.3 本設計の位置づけ
- まずは月次リバランスのシンプル版を実装する
- 日次シグナル最適化やオプションヘッジは第2フェーズで追加する

---

## 2. 戦略コンセプト

### 2.1 コア思想
- `RSP` をコア資産とする
- `XLE` はサテライトとして限定的に使う
- Breadth は「攻める/守る」の切り替えスイッチとして使う

### 2.2 なぜRSP主軸か
- SPY比で銘柄集中度を下げやすい
- メガキャップ偏重修正局面で相対優位を取りやすい
- ただし下落耐性そのものは高くないため、Breadthでサイズ調整する

---

## 3. シグナル定義

### 3.1 相対系列
- `R_ratio_t = Close(RSP)_t / Close(SPY)_t`
- `X_ratio_t = Close(XLE)_t / Close(SPY)_t`

月次終値ベースで以下を定義する。

- `R_trend_t = 1 if R_ratio_t > SMA10(R_ratio)_t else 0`
- `X_trend_t = 1 if X_ratio_t > SMA10(X_ratio)_t else 0`

### 3.2 Breadth レジーム
- `Raw_t = Breadth_Index_Raw_t`
- `MA8_t = Breadth_Index_8MA_t`
- `MA200_t = Breadth_Index_200MA_t`

レジームは次の4分類とする。

1. `Recovery`:
`Raw_t < MA200_t`
2. `RiskOn`:
`Raw_t >= MA200_t` かつ `MA8_t >= MA200_t` かつ `Raw_t <= 0.75`
3. `Overheat`:
`Raw_t > 0.75` かつ `Raw_t < MA8_t`
4. `Deterioration`:
上記以外（中立から悪化へ移る過程）

---

## 4. 配分ルール（ベース）

### 4.1 レジーム別ベース配分
合計100%（未投資は `SGOV` へ）

| Regime | SPY | RSP | XLE | SGOV |
|---|---:|---:|---:|---:|
| Recovery | 25 | 45 | 10 | 20 |
| RiskOn | 45 | 40 | 10 | 5 |
| Overheat | 40 | 30 | 0 | 30 |
| Deterioration | 35 | 35 | 5 | 25 |

### 4.2 トレンド補正
- `R_trend=0` の場合: `RSP -15%`、`SPY +10%`、`SGOV +5%`
- `X_trend=0` の場合: `XLE` を `0%` にし、差分を `SGOV` に移す
- `R_trend=1` かつ `X_trend=1` かつ `Recovery` の場合:
`SPY -5%`, `RSP +3%`, `XLE +2%`

### 4.3 上限・下限
- `RSP` 上限: 55%
- `XLE` 上限: 15%
- `SGOV` 下限: 5%（Overheat/Deteriorationでは下限20%）

---

## 5. 執行ルール

1. 判定タイミング: 月末終値
2. 執行タイミング: 翌営業日寄り（または同値近似）
3. リバランス閾値: 目標比率との乖離が2%以上の資産のみ売買
4. 手数料・スリッページ: バックテストで必ず控除

---

## 6. リスク管理

### 6.1 ハードルール
- ポートフォリオのピーク比ドローダウンが `-12%` を超えた場合:
翌月は `RSP -10%`, `XLE -5%`, `SGOV +15%`
- `XLE` は単独で月次 `-12%` を下回ったら翌月は `XLE=0%`

### 6.2 レジーム切替の保守条件
ダマシ対策として、レジーム切替は以下を採用。

- 原則: 月末値で確定
- Overheat 解除: `Raw <= 0.70` を確認して解除

---

## 7. バックテスト設計

### 7.1 テスト期間
- In-sample: 2016-02-17 から 2021-12-31
- Out-of-sample: 2022-01-01 から最新

### 7.2 ベンチマーク
- `SPY`（Buy & Hold）
- `RSP`（Buy & Hold）
- 参考: `60/40`（SPY/SGOV）

### 7.3 評価指標
- CAGR
- 最大ドローダウン（MDD）
- Calmar Ratio
- 年率ボラティリティ
- 月次勝率
- ターンオーバー

### 7.4 感度分析
以下をグリッド検証する。

- 相対トレンドMA: 8, 10, 12か月
- Overheat閾値: 0.72, 0.75, 0.78
- RSP補正量: 10%, 15%, 20%

---

## 8. 実装タスク

1. データ結合モジュール追加
- Breadth CSV と ETF価格を月次で結合する関数を作成

2. シグナル計算モジュール追加
- レジーム判定
- `R_trend`, `X_trend` 計算

3. ウェイト生成ロジック追加
- ベース配分 + 補正 + 上限下限適用

4. バックテスト拡張
- 既存バックテスタに目標ウェイト運用を追加
- 売買コスト控除を統一

5. レポート出力
- 月次ウェイト推移
- レジーム推移
- 累積リターンとドローダウン

---

## 9. 将来拡張（第2フェーズ）

- VIXや実質金利を補助シグナルに追加
- XLEの代替セクター（XLI/XLF）を切替対象に追加
- 週次の部分リバランス（段階執行）
- テールヘッジ（SPYプット）との組み合わせ

---

## 10. 成功基準

本戦略を採用する条件を次とする。

1. Out-of-sampleで `SPY` 比 `MDD` が改善
2. Out-of-sampleで `RSP` 単体比で `Calmar` が改善
3. 年間ターンオーバーが過度（例: 400%超）でない
4. 主要パラメータの小変更で性能が大崩れしない

---

## 11. Baseline パラメータ凍結（2026-02-14 確定）

以下を baseline パラメータとして凍結する。

| パラメータ | 値 | 根拠 |
|---|---|---|
| `rebalance_freq` | `month_end` | 2x2分解で daily 比 +0.016 CAGR, +0.11 Sharpe |
| `rebalance_threshold` | `0.05` | 感度分析でノイズ低減と応答性のバランス |
| `ratio_ma` | `10` | IS/OOS相関が負(rho=-0.32)のため中間値を維持 |
| `overheat_entry` | `0.75` | 同上 |
| `overheat_exit` | `0.70` | Overheat解除のヒステリシス幅 5pt |
| `transaction_cost_bps` | `5` | ETF実コスト推定値 |
| `regime_mode` | `series` | ヒステリシス付きレジーム安定化 |

### Baseline 性能（2021-02 ～ 2025-09）

| 指標 | 値 |
|---|---|
| CAGR | 10.1% |
| Sharpe (geometric) | 0.84 |
| Sharpe (arithmetic) | 0.86 |
| Max Drawdown | -14.5% |
| Calmar | 0.70 |
| Final Equity | 1.560 |
| Rebalance Count | 44 |

---

## 12. パラメータ変更採用ルール

新パラメータの採用には以下の **3条件すべて** を満たすことを要求する。

1. **OOS Sharpe 改善**: OOS 期間の Sharpe が baseline 以上
2. **OOS MDD 悪化なし**: OOS 期間の Max Drawdown が baseline より深くない
3. **IS/OOS 順位相関が非負**: Spearman rho ≥ 0（過適合でないことの証拠）

### 根拠
- 180パラメータの感度分析で IS vs OOS の Spearman rho = **-0.32** (p < 0.001)
- IS最適パラメータ (`overheat=0.70, ratio_ma=5`) と OOS最適パラメータ (`overheat=0.80, ratio_ma=20`) が正反対
- パラメータ最適化よりも頑健性を優先する運用が合理的

### 再現コマンド

```bash
# 感度分析（フル期間）
PYTHONPATH=src python3 scripts/sensitivity_sweep.py \
  --breadth-csv ../reports/market_breadth_data_20260214_ma8.csv \
  --prices-csv ../data/prices_combined.csv

# 感度分析（IS/OOS分割）
PYTHONPATH=src python3 scripts/sensitivity_sweep.py \
  --breadth-csv ../reports/market_breadth_data_20260214_ma8.csv \
  --prices-csv ../data/prices_combined.csv --oos
```

---

## 13. 2x2 分解結果

改善がシグナル安定化由来かリバランス頻度由来かを分離した。

| Signal Mode | Rebalance | CAGR | Sharpe | MaxDD | Equity |
|---|---|---|---|---|---|
| pointwise (OLD) | daily | 9.73% | 0.80 | -14.5% | 1.536 |
| pointwise (OLD) | month_end | 11.31% | 0.91 | -13.6% | 1.640 |
| series (NEW) | daily | 8.29% | 0.71 | -15.5% | 1.445 |
| series (NEW) | month_end | 10.11% | 0.84 | -14.5% | 1.560 |

### 限界寄与

| 効果 | dCAGR | dSharpe |
|---|---|---|
| Signal (series - pointwise) | -0.012 ~ -0.014 | -0.07 ~ -0.09 |
| Rebalance (month_end - daily) | +0.016 ~ +0.018 | +0.11 ~ +0.13 |

**結論**: 月次リバランスが改善の主因（+1.6~1.8% CAGR）。series モードの override は微弱なマイナス寄与。

---

## 14. アブレーション: override 個別効果

`Bearish_Signal` と `Breadth_200MA_Trend` を個別に ON/OFF して CAGR 押し下げ要因を特定。

| 設定 | Bearish | Trend | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|
| both_ON | ON | ON | 10.11% | 0.84 | -14.5% |
| bearish_only | ON | OFF | 10.11% | 0.84 | -14.5% |
| trend_only | OFF | ON | 10.11% | 0.84 | -14.5% |
| neither | OFF | OFF | 10.70% | 0.87 | -13.6% |

### 限界効果

| Override | dCAGR | dSharpe |
|---|---|---|
| Bearish_Signal | -0.003 | -0.017 |
| Breadth_200MA_Trend | -0.003 | -0.017 |

### 所見

- `both_ON` = `bearish_only` = `trend_only` が完全同値 → **両 override が同一期間でのみ発動**しており冗長
- `neither` が最良パフォーマンス（+0.6% CAGR, +0.03 Sharpe, MDD改善 0.9pt）
- override 合計ドラッグは -0.6% CAGR / -0.03 Sharpe
- 現 baseline では override ON を維持（安全側の選択）。ただし override の実効性は疑わしく、今後のデータ蓄積で再評価する

---

## 15. 統合評価スクリプト

全分析パイプラインを一括実行する `scripts/evaluate_all.py` を提供。

```bash
PYTHONPATH=src python3 scripts/evaluate_all.py \
  --breadth-csv ../reports/market_breadth_data_20260214_ma8.csv \
  --prices-csv ../data/prices_combined.csv \
  --output-dir outputs_eval

# 感度分析をスキップする場合
PYTHONPATH=src python3 scripts/evaluate_all.py \
  --breadth-csv ../reports/market_breadth_data_20260214_ma8.csv \
  --prices-csv ../data/prices_combined.csv \
  --skip-sensitivity
```

出力ディレクトリ構成:
```
outputs_eval/
├── baseline/          # CLI backtest (summary.json, backtest.csv, signals.csv)
├── 2x2/              # decomposition_2x2.csv
├── ablation/          # ablation_overrides.csv
└── sensitivity/       # sensitivity_full.csv, sensitivity_IS.csv, sensitivity_OOS.csv
```
