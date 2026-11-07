# Market Breadth データ取得 & ワークフロートリガー 設計書

## 1. 概要

### 1.1 目的

LLM (Claude等) や外部スクリプトから Market Breadth の最新データをオンデマンドで取得するためのインターフェースを提供する。
GitHub Actions で1日2回自動生成されるCSVデータの鮮度を判定し、古ければワークフローを再トリガーする。

### 1.2 背景

- GitHub Actions (`daily-market-breadth.yml`) が1日2回 (プレマーケット・ポストマーケット) に CSVデータを GitHub Pages へ自動デプロイしている
- LLM がデータを利用する際、CSVが十分新しいかどうかを自動判定する仕組みが必要
- 手動で GitHub UI にアクセスせずにワークフローをトリガーしたい

### 1.3 変更しないもの

- GitHub Actions ワークフロー定義 (`daily-market-breadth.yml`)
- `market_breadth.py` のデータ生成ロジック
- GitHub Pages のデプロイ構成
- 既存の CSVエクスポート形式

---

## 2. アーキテクチャ

### 2.1 データフロー

```
LLM / スクリプト
    │
    ▼
trigger_market_breadth.py
    │
    ├─ GET リクエスト ──▶ GitHub Pages (データCSVとサマリーCSVをダウンロード)
    │
    ├─ [内容が不正] return: status="error", message=...
    │
    ├─ [鮮度OK] return: status="fresh", csv_text=...
    │
    └─ [古い] GET リクエスト ──▶ GitHub API (ワークフロー実行一覧)
        ├─ [未完了または60分以内の実行あり] return: status="pending", message=...
        └─ [なし] POST リクエスト ──▶ GitHub API (workflow_dispatch)
            └─ return: status="triggered", message=...
```

### 2.2 鮮度判定ロジック

```
1. データCSVとサマリーCSVをダウンロード
2. HTML、スキーマ、全行の日付、最新行の値、未来日、サマリーとの一致を検証 → 不正なら "error"（トリガーしない）
3. CSV の最新日付と、market_dates.resolve_market_date() の想定セッションを比較 → 達していれば "fresh"
4. 古い場合、未完了の実行または60分以内に作成された実行があれば "pending"（トリガーしない）
5. それ以外は "triggered" としてワークフローをトリガー
6. Last-Modified は補助情報。max_age_hours を超えると "fresh" のメッセージに警告を付けるだけで、判定には使わない
```

---

## 3. 実装詳細

### 3.1 ファイル構成

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `trigger_market_breadth.py` | **新規** | メインスクリプト |
| `.env.sample` | **修正** | `GITHUB_TOKEN` プレースホルダー追加 |

### 3.2 公開関数

#### `fetch_market_breadth(max_age_hours=12)`

メイン関数。鮮度チェック → フェッチ or トリガーを自動判定。

**引数:**
- `max_age_hours` (float): Last-Modified の警告閾値（時間単位、デフォルト12）。有限の0以上の値のみ

**戻り値:** dict
| キー | 型 | 説明 |
|---|---|---|
| `status` | str | `"fresh"` / `"triggered"` / `"pending"` / `"error"` |
| `data_url` | str | データCSVの URL |
| `summary_url` | str | サマリーCSVの URL |
| `last_modified` | datetime / None | データの最終更新日時 |
| `csv_text` | str | CSVの中身 (status=="fresh" のみ) |
| `summary_text` | str | サマリーCSVの中身 (status=="fresh" のみ) |
| `message` | str | 状態の説明文 |

#### `trigger_workflow()`

GitHub API で `workflow_dispatch` イベントを送信。

**戻り値:** dict with `status`, `message`, `runs_url`

**認証:** `GITHUB_TOKEN` 環境変数 (Fine-grained PAT, Actions: Read and write)

#### `fetch_csv()`

GitHub Pages から CSV をダウンロード。

**戻り値:** dict with `csv_text`, `summary_text`, `last_modified`

### 3.3 CLI インターフェース

```
python trigger_market_breadth.py                  # 自動判定: フェッチ or トリガー
python trigger_market_breadth.py --trigger-only   # 強制トリガー (鮮度チェックなし)
python trigger_market_breadth.py --fetch-only     # フェッチのみ (トリガーなし)
python trigger_market_breadth.py --max-age 6      # Last-Modified の警告閾値を6時間に変更
```

### 3.4 URL 定数

| 用途 | URL |
|---|---|
| データCSV | `https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv` |
| サマリーCSV | `https://tradermonty.github.io/market-breadth-analysis/market_breadth_summary.csv` |
| ワークフロー API | `POST /repos/tradermonty/market-breadth-analysis/actions/workflows/daily-market-breadth.yml/dispatches` |

### 3.5 エラーハンドリング

| エラー | 対応 |
|---|---|
| `GITHUB_TOKEN` 未設定 | `EnvironmentError` を raise、status="error" を返す |
| GitHub Pages への接続失敗 | status="error"、メッセージにエラー詳細を含める |
| CSV の形式やスキーマが不正 | status="error"、トリガーしない |
| 実行状況の確認に失敗 | status="error"、トリガーしない |
| Last-Modified ヘッダーなし | 鮮度は CSV の内容で判定し、警告は出さない |
| ワークフロートリガー失敗 (非204応答) | `raise_for_status()` でHTTPエラーを伝播 |

---

## 4. 認証設定

### 4.1 GitHub Fine-grained Personal Access Token

1. GitHub Settings → Developer settings → Fine-grained tokens → Generate new token
2. Repository access: `tradermonty/market-breadth-analysis` のみ
3. Permissions: **Actions: Read and write**
4. トークンを `.env` に追加:
   ```
   GITHUB_TOKEN=github_pat_xxxx
   ```

### 4.2 依存パッケージ

新規追加なし。`requests` と `python-dotenv` は `requirements.txt` に既存。

---

## 5. LLM からの利用パターン

### 5.1 基本パターン

```python
from trigger_market_breadth import fetch_market_breadth

result = fetch_market_breadth(max_age_hours=12)

if result["status"] == "fresh":
    csv_data = result["csv_text"]
    # pandas 等で解析
    import pandas as pd
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))

elif result["status"] == "triggered":
    print(result["message"])
    # → "Workflow triggered. Data will be ready in ~5 minutes."

elif result["status"] == "error":
    print(result["message"])
```

### 5.2 フェッチのみ (トリガー不要)

```python
from trigger_market_breadth import fetch_csv

result = fetch_csv()
csv_data = result["csv_text"]
```

---

## 6. テスト手順

| # | テスト項目 | コマンド | 期待結果 |
|---|---|---|---|
| 1 | フェッチのみ | `python trigger_market_breadth.py --fetch-only` | CSV データが表示される |
| 2 | 自動判定 (新鮮) | `python trigger_market_breadth.py` | status="fresh"、CSV プレビューが表示 |
| 3 | 自動判定 (古い) | 想定セッションより古いフィードで `python trigger_market_breadth.py` | status="triggered"。最近の実行があれば status="pending" |
| 4 | 強制トリガー | `python trigger_market_breadth.py --trigger-only` | ワークフロー起動メッセージ |
| 5 | GitHub Actions 確認 | GitHub UI でワークフロー実行を確認 | ワークフローが実行中/完了 |
| 6 | 排他オプション | `python trigger_market_breadth.py --trigger-only --fetch-only` | エラーメッセージ |
