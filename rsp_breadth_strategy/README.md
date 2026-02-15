# RSP Breadth Strategy

RSP主軸 + Market Breadth戦略の独立実装。
このフォルダ単体でテスト/実行できる構成にしているため、将来的に別リポジトリへ切り出しやすい。

## Structure

- `src/rsp_breadth_strategy/signals.py`: シグナル生成
- `src/rsp_breadth_strategy/weights.py`: 配分ロジック
- `src/rsp_breadth_strategy/backtest.py`: 月次バックテスト
- `src/rsp_breadth_strategy/cli.py`: CLI実行
- `tests/`: ユニットテスト

## Quick Start

```bash
cd rsp_breadth_strategy
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

インストールせずに実行する場合:

```bash
cd rsp_breadth_strategy
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## CLI

```bash
cd rsp_breadth_strategy
PYTHONPATH=src python3 -m rsp_breadth_strategy.cli \
  --breadth-csv /path/to/market_breadth_data.csv \
  --prices-csv /path/to/prices.csv \
  --output-dir ./outputs
```

`prices.csv` は以下列を含む想定:

- `Date`
- `SPY`, `RSP`, `XLE`, `SGOV` (終値)
