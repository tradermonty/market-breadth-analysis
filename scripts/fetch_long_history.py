"""Fetch isolated research caches with a bounded request rate and safe diagnostics."""

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


def fetch_history(directory, env_file):
    load_dotenv(env_file)
    api_key = os.environ['FMP_API_KEY']
    symbols = json.loads((directory / 'universe.json').read_text())
    prices = directory / 'prices'
    prices.mkdir(exist_ok=True)
    lock = threading.Lock()
    next_request = [0.0]

    def request(symbol, start='1990-01-01', end='2026-10-02'):
        with lock:
            wait = max(0, next_request[0] - time.monotonic())
            if wait:
                time.sleep(wait)
            next_request[0] = time.monotonic() + 0.18
        url = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + quote(symbol, safe='')
        try:
            response = requests.get(url, params={'from': start, 'to': end, 'apikey': api_key}, timeout=35)
            if response.status_code != 200:
                return None, {'status': response.status_code}
            payload = response.json()
            rows = payload.get('historical', []) if isinstance(payload, dict) else []
            return rows, {'status': 200, 'response_sha256': hashlib.sha256(response.content).hexdigest()}
        except Exception as error:
            # Do not log HTTP exception strings: they can contain credential-bearing URLs.
            return None, {'error_type': type(error).__name__}

    def one(symbol):
        target = prices / (quote(symbol, safe='') + '.csv.gz')
        metadata_path = target.with_suffix('.json')
        if target.exists() and metadata_path.exists():
            return json.loads(metadata_path.read_text())
        rows, meta = request(symbol)
        if rows is None and (meta.get('status') in (429, 500, 502, 503) or 'error_type' in meta):
            time.sleep(2)
            rows, meta = request(symbol)
        meta['symbol'] = symbol
        if rows and len(rows) == 5000:
            older, old_meta = request(symbol, end='2007-12-31')
            newer, new_meta = request(symbol, start='2008-01-01')
            if older is None or newer is None:
                meta.update({'usable': False, 'reason': 'truncation_check_failed'})
                return meta
            rows = older + newer
            meta['split_request_hashes'] = [old_meta.get('response_sha256'), new_meta.get('response_sha256')]
        if not rows:
            meta.update({'usable': False, 'reason': 'empty_or_failed_history'})
            return meta
        frame = pd.DataFrame(rows)
        if not {'date', 'adjClose', 'close'} <= set(frame.columns):
            meta.update({'usable': False, 'reason': 'missing_adjusted_price'})
            return meta
        frame = frame[[x for x in ('date', 'open', 'high', 'low', 'close', 'adjClose', 'volume') if x in frame]]
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        frame = frame.sort_values('date').drop_duplicates('date')
        numeric = pd.to_numeric(frame['adjClose'], errors='coerce')
        if numeric.isna().any() or not np.isfinite(numeric).all() or not (numeric > 0).all():
            meta.update({'usable': False, 'reason': 'invalid_adjusted_price'})
            return meta
        frame.to_csv(target, index=False, compression='gzip')
        meta.update(
            {
                'usable': True,
                'rows': len(frame),
                'first_date': str(frame.date.iloc[0].date()),
                'last_date': str(frame.date.iloc[-1].date()),
                'cache_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        metadata_path.write_text(json.dumps(meta, indent=2))
        return meta

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'symbol': symbol, 'usable': False, 'error_type': type(error).__name__}
            results.append(result)
            if len(results) % 25 == 0:
                (directory / 'acquisition_manifest.json').write_text(json.dumps(results, indent=2))
                print(
                    f'{len(results)}/{len(symbols)} completed; usable={sum(bool(x.get("usable")) for x in results)}',
                    flush=True,
                )
    (directory / 'acquisition_manifest.json').write_text(json.dumps(results, indent=2))
    print(f'Done: {len(results)} symbols; usable={sum(bool(x.get("usable")) for x in results)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    args = parser.parse_args()
    fetch_history(args.directory, args.env_file)
