"""Create a new dated long-history cache directory with refreshed index metadata.

The candidate universe of the previous snapshot is kept unchanged and only symbols
touched by membership changes after the previous cutoff are added, so a refreshed
study stays comparable with the archived one. Prices are fetched separately with
``scripts.fetch_long_history --end-date``.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = 'https://financialmodelingprep.com/api/v3/'


def _symbols(record):
    added = str(record.get('symbol') or '').strip()
    removed = str(record.get('removedTicker') or '').strip()
    return added, removed


def build_universe(previous_universe, current, changes, previous_end, end):
    """Previous universe plus current members and symbols changed in (previous_end, end]."""
    universe = set(previous_universe) | {x['symbol'] for x in current}
    for record in changes:
        date = record.get('date') or ''
        if previous_end < date <= end:
            universe.update(s for s in _symbols(record) if s)
    return sorted(universe)


def membership_audit(current, changes, end):
    """Replay membership changes backwards from the current list; record sizes and anomalies."""
    groups = collections.defaultdict(list)
    for record in changes:
        if record.get('date') and '1990-01-01' <= record['date'] <= end:
            groups[record['date']].append(record)
    state = {x['symbol'] for x in current}
    sizes, anomalies = [], []
    for date in sorted(groups, reverse=True):
        additions = {_symbols(x)[0] for x in groups[date]} - {''}
        removals = {_symbols(x)[1] for x in groups[date]} - {''}
        missing = sorted(additions - state)
        if missing:
            anomalies.append([date, missing])
        state -= additions
        state |= removals
        sizes.append([date, len(state)])
    return {'sizes': sizes, 'anomalies': anomalies}


def _get(path, api_key, params=None):
    response = requests.get(BASE + path, params={**(params or {}), 'apikey': api_key}, timeout=60)
    # Do not include the URL in errors: it carries the API key.
    if response.status_code != 200:
        raise RuntimeError(f'{path}: HTTP {response.status_code}')
    return response.json(), hashlib.sha256(response.content).hexdigest()


def refresh(directory, previous_directory, env_file, end):
    load_dotenv(env_file)
    api_key = os.environ['FMP_API_KEY']
    directory.mkdir(parents=True, exist_ok=False)
    current, current_hash = _get('sp500_constituent', api_key)
    changes, changes_hash = _get('historical/sp500_constituent', api_key)
    spy, spy_hash = _get('historical-price-full/SPY', api_key, {'from': '1993-01-01', 'to': end})
    spy['historical'] = [x for x in spy['historical'] if x['date'] <= end]
    previous_universe = json.loads((previous_directory / 'universe.json').read_text())
    previous_end = max(
        x['date'] for x in json.loads((previous_directory / 'spy_legacy.json').read_text())['historical']
    )
    universe = build_universe(previous_universe, current, changes, previous_end, end)
    audit = membership_audit(current, changes, end)
    outputs = {
        'current_constituents.json': current,
        'historical_constituents.json': changes,
        'spy_legacy.json': spy,
        'universe.json': universe,
        'membership_audit.json': audit,
        'metadata_manifest.json': {
            'end': end,
            'previous_directory': previous_directory.name,
            'previous_end': previous_end,
            'response_sha256': {'current': current_hash, 'changes': changes_hash, 'spy': spy_hash},
            'added_symbols': sorted(set(universe) - set(previous_universe)),
        },
    }
    for name, payload in outputs.items():
        (directory / name).write_text(json.dumps(payload, indent=2))
    spy_dates = [x['date'] for x in spy['historical']]
    print(
        json.dumps(
            {
                'current': len(current),
                'changes': len(changes),
                'latest_change': max(x.get('date') or '' for x in changes),
                'spy_rows': len(spy_dates),
                'spy_range': [min(spy_dates), max(spy_dates)],
                'universe': len(universe),
                'added_symbols': outputs['metadata_manifest.json']['added_symbols'],
                'anomalies': len(audit['anomalies']),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--previous-directory', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--end-date', required=True)
    args = parser.parse_args()
    refresh(args.directory, args.previous_directory, args.env_file, args.end_date)
