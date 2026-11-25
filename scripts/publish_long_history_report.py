"""Offline assembly of the English long-history research page; does not deploy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_downtrend_forecast import build, summarize_outcomes
from scripts.render_downtrend_forecast_english import render


def previous_snapshot(directory):
    """Summarize the reference model of a previous cache for the 'changes' section."""
    frame = pd.read_csv(directory / 'current_fixed.csv', parse_dates=['Date'])
    mask = frame['Bearish_Signal'].astype(str).str.lower().eq('true')
    starts = np.flatnonzero(mask & ~mask.shift(1, fill_value=False))
    return {
        'as_of': str(frame.Date.iloc[-1].date()),
        'start': str(frame.Date.iloc[starts[-1]].date()),
        # Same onset eligibility as the study: skip the first 200 observations.
        'onsets': [str(frame.Date.iloc[i].date()) for i in starts if i >= 200],
        'current_count': len(json.loads((directory / 'current_constituents.json').read_text())),
    }


def assemble(directory, output_dir, exclude_chattering=False, previous=None):
    audit = json.loads((directory / 'dataset_audit.json').read_text())
    manifest = json.loads((directory / 'acquisition_manifest.json').read_text())
    membership = json.loads((directory / 'membership_audit.json').read_text())
    models = {}
    for name in ('current_fixed', 'current_eligible', 'historical_reconstructed'):
        target = output_dir / ('long_' + name)
        build(directory / (name + '.csv'), target, exclude_chattering)
        models[name] = json.loads((target / 'downtrend_forecast_statistics.json').read_text())
    result = models['current_fixed']
    result['source'] = 'https://site.financialmodelingprep.com/developer/docs/batch-eod-prices'
    result['long_history_audit'] = {
        'requested': len(manifest),
        'usable': sum(bool(x.get('usable')) for x in manifest),
        'current_count': len(json.loads((directory / 'current_constituents.json').read_text())),
        'membership_anomalies': len(membership['anomalies']),
        'datasets': audit,
        'model_summaries': {name: model['episode_summary'] for name, model in models.items()},
        'current_starts': {name: model['current']['start'] for name, model in models.items()},
        'same_age_summary': summarize_outcomes(
            [e['episode_outcome'] for e in result['events'] if e['same_age_eligible']]
        ),
    }
    stats = output_dir / 'long_current_fixed' / 'downtrend_forecast_statistics.json'
    stats.write_text(json.dumps(result, indent=2))
    render(
        stats,
        output_dir / 'long_current_fixed' / 'downtrend_forecast_input.csv',
        output_dir / 'downtrend_forecast.html',
        previous=previous,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--exclude-chattering', action='store_true')
    parser.add_argument('--previous-directory', type=Path, help='Cache directory of the previous publication')
    args = parser.parse_args()
    previous = previous_snapshot(args.previous_directory) if args.previous_directory else None
    assemble(args.directory, args.output_dir, args.exclude_chattering, previous)
