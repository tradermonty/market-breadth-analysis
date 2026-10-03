"""Offline assembly of the English long-history research page; does not deploy."""

import argparse
import json
from pathlib import Path

from scripts.build_downtrend_forecast import build, summarize_outcomes
from scripts.render_downtrend_forecast_english import render


def assemble(directory, output_dir, exclude_chattering=False):
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
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    parser.add_argument('--exclude-chattering', action='store_true')
    args = parser.parse_args()
    assemble(args.directory, args.output_dir, args.exclude_chattering)
