"""llms.txt is a generated data dictionary for agents reading the Pages CSVs."""

import hashlib
import json

import pytest

from scripts import publish_market_breadth as pub
from tests.test_publication_gate import DAY, NOW, fixture_outputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('network forbidden'))
    monkeypatch.setattr(pub, '_calendar_bounds', lambda *_: (DAY, DAY))


def run(dest, analyze):
    return pub.publish_run(dest, analyze=analyze, clock=lambda: NOW, source_revision='a' * 40)


def test_01_published_with_successful_run(tmp_path):
    dest = tmp_path / 'site'
    assert run(dest, lambda p: (fixture_outputs(p), None)[1]) == 0
    text = (dest / 'llms.txt').read_text()
    assert text.startswith('# Market Breadth Analysis\n')
    assert f'Latest market date in this build: {DAY}' in text


def test_02_absent_when_publication_fails(tmp_path):
    def broken(path):
        fixture_outputs(path)
        (path / 'market_breadth.html').write_text('not html')

    dest = tmp_path / 'site'
    assert run(dest, broken) == 1
    assert not (dest / 'llms.txt').exists()


def test_03_links_every_published_dataset(tmp_path):
    dest = tmp_path / 'site'
    run(dest, lambda p: (fixture_outputs(p), None)[1])
    text = (dest / 'llms.txt').read_text()
    for name in ('market_breadth_data.csv', 'market_breadth_summary.csv', 'market_date_resolution.json'):
        assert f'({pub.SITE_URL}{name})' in text


def test_04_every_csv_column_is_documented_in_header_order(tmp_path):
    dest = tmp_path / 'site'
    row = fixture_outputs(tmp_path / 'src')
    run(dest, lambda p: (fixture_outputs(p), None)[1])
    text = (dest / 'llms.txt').read_text()
    positions = [text.index(f'- `{column}`: ') for column in row]
    assert positions == sorted(positions)
    assert 'undocumented' not in text


def test_05_short_ma_columns_follow_configuration():
    text = pub.render_llms_txt(['Date', 'Breadth_Index_10MA', 'Is_Trough_10MA_Below_04'], short_ma=10, latest=DAY)
    assert '- `Breadth_Index_10MA`: 10-day EMA' in text
    assert 'undocumented' not in text


def test_06_unknown_column_is_flagged_not_fatal():
    text = pub.render_llms_txt(['Date', 'New_Column'], short_ma=8, latest=DAY)
    assert '- `New_Column`: (undocumented)' in text


def test_07_price_column_is_identified_as_adjusted_spy():
    text = pub.render_llms_txt(['S&P500_Price'], short_ma=8, latest=DAY)
    assert 'SPY' in text and 'adjusted close' in text


def test_08_included_in_validation_hashes(tmp_path):
    dest = tmp_path / 'site'
    run(dest, lambda p: (fixture_outputs(p), None)[1])
    report = json.loads((dest / 'validation.json').read_text())
    assert report['sha256']['llms.txt'] == hashlib.sha256((dest / 'llms.txt').read_bytes()).hexdigest()


def test_09_trend_is_sticky_not_flat():
    text = pub.render_llms_txt(['Breadth_200MA_Trend', 'Breadth_50_MA_Trend'], short_ma=8, latest=DAY)
    assert '0 flat' not in text
    assert text.count('0 only before the first qualifying move') == 2


def test_10_notes_warn_about_universe_and_recompute():
    text = pub.render_llms_txt(['Date'], short_ma=8, latest=DAY)
    assert 'current S&P 500 membership' in text
    assert 'recomputed' in text
