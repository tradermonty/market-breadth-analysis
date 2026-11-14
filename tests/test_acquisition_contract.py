"""Fake-transport provider boundary coverage; never tests provider availability."""

from unittest.mock import Mock

import pytest
import requests

from fmp_data_fetcher import FMPDataFetcher


@pytest.fixture
def fetcher(monkeypatch):
    client = FMPDataFetcher(api_key='synthetic-acquisition-key')  # pragma: allowlist secret
    monkeypatch.setattr(client, '_rate_limit_check', lambda: None)
    monkeypatch.setattr('fmp_data_fetcher.time.sleep', lambda *_: None)
    monkeypatch.setattr('fmp_data_fetcher.time.time', lambda: 1000.5)
    return client


def response(payload, code=200):
    result = Mock(status_code=code)
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def test_01_transport_retry_and_recovery(fetcher):
    fetcher.session.get = Mock(side_effect=[requests.Timeout('fixture timeout'), response([{'symbol': 'AAPL'}])])
    assert fetcher.get_sp500_constituents() == ['AAPL']
    assert fetcher.session.get.call_count == 2


def test_02_exhausted_transport_is_redacted(fetcher, caplog):
    fetcher.session.get = Mock(side_effect=requests.ConnectionError('url?apikey=synthetic-acquisition-key'))
    assert fetcher._make_request('fixture', max_retries=1) is None
    assert fetcher.session.get.call_count == 2
    assert 'synthetic-acquisition-key' not in caplog.text


@pytest.mark.parametrize('code', [403, 404, 429])
def test_03_http_failures_are_non_data(fetcher, code):
    fetcher.session.get = Mock(return_value=response({}, code))
    assert fetcher._make_request('fixture', max_retries=0) is None


@pytest.mark.parametrize('payload', [None, [], {'Error Message': 'fixture unavailable'}])
def test_04_invalid_responses_are_non_data(fetcher, payload):
    fetcher.session.get = Mock(return_value=response(payload))
    assert fetcher._make_request('fixture') is None


@pytest.mark.parametrize(
    'payload',
    [
        {'historical': [{'date': '2024-01-02', 'close': 100}]},
        {'results': [{'date': '2024-01-02', 'close': 100}]},
        [{'date': '2024-01-02', 'close': 100}],
    ],
)
def test_05_historical_response_contracts(fetcher, payload):
    fetcher.session.get = Mock(return_value=response(payload))
    assert fetcher.get_historical_price_data('SPY', '2024-01-02', '2024-01-02') == [
        {'date': '2024-01-02', 'close': 100}
    ]


def test_06_429_retry_recovers(fetcher):
    fetcher.session.get = Mock(side_effect=[response({}, 429), response({'historical': []})])
    assert fetcher._make_request('fixture', max_retries=1) == {'historical': []}
    assert fetcher.rate_limiting_active
