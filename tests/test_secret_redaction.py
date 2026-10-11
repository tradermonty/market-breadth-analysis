import logging

import pytest
import requests

import secret_redaction as sr
from fmp_data_fetcher import FMPDataFetcher
from secret_redaction import REDACTED, redact, register_secret

# Dummy secrets only; keep them out of real environments. Each is >=8 chars and not a
# known placeholder so register_secret accepts them.
FMP_DUMMY = 'FMPDUMMYKEY123'  # pragma: allowlist secret
ALPACA_DUMMY = 'ALPADUMMYKEY456'  # pragma: allowlist secret
TOKEN_DUMMY = 'GHDUMMYTOKEN789'  # pragma: allowlist secret
URL = f'https://financialmodelingprep.com/api/v3/foo?apikey={FMP_DUMMY}'


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isolate tests: each one starts with a clean registered-secret set."""
    sr._registry.clear()
    yield
    sr._registry.clear()


class TestRedactUnit:
    def test_masks_apikey_query_param(self):
        assert FMP_DUMMY not in redact(URL)
        assert REDACTED in redact(URL)

    def test_masks_authorization_header(self):
        text = f'Authorization: Bearer {TOKEN_DUMMY}'
        result = redact(text)
        assert TOKEN_DUMMY not in result
        assert 'Authorization: Bearer' in result
        assert REDACTED in result

    def test_masks_registered_secret_anywhere(self):
        register_secret(ALPACA_DUMMY)
        assert ALPACA_DUMMY not in redact(f'plain log {ALPACA_DUMMY} suffix')
        assert REDACTED in redact(f'plain log {ALPACA_DUMMY} suffix')

    def test_masks_multiple_occurrences_of_secret(self):
        register_secret(FMP_DUMMY)
        text = f'{FMP_DUMMY} then again {FMP_DUMMY}'
        result = redact(text)
        assert FMP_DUMMY not in result
        assert result.count(REDACTED) == 2

    def test_masks_multiple_registered_secrets(self):
        register_secret(ALPACA_DUMMY)
        register_secret(TOKEN_DUMMY)
        result = redact(f'{ALPACA_DUMMY}|{TOKEN_DUMMY}')
        assert ALPACA_DUMMY not in result
        assert TOKEN_DUMMY not in result

    def test_plain_text_unchanged(self):
        assert redact('no secrets here at all') == 'no secrets here at all'

    def test_none_returns_empty(self):
        assert redact(None) == ''

    def test_non_string_returns_empty(self):
        assert redact(123) == ''
        assert redact(['a', 'b']) == ''

    def test_empty_string_returns_empty(self):
        assert redact('') == ''

    def test_register_rejects_short_and_placeholder(self):
        # 'demo' must not be registered, so ordinary text containing it is untouched.
        register_secret('demo')
        assert 'demo' in redact('demonstrating a demo feature')
        register_secret('short')
        assert 'short' in redact('a short string')


class TestFmpRedaction:
    def _fetcher(self):
        return FMPDataFetcher(api_key=FMP_DUMMY)

    def test_http_error_url_masked(self, caplog):
        fetcher = self._fetcher()
        error = requests.exceptions.HTTPError(
            f'400 Client Error: Bad Request for url: {URL}', response=requests.Response()
        )

        def _boom(*args, **kwargs):
            raise error

        fetcher.session.get = _boom
        caplog.set_level(logging.INFO, logger='fmp_data_fetcher')
        result = fetcher._make_request('foo', max_retries=3)
        assert result is None
        assert FMP_DUMMY not in caplog.text
        assert REDACTED in caplog.text

    def test_read_timeout_masked(self, caplog):
        fetcher = self._fetcher()
        error = requests.exceptions.ReadTimeout(f'Read timed out for url: {URL}')

        def _boom(*args, **kwargs):
            raise error

        fetcher.session.get = _boom
        caplog.set_level(logging.INFO, logger='fmp_data_fetcher')
        result = fetcher._make_request('foo', max_retries=3)
        assert result is None
        assert FMP_DUMMY not in caplog.text

    def test_exhausted_retries_debug_masked(self, caplog):
        fetcher = self._fetcher()
        error = requests.exceptions.ConnectionError(f'Connection aborted: {URL}')

        def _boom(*args, **kwargs):
            raise error

        fetcher.session.get = _boom
        caplog.set_level(logging.DEBUG, logger='fmp_data_fetcher')
        result = fetcher._make_request('foo', max_retries=0)
        assert result is None
        assert FMP_DUMMY not in caplog.text
        assert 'after 0 retries' in caplog.text


class TestGithubRedaction:
    def test_request_exception_message_redacted(self):
        from trigger_market_breadth import fetch_market_breadth

        register_secret(TOKEN_DUMMY)

        def _boom(*args, **kwargs):
            raise requests.exceptions.RequestException(f'Connection aborted: {URL}&token={TOKEN_DUMMY}')

        import trigger_market_breadth

        trigger_market_breadth.requests.head = _boom
        result = fetch_market_breadth()
        assert result['status'] == 'error'
        assert TOKEN_DUMMY not in result['message']
        assert FMP_DUMMY not in result['message']


class TestAlpacaRedaction:
    def test_broker_exception_redacted_in_logs(self, caplog):
        from unittest.mock import Mock, patch

        import trade.run_market_breadth_trade as trade_mod

        register_secret(ALPACA_DUMMY)
        with patch.object(trade_mod.MarketBreadthTrader, '_initialize_alpaca', return_value=Mock()):
            trader = trade_mod.MarketBreadthTrader(
                short_ma=8, long_ma=200, initial_capital=50000, symbol='SSO', use_saved_data=True
            )
        trader.testmode = False

        def _submit_order(*args, **kwargs):
            raise RuntimeError(f'order failed for secret {ALPACA_DUMMY}')

        trader.api.submit_order = _submit_order
        caplog.set_level(logging.ERROR, logger='market_breadth_trade')
        result = trader.execute_buy(10, reason='test')
        assert result is None
        assert ALPACA_DUMMY not in caplog.text
        assert REDACTED in caplog.text
