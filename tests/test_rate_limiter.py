"""MJ-007: Deterministic tests for FMPDataFetcher rate limiting using _now() injection."""

import unittest
from datetime import datetime, timedelta

from fmp_data_fetcher import FMPDataFetcher, RateLimitState


class StubFMPFetcher(FMPDataFetcher):
    """Subclass with injectable clock for deterministic testing."""

    def __init__(self, api_key='test-key', start_time=None):  # pragma: allowlist secret
        self._fake_time = start_time or datetime(2026, 1, 1, 12, 0, 0)
        super().__init__(api_key=api_key)

    def _now(self):
        return self._fake_time

    def advance_time(self, **kwargs):
        self._fake_time += timedelta(**kwargs)


class TestRateLimitStateTransitions(unittest.TestCase):
    """Test state transitions: MAX_PERFORMANCE → CONSERVATIVE → MAX_PERFORMANCE."""

    def test_initial_state_is_max_performance(self):
        fetcher = StubFMPFetcher()
        self.assertEqual(fetcher._rate_state, RateLimitState.MAX_PERFORMANCE)
        self.assertTrue(fetcher.max_performance_mode)
        self.assertFalse(fetcher.rate_limiting_active)

    def test_activate_transitions_to_conservative(self):
        fetcher = StubFMPFetcher()
        fetcher._activate_rate_limiting(duration_minutes=5)

        self.assertEqual(fetcher._rate_state, RateLimitState.CONSERVATIVE)
        self.assertTrue(fetcher.rate_limiting_active)
        self.assertFalse(fetcher.max_performance_mode)

    def test_cooldown_expires_after_duration(self):
        fetcher = StubFMPFetcher()
        fetcher._activate_rate_limiting(duration_minutes=5)

        # Still conservative after 4 minutes
        fetcher.advance_time(minutes=4)
        fetcher._rate_limit_check()
        self.assertEqual(fetcher._rate_state, RateLimitState.CONSERVATIVE)

        # Deactivated after 6 minutes
        fetcher.advance_time(minutes=2)
        fetcher._rate_limit_check()
        self.assertEqual(fetcher._rate_state, RateLimitState.MAX_PERFORMANCE)

    def test_cooldown_uses_injected_clock(self):
        """Verify _activate_rate_limiting uses _now(), not datetime.now()."""
        start = datetime(2026, 6, 15, 10, 0, 0)
        fetcher = StubFMPFetcher(start_time=start)
        fetcher._activate_rate_limiting(duration_minutes=5)

        expected_until = start + timedelta(minutes=5)
        self.assertEqual(fetcher.rate_limit_cooldown_until, expected_until)

    def test_multiple_429_resets_cooldown(self):
        fetcher = StubFMPFetcher()
        fetcher._activate_rate_limiting(duration_minutes=5)

        fetcher.advance_time(minutes=3)
        # Second 429 within cooldown resets the timer
        fetcher._activate_rate_limiting(duration_minutes=5)

        # 3 more minutes from second activation — should still be conservative
        fetcher.advance_time(minutes=3)
        fetcher._rate_limit_check()
        self.assertEqual(fetcher._rate_state, RateLimitState.CONSERVATIVE)

        # 3 more minutes — now 6 min after second activation, should expire
        fetcher.advance_time(minutes=3)
        fetcher._rate_limit_check()
        self.assertEqual(fetcher._rate_state, RateLimitState.MAX_PERFORMANCE)


class TestRateLimitCallTracking(unittest.TestCase):
    """Test call timestamp tracking in conservative mode."""

    def test_timestamps_recorded_in_conservative_mode(self):
        fetcher = StubFMPFetcher()
        fetcher._activate_rate_limiting(duration_minutes=5)
        # Ensure enough time since "last request" to avoid sleep
        fetcher.last_request_time = fetcher._fake_time - timedelta(seconds=1)

        fetcher._rate_limit_check()
        self.assertEqual(len(fetcher.call_timestamps), 1)

    def test_timestamps_not_recorded_in_max_performance(self):
        fetcher = StubFMPFetcher()
        fetcher._rate_limit_check()
        self.assertEqual(len(fetcher.call_timestamps), 0)


if __name__ == '__main__':
    unittest.main()
