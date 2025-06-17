import unittest
from datetime import datetime

from market_breadth import get_last_trading_day

class TestGetLastTradingDay(unittest.TestCase):
    def test_previous_weekday_from_weekday(self):
        wednesday = datetime(2024, 6, 19)
        self.assertEqual(get_last_trading_day(wednesday), '2024-06-18')

    def test_previous_weekday_from_monday(self):
        monday = datetime(2024, 6, 17)
        self.assertEqual(get_last_trading_day(monday), '2024-06-14')

    def test_previous_weekday_from_weekend(self):
        sunday = datetime(2024, 6, 16)
        self.assertEqual(get_last_trading_day(sunday), '2024-06-14')

if __name__ == '__main__':
    unittest.main()
