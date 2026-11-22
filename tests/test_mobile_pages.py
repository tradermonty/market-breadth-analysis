"""Mobile readability of the published GitHub Pages files.

Static checks run everywhere. Browser checks drive the generated chart at phone and
desktop widths with Playwright and are skipped when Playwright/Chromium is unavailable.
"""

import re
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from market_breadth import plot_breadth_and_sp500_with_peaks
from tests.test_plotly_chart import _make_synthetic_data

ROOT = Path(__file__).resolve().parents[1]
VIEWPORT = '<meta name="viewport" content="width=device-width, initial-scale=1'


def _build_chart(tmpdir, three_panel=False):
    """Two-panel chart by default; three_panel adds the 50MA panel like the published (--include_50ma) chart."""
    above_ma, sp500 = _make_synthetic_data()
    fig, _ = plot_breadth_and_sp500_with_peaks(
        above_ma,
        sp500,
        short_ma_period=8,
        output_dir=tmpdir,
        above_ma_50=above_ma if three_panel else None,
    )
    return fig, sp500, Path(tmpdir) / 'market_breadth.html'


class TestChartHtmlIsResponsive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.fig, cls.sp500, html_path = _build_chart(cls.tmpdir)
        cls.html = html_path.read_text()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_layout_has_no_fixed_width(self):
        self.assertIsNone(self.fig.layout.width)

    def test_02_html_declares_viewport(self):
        self.assertIn(VIEWPORT, self.html)

    def test_03_wrapper_div_is_not_fixed_width(self):
        self.assertIsNone(re.search(r'<div style="[^"]*width:\s*\d+px', self.html))

    def test_04_html_embeds_mobile_script(self):
        self.assertIn('market-breadth-mobile', self.html)

    def test_05_range_selector_offers_2y(self):
        labels = [b.label for b in self.fig.layout.xaxis.rangeselector.buttons]
        self.assertEqual(labels, ['1Y', '2Y', '3Y', '5Y', 'ALL'])


class TestIndexPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / 'reports' / 'index.html').read_text()

    def test_06_index_has_phone_layout_with_fixed_chart_height(self):
        phone = re.search(r'@media \(max-width: 768px\) \{(.*?)\n        \}', self.html, re.S)
        self.assertIsNotNone(phone, 'phone media query missing')
        self.assertRegex(phone.group(1), r'\.chart-container \{[^}]*height: \d+px')

    def test_07_index_links_are_tap_sized_on_phones(self):
        self.assertIn('min-height: 44px', self.html)

    def test_07b_index_links_to_ai_breadth_skill_guide(self):
        link = re.search(r'<a [^>]*href="([^"]+)"[^>]*class="ai-skill-link"[^>]*>', self.html)
        self.assertIsNotNone(link, 'AI analysis link missing')
        self.assertEqual(
            link.group(1), 'https://tradermonty.github.io/claude-trading-skills/en/skills/market-breadth-analyzer/'
        )
        self.assertIn('rel="noopener"', link.group(0))


class TestHighPricePage(unittest.TestCase):
    def test_08_generator_head_declares_viewport_and_phone_css(self):
        from scripts.analyze_high_price_pink import page_head

        head = page_head('Pink Onsets Near Price Highs')
        self.assertIn(VIEWPORT, head)
        self.assertIn('@media(max-width:700px)', head)
        self.assertIn('<title>Pink Onsets Near Price Highs</title>', head)

    def test_09_published_snapshot_declares_viewport(self):
        html = (ROOT / 'reports' / 'downtrend_forecast_high_price.html').read_text()
        self.assertIn(VIEWPORT, html)
        self.assertIn('@media(max-width:700px)', html)


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    return sync_playwright


STATE_JS = """() => {
  const gd = document.querySelector('.plotly-graph-div');
  const fl = gd._fullLayout;
  const last = fl.xaxis3 || fl.xaxis2;
  return {
    scrollWidth: document.documentElement.scrollWidth, innerWidth: innerWidth,
    clientWidth: document.documentElement.clientWidth,
    scrollHeight: document.documentElement.scrollHeight,
    divHeight: gd.getBoundingClientRect().height,
    width: fl.width, height: fl.height, legend: fl.showlegend, drag: fl.dragmode,
    mode: document.documentElement.dataset.mbMode,
    range: last.range.map(v => String(v).slice(0, 10)),
    xAutorange: last.autorange, yAutorange: fl.yaxis.autorange,
    title: fl.annotations[0].text,
    dots: gd.data.filter(t => t.name === 'latest' && t.visible === true).length,
  };
}"""

RELAYOUT_JS = """([key, range]) => {
  const gd = document.querySelector('.plotly-graph-div');
  return Plotly.relayout(gd, {[key]: range}).then(() => new Promise(r => setTimeout(r, 300)));
}"""


@unittest.skipIf(_playwright() is None, 'playwright not installed')
class TestChartInBrowser(unittest.TestCase):
    """Two-panel chart (no 50MA panel)."""

    THREE_PANEL = False
    LINES = 3  # price, 200MA, short MA

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _, sp500, cls.html_path = _build_chart(cls.tmpdir, three_panel=cls.THREE_PANEL)
        cls.last = sp500.index[-1].date()
        cls.last_x = 'xaxis3' if cls.THREE_PANEL else 'xaxis2'
        cls.pw = _playwright()().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as exc:  # browsers not installed
            cls.pw.stop()
            raise unittest.SkipTest(f'chromium unavailable: {exc}') from exc

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _open(self, width, height=844):
        ctx = self.browser.new_context(viewport={'width': width, 'height': height}, is_mobile=width < 769)
        self.addCleanup(ctx.close)
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(self.html_path.as_uri())
        page.wait_for_function('() => document.documentElement.dataset.mbMode !== undefined')
        page.wait_for_timeout(300)
        return page, errors

    def _resize(self, page, width, height=844):
        page.set_viewport_size({'width': width, 'height': height})
        expected = 'mobile' if width <= 768 else 'desktop'
        page.wait_for_function(f'() => document.documentElement.dataset.mbMode === "{expected}"')
        page.wait_for_timeout(600)
        return page.evaluate(STATE_JS)

    def _assert_latest_inside_right_edge(self, state):
        start, end = (date.fromisoformat(v) for v in state['range'])
        self.assertLessEqual(self.last, end, 'latest date clipped off the right edge')
        self.assertGreater(end, self.last, 'latest date flush against the right edge')
        self.assertLess((end - self.last).days, (end - start).days * 0.1, 'too much empty space on the right')
        return start, end

    def test_10_phone_fits_width_and_opens_on_two_years(self):
        page, errors = self._open(390)
        state = page.evaluate(STATE_JS)
        self.assertEqual(errors, [])
        self.assertLessEqual(state['scrollWidth'], state['innerWidth'])
        self.assertFalse(state['legend'])
        self.assertFalse(state['drag'])
        self.assertEqual(state['dots'], self.LINES)
        start, _ = self._assert_latest_inside_right_edge(state)
        self.assertAlmostEqual((self.last - start).days, 730, delta=3)

    def test_11_phone_titles_show_latest_price(self):
        page, _ = self._open(390)
        title = page.evaluate(STATE_JS)['title']
        self.assertIn(f'{self.last.month}/{self.last.day}', title)

    def test_12_range_buttons_stay_anchored_to_latest_date(self):
        page, _ = self._open(390)
        for label in ('ALL', '1Y', '3Y', '2Y'):
            page.locator('g.rangeselector g.button', has_text=label).first.click()
            page.wait_for_timeout(400)
            with self.subTest(button=label):
                self._assert_latest_inside_right_edge(page.evaluate(STATE_JS))

    def test_13_desktop_keeps_legend_and_zoom(self):
        page, errors = self._open(1400, 1000)
        state = page.evaluate(STATE_JS)
        self.assertEqual(errors, [])
        self.assertTrue(state['legend'])
        self.assertEqual(state['drag'], 'zoom')
        self.assertEqual(state['width'], state['clientWidth'])
        self.assertEqual(state['dots'], 0)

    def test_14_phone_double_tap_does_not_reset_to_full_history(self):
        page, _ = self._open(390)
        before = page.evaluate(STATE_JS)['range']
        box = page.locator('.plotly-graph-div .nsewdrag').first.bounding_box()
        page.mouse.dblclick(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        page.wait_for_timeout(600)
        self.assertEqual(page.evaluate(STATE_JS)['range'], before)

    def test_15_phone_page_is_exactly_as_tall_as_the_chart(self):
        page, _ = self._open(390, 860)
        state = page.evaluate(STATE_JS)
        self.assertEqual(state['divHeight'], state['height'], 'chart div keeps its desktop height')
        self.assertLessEqual(state['scrollHeight'], state['height'], 'blank scroll area below the chart')

    def test_16_crossing_the_breakpoint_refits_width(self):
        page, _ = self._open(1400, 1000)
        for width in (390, 1000, 390):
            with self.subTest(width=width):
                state = self._resize(page, width)
                self.assertEqual(state['width'], state['clientWidth'])
                self.assertLessEqual(state['scrollWidth'], state['innerWidth'])
                self.assertEqual(state['divHeight'], state['height'])

    def test_17_desktop_zoom_and_autorange_are_untouched(self):
        page, _ = self._open(1400, 1000)
        self.assertTrue(page.evaluate(STATE_JS)['yAutorange'], 'desktop price axis should autorange')
        window = [(self.last - timedelta(days=180)).isoformat(), self.last.isoformat()]
        page.evaluate(RELAYOUT_JS, [f'{self.last_x}.range', window])
        self.assertEqual(page.evaluate(STATE_JS)['range'], window)

    def test_18_phone_to_desktop_restores_full_history(self):
        page, _ = self._open(390)
        state = self._resize(page, 1400, 1000)
        self.assertTrue(state['xAutorange'])
        self.assertTrue(state['yAutorange'])


class TestThreePanelChartInBrowser(TestChartInBrowser):
    """Three-panel chart, as published (--include_50ma)."""

    THREE_PANEL = True
    LINES = 5  # price, 50MA, short MA (50), 200MA, short MA (200)


if __name__ == '__main__':
    unittest.main()
