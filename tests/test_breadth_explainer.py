"""Daily plain-English breadth explainer: facts, publish gate, fallbacks and workflow isolation."""

import json
import re
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from scripts import breadth_explainer as bx

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'web' / 'breadth_explainer_example.html'
WORKFLOW = ROOT / '.github' / 'workflows' / 'daily-market-breadth.yml'

ANALYSIS = {
    'data_provenance': {'as_of': '2026-11-20'},
    'metadata': {'data_freshness': {'latest_date': '2026-11-20'}},
    'composite': {'composite_score': 33.5, 'zone': 'Weakening'},
    'components': {'breadth_level_trend': {'current_8ma': 0.5736230139837473}},
}
FACTS = {'data_date': '2026-11-20', 'score': 33.5, 'zone': 'Weakening', 'breadth_pct': 57}


def example():
    return EXAMPLE.read_text()


def inject(snippet):
    return example().replace('</main>', snippet + '</main>')


# ---------------------------------------------------------------- facts


def test_01_extract_facts_from_analyzer_json():
    assert bx.extract_facts(ANALYSIS) == FACTS


@pytest.mark.parametrize(
    'path, value',
    [
        (('composite', 'composite_score'), None),
        (('composite', 'zone'), '<b>x</b>'),
        (('composite', 'composite_score'), 150),
    ],
)
def test_02_extract_facts_rejects_missing_or_bad_fields(path, value):
    broken = json.loads(json.dumps(ANALYSIS))
    broken[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        bx.extract_facts(broken)


# ---------------------------------------------------------------- regeneration decision


def test_03_page_meta_reads_all_fact_tags():
    assert bx.page_meta(example()) == bx.fact_strings(FACTS)
    assert bx.page_data_date('<html><head></head></html>') is None


def test_04_reuse_only_when_published_page_shows_exactly_todays_facts():
    assert bx.needs_regeneration(example(), FACTS) is False
    assert bx.needs_regeneration(bx.with_csp(example()), FACTS) is False


@pytest.mark.parametrize(
    'previous',
    [
        None,
        '',
        bx.placeholder_page(),
        example().replace('content="2026-11-20"', 'content="2026-11-19"'),
        example().replace('name="breadth-score" content="33.5"', 'name="breadth-score" content="34.0"'),
    ],
    ids=['none', 'empty', 'placeholder', 'older-date', 'same-date-new-numbers'],
)
def test_05_regenerate_otherwise(previous):
    assert bx.needs_regeneration(previous, FACTS) is True


# ---------------------------------------------------------------- publish gate


def test_06_reference_example_passes_gate():
    assert bx.validate_explainer(example(), FACTS) == []


def test_07_placeholder_is_safe():
    assert bx.safety_problems(bx.placeholder_page()) == []


BYPASSES = {
    'inline-script': '<script>fetch("/x")</script>',
    'svg-script-href': '<svg><script href="//evil.example/x.js"></script></svg>',
    'script-title-gt': '<script title=">" src="https://evil.example/a.js"></script>',
    'img-protocol-relative': '<img src="//evil.example/p.png">',
    'img-uppercase-scheme': '<img src="HTTPS://evil.example/p.png">',
    'entity-encoded-link': '<a href="&#104;ttps://evil.example/">x</a>',
    'meta-refresh': '<meta http-equiv="refresh" content="0;url=//evil.example">',
    'base-href': '<base href="//evil.example/">',
    'css-url': '<div style="background:url(//evil.example/t.png)">x</div>',
    'style-import': '<style>@import "//evil.example/a.css";</style>',
    'srcset': '<img srcset="//evil.example/a.png 1x">',
    'svg-use': '<svg><use href="//evil.example/s.svg#a"/></svg>',
    'event-handler': '<div onclick="fetch(1)">x</div>',
    'svg-onload': '<svg onload="alert(1)"></svg>',
    'js-url-split': '<a href="java&#10;script:alert(1)">x</a>',
    'frameset': '<frameset><frame src="x"></frameset>',
    'iframe': '<iframe src="x"></iframe>',
    'form': '<form action="https://tradermonty.github.io/"><input></form>',
    'link-stylesheet': '<link rel="stylesheet" href="https://tradermonty.github.io/x.css">',
    'foreign-link': '<a href="https://evil.example/">x</a>',
    'secret': '<p>sk-ant-api03-abcdef</p>',
    # re-review N1: browser/parser differentials and CSS that can fetch
    'svg-style-markup': '<svg><style><img src=x onerror="fetch(1)"></style></svg>',
    'css-escape': '<div style="background:u\\72l(//evil.example/t.png)">x</div>',
    'css-image-set': '<div style="background-image:image-set(\'//evil.example/a.png\' 1x)">x</div>',
    'svg-external-paint': '<svg><rect fill="url(https://evil.example/p.svg#a)"/></svg>',
    'plaintext': '<plaintext>',
    'html-comment': '<!-- note -->',
}


@pytest.mark.parametrize('snippet', BYPASSES.values(), ids=BYPASSES.keys())
def test_08_gate_rejects_unsafe_markup(snippet):
    assert bx.validate_explainer(inject(snippet), FACTS) != []


@pytest.mark.parametrize(
    'mutate, problem',
    [
        (lambda h: h.replace('<span data-fact="score">33.5</span>', '<span data-fact="score">41.0</span>'), 'score'),
        (lambda h: h.replace('<b data-fact="breadth">57</b>', '<b>57</b>'), 'breadth'),
        (lambda h: h.replace('content="Weakening"', 'content="Healthy"'), 'breadth-zone'),
        (lambda h: h.replace('<meta name="breadth-pct" content="57">', ''), 'breadth-pct'),
        (lambda h: h.replace('</head>', '<meta name="breadth-score" content="33.5"></head>'), 'breadth-score'),
        (lambda h: h.replace('datetime="2026-11-20"', 'datetime="2026-11-19"'), 'date'),
        (lambda h: h.replace('Not investment advice', 'Advice'), 'disclaimer'),
        (lambda h: h.replace('href="index.html"', 'href="#top"'), 'dashboard link'),
        (lambda h: h + ' ' * 400_000, 'size'),
    ],
)
def test_09_gate_rejects_wrong_or_missing_facts(mutate, problem):
    problems = bx.validate_explainer(mutate(example()), FACTS)
    assert any(problem in p for p in problems), problems


def test_10_facts_hidden_in_comments_or_title_do_not_count():
    page = example()
    for label in ('score', 'zone', 'breadth'):
        page = re.sub(rf'<(\w+) data-fact="{label}">([^<]*)</\1>', r'\2', page)
    page = page.replace('<title>', '<title>33.5 Weakening 57 ').replace('</main>', '<!-- 33.5 Weakening 57 --></main>')
    problems = bx.validate_explainer(page, FACTS)
    assert sum('no data-fact' in p for p in problems) == 3, problems


def test_11_gate_is_linear_on_pathological_input():
    page = example().replace('</main>', '<script ' * 30_000 + '</main>')
    start = time.monotonic()
    assert bx.validate_explainer(page, FACTS) != []
    assert time.monotonic() - start < 3


def test_12_csp_is_first_in_head_and_blocks_scripts():
    page = bx.with_csp(example())
    head = page[page.index('<head>') : page.index('</head>')]
    assert head.split('\n')[1] == bx.CSP
    assert "default-src 'none'" in bx.CSP and 'script-src' not in bx.CSP


PROLOGUE_TRICKS = {
    # re-review N2: the CSP must never land anywhere but the real <head>
    'comment-wrapped-head': lambda h: '<!-- <head> -->' + h,
    'head-in-title': lambda h: h.replace('<title>', '<title><head>', 1),
    'header-not-head': lambda h: h.replace('<head>', '<header>', 1),
    'markup-before-head': lambda h: h.replace('<head>', '<p>x</p><head>', 1),
    'second-head': lambda h: h.replace('</body>', '<head></head></body>', 1),
}


@pytest.mark.parametrize('mutate', PROLOGUE_TRICKS.values(), ids=PROLOGUE_TRICKS.keys())
def test_12b_gate_and_csp_refuse_ambiguous_head(mutate):
    page = mutate(example())
    assert bx.validate_explainer(page, FACTS) != []
    if not bx.PROLOGUE.match(page):
        with pytest.raises(ValueError):
            bx.with_csp(page)


HIDDEN_FACTS = {
    # re-review N4: facts must be visible to the reader, not just present for the gate
    'hidden-attr': ('<span data-fact="score">33.5</span>', '<span data-fact="score" hidden>33.5</span>'),
    'aria-hidden': ('<span data-fact="score">33.5</span>', '<span data-fact="score" aria-hidden="true">33.5</span>'),
    'display-none': ('<b data-fact="breadth">57</b>', '<b data-fact="breadth" style="display: none">57</b>'),
    'opacity-zero': ('<b data-fact="breadth">57</b>', '<b data-fact="breadth" style="opacity:0">57</b>'),
    'empty-time': ('>Friday, Nov 20, 2026</time>', '></time>'),
    # final review: more ways to show the gate one thing and the reader another
    'time-text-contradicts': ('>Friday, Nov 20, 2026</time>', '>Thursday, Nov 19, 2026</time>'),
    'css-comment-hiding': ('<b data-fact="breadth">57</b>', '<b data-fact="breadth" style="display:/**/none">57</b>'),
    'fact-with-class': ('<b data-fact="breadth">57</b>', '<b data-fact="breadth" class="x">57</b>'),
    'fact-in-svg-desc': ('<b data-fact="breadth">57</b>', '<svg><desc data-fact="breadth">57</desc></svg>57'),
    'fact-in-div': ('<b data-fact="breadth">57</b>', '<div data-fact="breadth">57</div>'),
}


HEAD_TRICKS = {
    # final review N1: <svg> in <head> makes browsers leave the head, so its <style> is markup
    'svg-style-link-in-head': '<svg><style><p><a href="https://evil.example/login">Sign in</a></p></style></svg>',
    'svg-in-head': '<svg></svg>',
    'markup-in-style': '<style>p{}</style><style><a href="https://evil.example/">x</a></style>',
}


@pytest.mark.parametrize('snippet', HEAD_TRICKS.values(), ids=HEAD_TRICKS.keys())
def test_12e_head_accepts_only_meta_title_style(snippet):
    page = example().replace('</head>', snippet + '\n</head>', 1)
    assert bx.validate_explainer(page, FACTS) != []


def test_12f_css_hiding_rules():
    for css in ('.a{color: transparent}', '.a{display:/* x */none}', '.a{opacity:0}', '/* open'):
        page = example().replace('</style>', css + '\n</style>', 1)
        assert bx.validate_explainer(page, FACTS) != [], css


@pytest.mark.parametrize('old, new', HIDDEN_FACTS.values(), ids=HIDDEN_FACTS.keys())
def test_12c_gate_rejects_hidden_or_empty_facts(old, new):
    page = example()
    assert page.count(old) == 1
    assert bx.validate_explainer(page.replace(old, new), FACTS) != []


def test_12d_gate_allows_fractional_sizes():
    page = example().replace('</style>', '.x { opacity: 0.85; font-size: 0.9rem; }\n</style>', 1)
    assert bx.validate_explainer(page, FACTS) == []


# ---------------------------------------------------------------- finalize (explainer-publish job)


def _work(tmp_path, *, facts=FACTS, explainer=None, raw=None, csv='2026-11-20,0.5736'):
    work = tmp_path / 'work'
    work.mkdir(parents=True)
    if facts is not None:
        (work / 'facts.json').write_text(facts if isinstance(facts, str) else json.dumps(facts))
    if explainer is not None:
        (work / 'explainer.html').write_text(explainer)
    if raw is not None:
        (work / 'explainer.html').write_bytes(raw)
    site = tmp_path / 'site'
    site.mkdir(parents=True)
    (site / bx.OUTPUT_NAME).write_text('carried')
    if csv is not None:
        (site / 'market_breadth_data.csv').write_text('Date,Breadth_Index_8MA\n2026-11-19,0.6\n' + csv + '\n')
    return work, site


def test_13_finalize_publishes_valid_page_with_csp(tmp_path):
    work, site = _work(tmp_path, explainer=example())
    assert bx.finalize(work, site) == 'generated'
    assert (site / bx.OUTPUT_NAME).read_text() == bx.with_csp(example())


@pytest.mark.parametrize(
    'kwargs, status',
    [
        ({'explainer': inject('<script>x</script>')}, 'rejected'),
        ({'raw': b'<html>\xff\xfe broken</html>'}, 'rejected'),
        ({}, 'missing'),
        ({'facts': None, 'explainer': '<html></html>'}, 'missing'),
        ({'facts': '{not json', 'explainer': '<html></html>'}, 'error'),
        ({'facts': {'score': 33.5}, 'explainer': '<html></html>'}, 'error'),
    ],
    ids=['unsafe', 'non-utf8', 'no-page', 'no-facts', 'bad-json', 'missing-key'],
)
def test_14_finalize_keeps_carried_page_and_never_raises(tmp_path, kwargs, status):
    work, site = _work(tmp_path, **kwargs)
    assert bx.finalize(work, site) == status
    assert (site / bx.OUTPUT_NAME).read_text() == 'carried'


@pytest.mark.parametrize(
    'last_row, status',
    [('2026-11-20,0.5736', 'generated'), ('2026-11-21,0.5736', 'rejected'), ('2026-11-20,0.61', 'rejected')],
    ids=['agrees', 'other-date', 'other-breadth'],
)
def test_14b_finalize_cross_checks_facts_against_published_csv(tmp_path, last_row, status):
    work, site = _work(tmp_path, explainer=example(), csv=last_row)
    assert bx.finalize(work, site) == status


def test_14c_finalize_fails_closed_without_published_csv(tmp_path):
    work, site = _work(tmp_path, explainer=example(), csv=None)
    assert bx.finalize(work, site) == 'rejected'
    assert (site / bx.OUTPUT_NAME).read_text() == 'carried'


# ---------------------------------------------------------------- carry-forward (build job)


def test_15_carry_forward_keeps_published_page_idempotently(tmp_path):
    published = tmp_path / 'live.html'
    published.write_text(bx.with_csp(example()))
    site = tmp_path / 'site'
    site.mkdir()
    assert bx.carry_forward(site, published.as_uri()) == 'carried'
    page = (site / bx.OUTPUT_NAME).read_text()
    assert page == bx.with_csp(example()) and page.count(bx.CSP) == 1


@pytest.mark.parametrize(
    'live', [None, bx.placeholder_page(), inject('<script>x</script>')], ids=['missing', 'placeholder', 'unsafe']
)
def test_16_carry_forward_falls_back_to_placeholder(tmp_path, live):
    site = tmp_path / 'site'
    site.mkdir()
    url = None
    if live is not None:
        (tmp_path / 'live.html').write_text(live)
        url = (tmp_path / 'live.html').as_uri()
    assert bx.carry_forward(site, url) == 'placeholder'
    assert bx.is_placeholder((site / bx.OUTPUT_NAME).read_text())


def test_17_carry_forward_never_raises(tmp_path):
    assert bx.carry_forward(tmp_path / 'missing-dir', None) == 'error'


# ---------------------------------------------------------------- analyze (explainer job)


STUB_ANALYZER = textwrap.dedent(
    """
    import argparse, json, os, urllib.request
    assert 'SECRET_FOR_TEST' not in os.environ, 'analyzer must run with a scrubbed environment'
    p = argparse.ArgumentParser()
    p.add_argument('--detail-url'); p.add_argument('--summary-url'); p.add_argument('--output-dir')
    a = p.parse_args()
    rows = urllib.request.urlopen(a.detail_url).read().decode().strip().splitlines()
    urllib.request.urlopen(a.summary_url).read()
    last = rows[-1].split(',')
    out = {
        'data_provenance': {'as_of': last[0]},
        'composite': {'composite_score': 33.5, 'zone': 'Weakening'},
        'components': {'breadth_level_trend': {'current_8ma': float(last[1])}},
    }
    os.makedirs(a.output_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.output_dir, 'market_breadth_2026-11-22_080000.json'), 'w'))
    open(os.path.join(a.output_dir, 'market_breadth_2026-11-22_080000.md'), 'w').write('# report')
    """
)


def _skills(tmp_path):
    analyzer = tmp_path / 'mba'
    (analyzer / 'scripts').mkdir(parents=True)
    (analyzer / 'SKILL.md').write_text('---\nname: market-breadth-analyzer\n---\n')
    (analyzer / 'scripts' / 'market_breadth_analyzer.py').write_text(STUB_ANALYZER)
    eli5 = tmp_path / 'eli5'
    eli5.mkdir()
    (eli5 / 'SKILL.md').write_text('---\nname: eli5\n---\n')
    csv = tmp_path / 'site'
    csv.mkdir()
    (csv / 'market_breadth_data.csv').write_text('Date,Breadth_Index_8MA\n2026-11-19,0.6\n2026-11-20,0.574\n')
    (csv / 'market_breadth_summary.csv').write_text('Metric,Value\n')
    return analyzer, eli5, csv


def test_18_analyze_prepares_isolated_claude_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('SECRET_FOR_TEST', 'x')
    analyzer, eli5, csv = _skills(tmp_path)
    work = tmp_path / 'work'
    assert bx.analyze(csv, analyzer, eli5, work, previous_url=None) is True
    assert json.loads((work / 'facts.json').read_text()) == FACTS
    assert (work / 'analysis.md').read_text() == '# report'
    assert (work / 'data' / 'market_breadth_data.csv').exists()  # analyzer served a copy
    assert (work / '.claude' / 'skills' / 'eli5' / 'SKILL.md').exists()
    assert (work / '.claude' / 'skills' / 'market-breadth-analyzer' / 'SKILL.md').exists()
    prompt = (work / 'prompt.md').read_text()
    assert '{{' not in prompt and 'content="2026-11-20"' in prompt and '<b data-fact="breadth">57</b>' in prompt
    assert (work / 'example.html').read_text() == example()


def test_19_analyze_skips_claude_when_published_page_is_current(tmp_path):
    analyzer, eli5, csv = _skills(tmp_path)
    live = tmp_path / 'live.html'
    live.write_text(bx.with_csp(example()))
    assert bx.analyze(csv, analyzer, eli5, tmp_path / 'work', previous_url=live.as_uri()) is False


def test_20_cli_reports_outputs_and_survives_analyzer_failure(tmp_path, monkeypatch):
    analyzer, eli5, csv = _skills(tmp_path)
    out = tmp_path / 'gh_output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(out))
    base = ['analyze', '--csv-dir', str(csv), '--eli5-dir', str(eli5), '--previous-url', '']
    assert bx.main([*base, '--analyzer-dir', str(analyzer), '--work-dir', str(tmp_path / 'w1')]) == 0
    assert bx.main([*base, '--analyzer-dir', str(tmp_path / 'nope'), '--work-dir', str(tmp_path / 'w2')]) == 0
    work, site = _work(tmp_path / 'f', explainer=example())
    assert bx.main(['finalize', '--work-dir', str(work), '--site-dir', str(site)]) == 0
    assert out.read_text().split() == ['regenerate=true', 'regenerate=false', 'publish=true']


# ---------------------------------------------------------------- workflow wiring / isolation


@pytest.fixture(scope='module')
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def test_21_build_job_only_carries_forward_with_trusted_code(wf):
    steps = wf['jobs']['build']['steps']
    explainer = [s for s in steps if 'breadth_explainer' in s.get('run', '')]
    assert len(explainer) == 1 and 'carry-forward' in explainer[0]['run']
    assert explainer[0]['continue-on-error'] is True
    text = json.dumps(steps)
    assert 'claude' not in text.replace('breadth_explainer', '') and 'ANTHROPIC' not in text
    assert steps[0]['with']['persist-credentials'] is False


def test_22_explainer_job_is_isolated_and_non_blocking(wf):
    job = wf['jobs']['explainer']
    assert job['needs'] == 'build'
    assert job['permissions'] == {'contents': 'read'}
    assert 'environment' not in job
    for step in job['steps']:
        assert step.get('continue-on-error') is True, step['name']
        if 'checkout' in step.get('uses', ''):
            assert step['with']['persist-credentials'] is False
    secret_steps = [s for s in job['steps'] if 'ANTHROPIC_API_KEY' in json.dumps(s)]
    assert len(secret_steps) == 1 and 'claude -p' in secret_steps[0]['run']
    assert 'npm install' not in secret_steps[0]['run']


def test_23_claude_tools_are_scoped_to_its_workspace(wf):
    run = next(s for s in wf['jobs']['explainer']['steps'] if 'claude -p' in s.get('run', ''))['run']
    tools = re.search(r'--allowedTools "([^"]+)"', run).group(1).split(',')
    assert set(tools) == {'Read(./**)', 'Glob', 'Skill', 'Write(./explainer.html)', 'Edit(./explainer.html)'}
    assert '--permission-mode dontAsk' in run and '--setting-sources project' in run
    assert re.search(r'--max-budget-usd \d', run)


def test_24_third_party_code_is_pinned(wf):
    text = WORKFLOW.read_text()
    assert re.search(r'@anthropic-ai/claude-code@\d+\.\d+\.\d+\b', text)
    for repo in ('tradermonty/claude-trading-skills', 'takusaotome/claude-skills-library'):
        ref = re.search(rf'repository: {repo}\n\s+ref: (\S+)', text)
        assert ref and re.fullmatch(r'[0-9a-f]{40}', ref.group(1)), repo


def test_25_explainer_publish_redeploys_only_validated_page(wf):
    job = wf['jobs']['explainer-publish']
    assert set(job['needs']) == {'build', 'deploy', 'explainer'}
    assert 'secrets' not in json.dumps(job)
    for step in job['steps']:
        assert step.get('continue-on-error') is True, step['name']
    names = [s['name'] for s in job['steps']]
    assert (
        names.index('Validate explainer')
        < names.index('Upload site with explainer')
        < names.index('Redeploy GitHub Pages')
    )
    for name in ('Upload site with explainer', 'Redeploy GitHub Pages'):
        step = job['steps'][names.index(name)]
        assert step['if'] == "steps.finalize.outputs.publish == 'true'"


@pytest.mark.parametrize('job', ['explainer', 'explainer-publish'])
def test_25b_hung_steps_cannot_time_out_the_job(wf, job):
    # A job-level timeout concludes the run as failure, which scripts/monitor_feed.py would
    # report as a feed outage. Every step has its own timeout and they fit in the job budget.
    steps = wf['jobs'][job]['steps']
    budgets = [s.get('timeout-minutes') for s in steps]
    assert all(budgets), [s['name'] for s in steps if not s.get('timeout-minutes')]
    assert sum(budgets) < wf['jobs'][job]['timeout-minutes']


def test_26_index_links_to_explainer():
    html = (ROOT / 'reports' / 'index.html').read_text()
    assert 'href="breadth_explained.html"' in html
