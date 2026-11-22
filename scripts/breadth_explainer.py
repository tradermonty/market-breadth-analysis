"""Daily plain-English breadth explainer (breadth_explained.html) for the Pages site.

The page is written by Claude, so it is treated as untrusted. The workflow keeps it away from
the chart publication:

* ``carry-forward`` (build job, trusted): put the currently published explainer, re-checked,
  into the new site so every deploy keeps the link working. Never blocks the build.
* ``analyze`` (explainer job, no Pages permissions): serve the fresh CSVs on localhost, run the
  pinned Market Breadth Analyzer skill, record the facts and prepare Claude's workspace. A new
  page is needed only when the published one is a placeholder or shows different facts.
* Claude (explainer job) writes ``explainer.html`` with tools scoped to its workspace.
* ``finalize`` (explainer-publish job, trusted, no secrets): accept the page only if it passes
  the gate below, inject a strict Content-Security-Policy, and ask for a redeploy.

The gate parses the HTML (no regex sanitising): no scripts, frames, forms, embedded media,
external URLs, event handlers or CSS url()/@import; only links to the dashboard; labelled
``data-fact`` elements and ``breadth-*`` meta tags must match the analyzer's numbers.
"""

import argparse
import csv
import functools
import http.server
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed argument list, no shell
import sys
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'web' / 'breadth_explainer_example.html'
PROMPT = ROOT / 'web' / 'breadth_explainer_prompt.md'
OUTPUT_NAME = 'breadth_explained.html'
LIVE_URL = 'https://tradermonty.github.io/market-breadth-analysis/' + OUTPUT_NAME
MAX_BYTES = 300_000
ANALYZER_TIMEOUT = 300
# Added by finalize/carry-forward after validation; a meta CSP can only be tightened later.
CSP = (
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'\">"
)

BLOCKED_TAGS = {
    'script', 'iframe', 'frame', 'frameset', 'object', 'embed', 'applet', 'form', 'input', 'button',
    'textarea', 'select', 'base', 'link', 'img', 'image', 'picture', 'source', 'video', 'audio',
    'track', 'use', 'foreignobject', 'animate', 'animatemotion', 'animatetransform', 'set',
    'portal', 'noscript', 'template', 'svg:script',
    # raw-text elements that browsers and html.parser tokenise differently
    'plaintext', 'xmp', 'listing', 'noembed', 'noframes',
}  # fmt: skip
# CSS that can fetch resources or smuggle tokens (checked on whitespace-free, lower-cased CSS).
CSS_BLOCKED = ('\\', 'url(', '@import', '@font-face', 'image-set', 'image(', 'src(', 'cross-fade', 'element(',
               'expression(', 'javascript:', 'behavior:', '-moz-binding')  # fmt: skip
# Ways to show a fact to the gate but not to the reader.
CSS_HIDING = re.compile(
    r'display:none|visibility:hidden|content-visibility:hidden|color:transparent'
    r'|(?:opacity|font-size):0(?![.\d]*[1-9])'
)
# Elements allowed inside <head>. Anything else (e.g. <svg>) would make browsers close the
# head implicitly while the scanner still thinks it is in the head.
HEAD_TAGS = {'meta', 'title', 'style'}
# A data-fact must be a plain inline element (no class/style to hide it, not inside SVG).
FACT_TAGS = {'span', 'b', 'strong', 'time'}
# The only markup allowed before <head>: optional doctype and <html ...>. This makes the
# CSP insertion point unambiguous (no <head> hidden in a comment, title or attribute).
PROLOGUE = re.compile(r'(?is)\A\s*(?:<!doctype\s+html\s*>\s*)?(?:<html(?:\s[^<>]*)?>\s*)?<head\s*>')
URL_ATTRS = {
    'href', 'src', 'srcset', 'xlink:href', 'action', 'formaction', 'poster', 'data', 'background',
    'ping', 'cite', 'longdesc', 'manifest', 'codebase', 'archive', 'lowsrc', 'dynsrc',
}  # fmt: skip
ALLOWED_META_NAMES = {'viewport', 'description', 'color-scheme', 'theme-color', 'breadth-page-kind'}
FACT_META = {
    'breadth-data-date': 'data_date',
    'breadth-score': 'score',
    'breadth-zone': 'zone',
    'breadth-pct': 'breadth_pct',
}
SECRET_PATTERNS = re.compile(r'sk-ant-|ANTHROPIC_API_KEY|FMP_API_KEY|ghp_|github_pat_|-----BEGIN', re.I)


# ---------------------------------------------------------------- facts


def extract_facts(analysis):
    """Key facts the page must show, taken from the analyzer JSON."""
    try:
        date = (analysis.get('data_provenance') or {}).get('as_of') or analysis['metadata']['data_freshness'][
            'latest_date'
        ]
        score = float(analysis['composite']['composite_score'])
        zone = str(analysis['composite']['zone'])
        breadth = float(analysis['components']['breadth_level_trend']['current_8ma'])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f'analysis JSON missing a required field: {exc}') from exc
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(date)) or not re.fullmatch(r'[A-Za-z][A-Za-z ]{0,30}', zone):
        raise ValueError('analysis JSON has an invalid date or zone')
    if not 0 <= breadth <= 1 or not 0 <= score <= 100:
        raise ValueError('analysis JSON has an out-of-range score or breadth')
    return {'data_date': date, 'score': round(score, 1), 'zone': zone, 'breadth_pct': round(breadth * 100)}


def fact_strings(facts):
    return {
        'data_date': str(facts['data_date']),
        'score': f'{float(facts["score"]):.1f}',
        'zone': str(facts['zone']),
        'breadth_pct': str(int(facts['breadth_pct'])),
    }


# ---------------------------------------------------------------- HTML scan


class _Scan(HTMLParser):
    """One linear pass over the page collecting what the gate needs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.problems = []
        self.meta = {}
        self.facts = {}  # data-fact name -> list of values
        self.has_dashboard_link = False
        self.text = []
        self._skip = 0  # inside style/title
        self._fact_stack = []  # [name, collected text parts, tag, datetime]
        self._in_style = False
        self._in_head = False
        self._svg_depth = 0

    def _bad(self, message):
        if message not in self.problems:
            self.problems.append(message)

    def _check_url(self, tag, name, value):
        v = re.sub(r'[\s\x00-\x1f]', '', value or '').lower()
        if tag == 'a' and name == 'href' and value == 'index.html':
            self.has_dashboard_link = True
            return
        if tag == 'a' and name == 'href' and (value.startswith(('#', 'https://tradermonty.github.io/'))):
            return
        if name == 'xmlns' or (name == 'xmlns:xlink' and v.startswith('http://www.w3.org/')):
            return
        self._bad(f'url not allowed in <{tag} {name}>: {value[:60]}')
        if 'javascript:' in v:
            self._bad('javascript: url')

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs)
        if tag == 'meta' or tag in BLOCKED_TAGS:
            return
        if self._fact_stack and self._fact_stack[-1][2] == tag:
            self.handle_endtag(tag)

    def _start(self, tag, attrs):
        tag = tag.lower()
        if tag in BLOCKED_TAGS or tag.endswith(':script'):
            self._bad(f'blocked element <{tag}>')
        attrs = [(k.lower(), v if v is not None else '') for k, v in attrs]
        for name, value in attrs:
            flat = re.sub(r'[\s\x00-\x1f]', '', value).lower()
            if name.startswith('on'):
                self._bad(f'event handler attribute {name}')
            elif name in URL_ATTRS or name.endswith(':href') or name.startswith('xmlns'):
                self._check_url(tag, name, value)
            elif name == 'style':
                self._check_css(value)
            elif name in ('hidden', 'aria-hidden') and tag != 'svg':
                self._bad(f'{name} attribute not allowed on <{tag}>')
            if 'javascript:' in flat:
                self._bad('javascript: url')
            # SVG paint references such as fill="url(#heat)" must stay inside the document.
            for ref in re.findall(r'url\(([^)]*)\)', flat):
                if not ref.strip('\'"').startswith('#'):
                    self._bad(f'non-local url() in {name}')
        if tag == 'head':
            self._in_head = True
        elif tag == 'body':
            self._in_head = False
        elif self._in_head and tag not in HEAD_TAGS:
            self._bad(f'<{tag}> not allowed inside <head>')
        if tag == 'svg':
            self._svg_depth += 1
        if tag == 'meta':
            self._meta(dict(attrs))
        if tag == 'style' and not self._in_head:
            self._bad('<style> only allowed inside <head>')
        if tag in ('style', 'title'):
            self._skip += 1
            self._in_style = tag == 'style'
        fact = dict(attrs).get('data-fact')
        if fact:
            names = {k for k, _ in attrs}
            if tag not in FACT_TAGS or self._svg_depth or names & {'class', 'style', 'id'}:
                self._bad(f'data-fact="{fact}" must be a plain <span>/<b>/<strong>/<time> outside SVG')
            when = dict(attrs).get('datetime') if tag == 'time' else None
            self._fact_stack.append([fact, [], tag, when])

    def _meta(self, attrs):
        if 'http-equiv' in attrs:
            self._bad('meta http-equiv not allowed')
            return
        if 'charset' in attrs:
            return
        name = attrs.get('name', '').lower()
        if name in FACT_META:
            self.meta.setdefault(name, []).append(attrs.get('content', ''))
        elif name not in ALLOWED_META_NAMES:
            self._bad(f'meta {name or "(unnamed)"} not allowed')

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == 'head':
            self._in_head = False
        if tag == 'svg' and self._svg_depth:
            self._svg_depth -= 1
        if tag in ('style', 'title') and self._skip:
            self._skip -= 1
            self._in_style = False
        if self._fact_stack and self._fact_stack[-1][2] == tag:
            name, parts, _, when = self._fact_stack.pop()
            shown = ' '.join(''.join(parts).split())
            if when is not None:
                # A <time> fact is its datetime, and the visible text must show that day and year.
                ok = re.fullmatch(r'\d{4}-\d{2}-\d{2}', when) and all(
                    re.search(rf'(?<!\d){p}(?!\d)', shown) for p in (when[:4], str(int(when[8:10])))
                )
                shown = when if ok else f'{when} (text "{shown}" does not show it)'
            self.facts.setdefault(name, []).append(shown)

    def handle_data(self, data):
        if self._in_style:
            if '<' in data:
                self._bad('"<" not allowed inside <style>')
            self._check_css(data)
        if self._skip:
            return
        self.text.append(data)
        for entry in self._fact_stack:
            entry[1].append(data)

    def handle_comment(self, data):
        self._bad('HTML comments not allowed')

    def _check_css(self, css):
        flat = re.sub(r'\s+', '', re.sub(r'(?s)/\*.*?\*/', '', css)).lower()
        if '/*' in flat:
            self._bad('unterminated CSS comment')
        for token in CSS_BLOCKED:
            if token in flat:
                self._bad(f'css {token!r} not allowed')
        hidden = CSS_HIDING.search(flat)
        if hidden:
            self._bad(f'css {hidden.group(0)} not allowed (could hide facts)')

    def handle_pi(self, data):
        self._bad('processing instruction not allowed')

    def unknown_decl(self, data):
        self._bad('unknown declaration not allowed')


def _scan(page):
    scan = _Scan()
    scan.feed(page)
    scan.close()
    return scan


def safety_problems(page):
    """Checks that do not depend on today's numbers."""
    if len(page.encode('utf-8', 'replace')) > MAX_BYTES:
        return [f'size over {MAX_BYTES} bytes']
    lower = page.lower()
    problems = []
    if '<html' not in lower or '</html>' not in lower:
        problems.append('not a complete HTML document')
    if '<!--' in page:
        problems.append('HTML comments not allowed')
    if not PROLOGUE.match(page) or len(re.findall(r'(?i)<head(?=[\s>/])', page)) != 1:
        problems.append('document must start with [<!DOCTYPE html>][<html ...>]<head> and have exactly one <head>')
    if SECRET_PATTERNS.search(page):
        problems.append('looks like it contains a secret')
    scan = _scan(page)
    problems += scan.problems
    if not scan.has_dashboard_link:
        problems.append('missing dashboard link href="index.html"')
    if 'not investment advice' not in ' '.join(''.join(scan.text).split()).lower():
        problems.append('missing disclaimer "Not investment advice"')
    return problems


def page_meta(page):
    """breadth-* meta values (first occurrence), for comparing published pages."""
    if not page or len(page.encode('utf-8', 'replace')) > MAX_BYTES:
        return {}
    scan = _scan(page)
    return {FACT_META[k]: v[0] for k, v in scan.meta.items()}


def page_data_date(page):
    return page_meta(page).get('data_date')


def is_placeholder(page):
    return bool(page) and 'name="breadth-page-kind" content="placeholder"' in page


def needs_regeneration(previous_page, facts):
    """Regenerate unless the published page is a real page showing exactly today's facts."""
    if not previous_page or is_placeholder(previous_page):
        return True
    return page_meta(previous_page) != fact_strings(facts)


def validate_explainer(page, facts):
    """Return a list of problems; empty means the page may be published."""
    problems = safety_problems(page)
    if problems and problems[0].startswith('size over'):
        return problems
    scan = _scan(page)
    expected = fact_strings(facts)
    for meta_name, key in FACT_META.items():
        values = scan.meta.get(meta_name, [])
        if values != [expected[key]]:
            problems.append(f'meta {meta_name} must be exactly once "{expected[key]}", got {values}')
    labels = {'data_date': 'date', 'score': 'score', 'zone': 'zone', 'breadth_pct': 'breadth'}
    for key, fact in labels.items():
        shown = scan.facts.get(fact, [])
        if not shown:
            problems.append(f'no data-fact="{fact}" element')
        elif any(v != expected[key] for v in shown):
            problems.append(f'data-fact="{fact}" shows {shown}, expected {expected[key]}')
    return problems


def with_csp(page):
    """Insert the CSP right after the real <head> tag (pages failing PROLOGUE are refused)."""
    match = PROLOGUE.match(page)
    if not match:
        raise ValueError('page does not start with a plain <head>; refusing to place the CSP')
    return page[: match.end()] + '\n' + CSP + page[match.end() :]


def csv_facts(site_dir):
    """Date and breadth % from the published CSV (trusted), or None when the CSV is absent."""
    path = Path(site_dir) / 'market_breadth_data.csv'
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    last = rows[-1]
    return {'data_date': last['Date'], 'breadth_pct': round(float(last['Breadth_Index_8MA']) * 100)}


def placeholder_page():
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="breadth-page-kind" content="placeholder">\n'
        '<title>Market Breadth, Explained</title>\n'
        '<style>body{margin:0;font:17px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}'
        'main{max-width:720px;margin:0 auto;padding:24px 16px}a{color:#0366d6;text-decoration:none}</style>\n'
        '</head>\n<body>\n<main>\n<p><a href="index.html">&larr; Market Breadth dashboard</a></p>\n'
        "<h1>Market breadth explained</h1>\n<p>Today's explanation is not ready yet. "
        'Please see the chart on the dashboard.</p>\n'
        '<p><small>Not investment advice.</small></p>\n</main>\n</body>\n</html>\n'
    )


# ---------------------------------------------------------------- helpers


def _read(path):
    try:
        return Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None


def _fetch(url):
    if not url or not url.startswith(('https://', 'file://')):
        return None
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:  # nosec B310 - https/file scheme checked above
            body = resp.read(MAX_BYTES + 1)
        return body.decode('utf-8', 'replace')
    except (OSError, ValueError):
        return None


def _github_output(**values):
    path = os.environ.get('GITHUB_OUTPUT')
    if path:
        with open(path, 'a', encoding='utf-8') as handle:
            for key, value in values.items():
                handle.write(f'{key}={value}\n')


def _warn(message):
    print(f'::warning title=Breadth explainer::{message}')


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _run_analyzer(csv_dir, analyzer_dir, out_dir):
    handler = functools.partial(_QuietHandler, directory=str(csv_dir))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        scripts = Path(analyzer_dir) / 'scripts'
        subprocess.run(  # nosec B603 - fixed argv, pinned skill script
            [
                sys.executable,
                str(scripts / 'market_breadth_analyzer.py'),
                '--detail-url',
                f'http://127.0.0.1:{port}/market_breadth_data.csv',
                '--summary-url',
                f'http://127.0.0.1:{port}/market_breadth_summary.csv',
                '--output-dir',
                str(out_dir),
            ],
            cwd=scripts,
            check=True,
            timeout=ANALYZER_TIMEOUT,
            env={'PATH': os.environ.get('PATH', ''), 'HOME': os.environ.get('HOME', '')},
        )
    finally:
        server.shutdown()
        server.server_close()
    reports = [
        p for p in sorted(Path(out_dir).glob('market_breadth_*.json')) if p.name != 'market_breadth_history.json'
    ]
    if not reports:
        raise RuntimeError('analyzer produced no JSON report')
    return json.loads(reports[-1].read_text(encoding='utf-8')), reports[-1].with_suffix('.md')


# ---------------------------------------------------------------- commands


def carry_forward(site_dir, previous_url=LIVE_URL):
    """Build job: keep the published explainer (re-checked) or a placeholder. Never raises."""
    try:
        previous = _fetch(previous_url)
        if previous and not is_placeholder(previous):
            body = previous.replace('\n' + CSP, '', 1)  # undo with_csp() exactly
            if page_data_date(body) and not safety_problems(body):
                (Path(site_dir) / OUTPUT_NAME).write_text(with_csp(body), encoding='utf-8')
                print('breadth explainer carried forward:', page_data_date(body))
                return 'carried'
        (Path(site_dir) / OUTPUT_NAME).write_text(placeholder_page(), encoding='utf-8')
        print('breadth explainer: placeholder')
        return 'placeholder'
    except Exception as exc:
        _warn(f'carry-forward failed: {exc}')
        return 'error'


def analyze(csv_dir, analyzer_dir, out_dir, previous_url=LIVE_URL, runner=None):
    """Analysis job (no secrets): run the analyzer; return True when a new page should be written.

    Writes only analysis.json, analysis.md and facts.json to ``out_dir`` (the artifact handed to
    the Claude job). ``runner`` replaces the localhost-HTTP analyzer run in tests.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / 'scratch'
    data = scratch / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('market_breadth_data.csv', 'market_breadth_summary.csv'):
        shutil.copyfile(Path(csv_dir) / name, data / name)  # the analyzer never sees the site dir

    analysis, markdown = (runner or _run_analyzer)(data, analyzer_dir, scratch / 'analysis_out')
    facts = extract_facts(analysis)
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    (out / 'analysis.md').write_text(Path(markdown).read_text(encoding='utf-8')[:MAX_BYTES], encoding='utf-8')
    (out / 'facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')

    regenerate = needs_regeneration(_fetch(previous_url), facts)
    print(f'facts={json.dumps(facts)} regenerate={str(regenerate).lower()}')
    return regenerate


def _copy_skill_docs(src, dest):
    """Copy only regular Markdown files of a skill; refuse symlinks anywhere in the tree."""
    src = Path(src)
    for path in [src, *src.rglob('*')]:
        if path.is_symlink():
            raise ValueError(f'symlink in skill: {path.relative_to(src.parent)}')
    for path in src.rglob('*.md'):
        if path.is_file():
            target = dest / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def prepare(analysis_dir, analyzer_skill_dir, eli5_dir, work_dir):
    """Claude job (trusted code, before the secret is used): build a fresh, clean workspace.

    Only regular files are created. Facts are re-derived here from analysis.json rather than
    trusting the facts.json in the artifact.
    """
    work = Path(work_dir)
    if work.exists():
        raise ValueError(f'{work} already exists; refusing to reuse a workspace')
    work.mkdir(parents=True)
    src = Path(analysis_dir)
    analysis = json.loads((src / 'analysis.json').read_text(encoding='utf-8'))
    facts = extract_facts(analysis)
    (work / 'analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    (work / 'analysis.md').write_text((src / 'analysis.md').read_text(encoding='utf-8')[:MAX_BYTES], encoding='utf-8')
    (work / 'facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')

    skills = work / '.claude' / 'skills'
    _copy_skill_docs(eli5_dir, skills / 'eli5')
    _copy_skill_docs(analyzer_skill_dir, skills / 'market-breadth-analyzer')
    shutil.copyfile(EXAMPLE, work / 'example.html')
    prompt = PROMPT.read_text(encoding='utf-8')
    for key, value in fact_strings(facts).items():
        prompt = prompt.replace('{{' + key.upper() + '}}', value)
    (work / 'prompt.md').write_text(prompt, encoding='utf-8')
    return facts


def finalize(work_dir, site_dir):
    """Explainer-publish job: accept Claude's page only if it passes the gate. Never raises.

    Returns 'generated' (site file replaced; redeploy) or 'rejected'/'missing'/'error' (keep the
    carried-forward page; no redeploy).
    """
    status, notes = 'error', []
    try:
        work = Path(work_dir)
        facts_text = _read(work / 'facts.json')
        page = _read(work / 'explainer.html')
        if not facts_text or page is None:
            status, notes = 'missing', ['no facts.json or explainer.html from the explainer job']
        else:
            facts = json.loads(facts_text)
            # facts.json comes from the explainer job (third-party analyzer); re-derive what we
            # can from the CSV that this trusted job is about to publish.
            trusted = csv_facts(site_dir)
            if trusted is None:
                problems = ['published market_breadth_data.csv not found; cannot verify facts']
            elif trusted['data_date'] != facts['data_date'] or trusted['breadth_pct'] != int(facts['breadth_pct']):
                problems = [f'facts.json {facts} disagrees with published CSV {trusted}']
            else:
                problems = []
            problems += validate_explainer(page, facts)
            if problems:
                status, notes = 'rejected', problems
            else:
                (Path(site_dir) / OUTPUT_NAME).write_text(with_csp(page), encoding='utf-8')
                status = 'generated'
    except Exception as exc:
        status, notes = 'error', [f'{type(exc).__name__}: {exc}']
    print(f'breadth explainer: {status}')
    for note in notes:
        print(f'  {note}')
    if status != 'generated':
        _warn(f'new page not published ({status}); keeping the previous page. ' + '; '.join(notes)[:900])
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as handle:
            handle.write(f'### Breadth explainer: {status}\n\n' + ''.join(f'- {n}\n' for n in notes))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('carry-forward')
    c.add_argument('--site-dir', required=True)
    c.add_argument('--previous-url', default=LIVE_URL)
    a = sub.add_parser('analyze')
    a.add_argument('--csv-dir', required=True)
    a.add_argument('--analyzer-dir', required=True)
    a.add_argument('--out-dir', required=True)
    a.add_argument('--previous-url', default=LIVE_URL, help='published page to compare ("" to skip)')
    p = sub.add_parser('prepare')
    p.add_argument('--analysis-dir', required=True)
    p.add_argument('--analyzer-skill-dir', required=True)
    p.add_argument('--eli5-dir', required=True)
    p.add_argument('--work-dir', required=True)
    f = sub.add_parser('finalize')
    f.add_argument('--site-dir', required=True)
    f.add_argument('--work-dir', required=True)
    args = parser.parse_args(argv)

    if args.command == 'carry-forward':
        carry_forward(args.site_dir, args.previous_url)
    elif args.command == 'analyze':
        try:
            regenerate = analyze(args.csv_dir, args.analyzer_dir, args.out_dir, args.previous_url)
        except Exception as exc:
            _warn(f'analysis failed: {type(exc).__name__}: {exc}')
            regenerate = False
        _github_output(regenerate=str(regenerate).lower())
    elif args.command == 'prepare':
        try:
            prepare(args.analysis_dir, args.analyzer_skill_dir, args.eli5_dir, args.work_dir)
            ready = True
        except Exception as exc:
            _warn(f'workspace preparation failed: {type(exc).__name__}: {exc}')
            ready = False
        _github_output(ready=str(ready).lower())
    else:
        _github_output(publish=str(finalize(args.work_dir, args.site_dir) == 'generated').lower())
    return 0


if __name__ == '__main__':
    sys.exit(main())
