#!/usr/bin/env python3
"""CI gate script for Issue #16.

Subcommands
-----------
pip-audit <audit.json> --allowlist <allowlist.toml>
    Classify pip-audit findings against a reviewed exception allowlist.
    Fail on unclassified findings, expired exceptions, missing/invalid
    allowlist fields, or empty/invalid JSON input. Warn on stale
    allowlist entries and allowed (documented, unexpired) exceptions.

secrets-check <live-scan.json> --baseline <.secrets.baseline>
    Compare a fresh detect-secrets scan against the reviewed baseline.
    Fail on new findings, hash-drift findings (same file+type, unknown
    hash), audited-but-flagged-unused entries, and empty/invalid JSON.
    Warn (do not fail) on baseline entries that no longer appear
    (prune hints). Never print raw secret values.

Exit code: 0 iff the report contains no FAIL-category rows.
Requires Python >= 3.11 (stdlib tomllib). No network access.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import tomllib

FAIL_CATEGORIES = ('NEW', 'EXPIRED', 'AUDIT-DERELICT', 'COMMENT', 'MALFORMED-INPUT', 'SCANNER-FAILURE')
WARN_CATEGORIES = ('ALLOWED', 'STALE', 'PRUNE')

ISO_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

ISO_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

SECRETS_BASELINE_PATH_DEFAULT = '.secrets.baseline'


def hash_secret(secret: str) -> str:
    """Same algorithm as detect-secrets v1.x (sha1 of the secret)."""
    return hashlib.sha1(secret.encode('utf-8'), usedforsecurity=False).hexdigest()


class GateReport:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        self.seen_keys: set[tuple] = set()  # dedupe helper for repeated findings

    def add(self, category: str, message: str) -> None:
        if category not in FAIL_CATEGORIES and category not in WARN_CATEGORIES:
            raise ValueError(f'unknown category: {category}')
        self.rows.append((category, message))

    @property
    def exit_code(self) -> int:
        return 1 if any(cat in FAIL_CATEGORIES for cat, _ in self.rows) else 0

    def render(self) -> str:
        lines = []
        for category, message in self.rows:
            marker = 'FAIL' if category in FAIL_CATEGORIES else 'WARN'
            lines.append(f'{marker} [{category}] {message}')
        lines.append('RESULT: ' + ('CI-GATE-FAIL' if self.exit_code else 'CI-GATE-PASS'))
        return '\n'.join(lines)


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise MalformedInputError(f'input file not found: {path}')
    text = p.read_text(encoding='utf-8').strip()
    if not text:
        raise MalformedInputError(f'empty scanner output: {path}')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedInputError(f'invalid JSON in {path}: {exc}') from exc
    if not isinstance(data, dict) or not isinstance(data.get('results', {}), dict):
        raise MalformedInputError(f'unexpected scanner JSON structure in {path}')
    return data


class MalformedInputError(Exception):
    pass


# ---------------------------------------------------------------------------
# pip-audit
# ---------------------------------------------------------------------------


def _parse_allowlist(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise MalformedInputError(f'allowlist file not found: {path}')
    try:
        data = tomllib.loads(p.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as exc:
        raise MalformedInputError(f'invalid TOML in {path}: {exc}') from exc
    exceptions = data.get('exceptions', [])
    if not isinstance(exceptions, list):
        raise MalformedInputError(f'invalid allowlist structure in {path}: "exceptions" must be a table array')
    problems: list[str] = []
    seen = set()
    for i, exc in enumerate(exceptions):
        if not isinstance(exc, dict):
            problems.append(f'exception[{i}]: entry is not a table')
            continue
        prefix = f'exception[{i}] ({exc.get("id", "?")})'
        if not exc.get('id'):
            problems.append(f'{prefix}: missing id')
        if not exc.get('reason'):
            problems.append(f'{prefix}: missing or empty reason')
        if exc.get('owner', None) is None or not str(exc.get('owner', '')).strip():
            problems.append(f'{prefix}: missing or empty owner')
        expires = exc.get('expires')
        if not expires:
            problems.append(f'{prefix}: missing expires')
        elif not isinstance(expires, str) or not ISO_DATE_RE.fullmatch(expires):
            problems.append(f"{prefix}: unparseable expires '{expires}' (expected YYYY-MM-DD)")
        key = (exc.get('id'), exc.get('package'), exc.get('expires'))
        if key in seen:
            problems.append(f'{prefix}: duplicate allowlist entry')
        seen.add(key)
    if problems:
        raise MalformedInputError('invalid allowlist: ' + '; '.join(problems))
    return exceptions


def _iter_vulns(audit_data: dict):
    for dep in audit_data.get('dependencies', []):
        package = dep.get('name', '')
        version = dep.get('version', '')
        for vuln in dep.get('vulns', []):
            yield package, version, vuln


def _allowlist_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _select_exception(allowlist: list[dict], ids: list[str], package: str):
    """Pick the most specific exception: exact-package match first, then the one with the latest expiry."""
    candidates = []
    for idx, exc in enumerate(allowlist):
        exc_id = exc.get('id')
        exc_pkg = str(exc.get('package', '') or '')
        if exc_id not in ids:
            continue
        if exc_pkg not in ('', package):
            continue
        exact = exc_pkg == package
        candidates.append((exact, _allowlist_date(str(exc['expires'])), idx))
    if not candidates:
        return None
    return max(candidates, key=lambda t: (t[0], t[1]))[2]


def run_pip_audit_gate(
    audit_path: str,
    allowlist_path: str,
    scanner_code: int | None = None,
    today: datetime.date | None = None,
) -> GateReport:
    report = GateReport()
    try:
        audit_data = _load_json_pip(audit_path)
        allowlist = _parse_allowlist(allowlist_path)
    except MalformedInputError as exc:
        report.add('MALFORMED-INPUT', str(exc))
        return report

    today = today or datetime.date.today()
    matched_exceptions = set()

    if scanner_code is not None and scanner_code > 1:
        report.add('SCANNER-FAILURE', f'pip-audit failed with exit code {scanner_code} — not bypassed.')
        # Continue building the rest of the report, but exit is already failure.

    for package, version, vuln in _iter_vulns(audit_data):
        vid = vuln.get('id', '?')
        aliases = vuln.get('aliases', []) or []
        ids = [vid, *aliases]
        label = f'{package}=={version} {vid}'
        dedupe_key = ('NEW', package, version, vid)
        if dedupe_key in report.seen_keys:
            continue
        report.seen_keys.add(dedupe_key)

        matched_idx = _select_exception(allowlist, ids, package)
        if matched_idx is None:
            report.add(
                'NEW',
                f'{label}: no allowlisted exception. '
                f'aliases={aliases or [vid]}. '
                'Fix by upgrading or add an owned, dated exception in '
                f'{allowlist_path}.',
            )
            continue

        exc = allowlist[matched_idx]
        matched_exceptions.add(matched_idx)
        expires = _allowlist_date(str(exc['expires']))
        fix_versions = vuln.get('fix_versions') or []
        fix_hint = ', '.join(fix_versions) or exc.get('fix_version', 'unknown')
        message = (
            f'{label}: allowed by exception expires={expires.isoformat()} '
            f'owner={exc["owner"]} fix_hint={fix_hint} reason={exc["reason"]}'
        )
        if expires < today:
            report.add('EXPIRED', message + ' — RENEW OR PRUNE THE EXCEPTION NOW.')
        else:
            report.add('ALLOWED', message)

    for idx, exc in enumerate(allowlist):
        if idx not in matched_exceptions:
            report.add(
                'STALE',
                (
                    f"exception[{idx}] ({exc['id']}, package='{exc.get('package', '')}', "
                    f'expires={exc["expires"]}) no longer matches any finding — prune it.'
                ),
            )
    return report


def _load_json_pip(path: str) -> dict:
    """pip-audit JSON: {dependencies: [{name, version, vulns: [...]}], fixes: []}."""
    p = Path(path)
    if not p.exists():
        raise MalformedInputError(f'input file not found: {path}')
    text = p.read_text(encoding='utf-8').strip()
    if not text:
        raise MalformedInputError(f'empty scanner output: {path}')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedInputError(f'invalid JSON in {path}: {exc}') from exc
    if not isinstance(data, dict) or not isinstance(data.get('dependencies', []), list):
        raise MalformedInputError(f'unexpected pip-audit JSON structure in {path}')
    return data


# ---------------------------------------------------------------------------
# detect-secrets
# ---------------------------------------------------------------------------

REDACTED_HASH_PREFIX_LEN = 8


def _entry_key(entry: dict) -> tuple:
    return (entry.get('filename'), entry.get('type'), entry.get('hashed_secret'))


def _collect_entries(live: dict, baseline: dict) -> tuple[dict, dict]:
    live_entries: dict[tuple, dict] = {}
    for entries in live.get('results', {}).values():
        if not isinstance(entries, list):
            raise MalformedInputError('invalid results structure (values must be lists)')
        for entry in entries:
            if not isinstance(entry, dict):
                raise MalformedInputError('invalid finding entry')
            live_entries[_entry_key(entry)] = entry

    baseline_by_key: dict[tuple, dict] = {}
    for entries in baseline.get('results', {}).values():
        if not isinstance(entries, list):
            raise MalformedInputError('invalid results structure (values must be lists)')
        for entry in entries:
            if not isinstance(entry, dict):
                raise MalformedInputError('invalid finding entry')
            baseline_by_key[_entry_key(entry)] = entry
    return live_entries, baseline_by_key


def run_secrets_gate(live_path: str, baseline_path: str) -> GateReport:
    report = GateReport()
    try:
        live = _load_json(live_path)
        baseline = _load_json(baseline_path)
    except MalformedInputError as exc:
        report.add('MALFORMED-INPUT', str(exc))
        return report

    try:
        live_entries, baseline_by_key = _collect_entries(live, baseline)
    except MalformedInputError as exc:
        report.add('MALFORMED-INPUT', str(exc))
        return report

    consumed_baseline_keys: set[tuple] = set()
    for live_key, live_entry in live_entries.items():
        if live_key in baseline_by_key:
            base_entry = baseline_by_key[live_key]
            is_secret = base_entry.get('is_secret')
            is_used = base_entry.get('is_used')
            if is_secret is True and is_used is not True:  # explicitly unused or not audited
                report.add(
                    'AUDIT-DERELICT',
                    _redacted_message(live_entry) + ': audited as a real secret (is_secret=true) but marked '
                    'unused — remediate or update the audit verdict.',
                )
            continue

        # Primary key miss: check for hash drift (same file+type, different hash).
        drift = [key for key in baseline_by_key if key[0] == live_key[0] and key[1] == live_key[1]]
        if drift:
            report.add(
                'COMMENT',
                _redacted_message(live_entry) + ': hash drift — the secret in an already-reviewed file+type '
                'changed. Review the diff; if the finding is known-good, '
                f'regenerate the baseline ({SECRETS_BASELINE_PATH_DEFAULT}) '
                'and re-run this gate.',
            )
            consumed_baseline_keys.update(drift)
        else:
            report.add(
                'NEW',
                _redacted_message(live_entry) + ': not in reviewed baseline (value redacted). Review the '
                'finding; if benign, add to the baseline via '
                f"'detect-secrets scan ... --update {SECRETS_BASELINE_PATH_DEFAULT}' "
                "and 'detect-secrets audit'.",
            )

    for base_key in baseline_by_key:
        if base_key not in live_entries and base_key not in consumed_baseline_keys:
            report.add(
                'PRUNE',
                f'baseline entry {base_key[0]}:{base_key[1]} '
                f'(hash {base_key[2][:REDACTED_HASH_PREFIX_LEN]}...) no longer '
                'found by the scan — prune it from the baseline.',
            )
    return report


def _redacted_message(entry: dict) -> str:
    filename = entry.get('filename', '?')
    line = entry.get('line_number', '?')
    secret_type = entry.get('type', '?')
    hashed = str(entry.get('hashed_secret', ''))[:REDACTED_HASH_PREFIX_LEN]
    return f'{filename}:{line} [{secret_type}] hash {hashed}...'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='CI security gates for issue #16')
    sub = parser.add_subparsers(dest='gate', required=True)

    p_audit = sub.add_parser('pip-audit', help='gate pip-audit JSON results')
    p_audit.add_argument('audit_json')
    p_audit.add_argument('--allowlist', required=True)
    p_audit.add_argument(
        '--scanner-code',
        type=int,
        default=None,
        help='exit code of the pip-audit process; >1 marks a scanner failure '
        '(exit 0/1 with valid JSON is handled by classification)',
    )
    p_audit.add_argument(
        '--today',
        default=None,
        help='override the current date (YYYY-MM-DD); for deterministic testing',
    )

    p_secrets = sub.add_parser('secrets-check', help='gate live detect-secrets scan vs baseline')
    p_secrets.add_argument('live_json')
    p_secrets.add_argument('--baseline', required=True)

    args = parser.parse_args(argv)
    if args.gate == 'pip-audit':
        override_today = None
        if args.today:
            try:
                override_today = datetime.date.fromisoformat(args.today)
            except ValueError:
                parser.error(f'invalid --today date: {args.today}')
        report = run_pip_audit_gate(
            args.audit_json,
            args.allowlist,
            scanner_code=args.scanner_code,
            today=override_today,
        )
    else:
        report = run_secrets_gate(args.live_json, args.baseline)

    print(report.render())
    return report.exit_code


if __name__ == '__main__':
    sys.exit(main())
