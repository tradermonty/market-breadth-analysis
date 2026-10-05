"""Tests for scripts/security_gate.py (Issue #16).

Covers pip-audit allowlist classification, detect-secrets new-finding
gate, scanner-failure fallback, exception precedence, and redaction
guarantees. All tests are API-free and deterministic: dates are injected
via --today / the run_pip_audit_gate `today` parameter, and fixtures use
per-test unique filenames so runs are order- and parallelism-safe.
"""

import contextlib
import datetime
import io
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import security_gate  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'security'

TODAY = datetime.date.today()  # deterministic only via the gate's injected `today`
DAYS = datetime.timedelta(days=1)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _run_gate(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = security_gate.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _iso(delta: datetime.timedelta) -> str:
    return (TODAY + delta).strftime('%Y-%m-%d')


def _vuln_json(package, version, vid, aliases=None):
    return json.dumps(
        {
            'dependencies': [
                {
                    'name': package,
                    'version': version,
                    'vulns': [
                        {
                            'id': vid,
                            'fix_versions': [],
                            'aliases': aliases or [],
                            'description': 'test vuln',
                        }
                    ],
                }
            ],
            'fixes': [],
        }
    )


class TestSecurityGate(unittest.TestCase):
    """test_01..test_26 per docs for issue #16."""

    # pip-audit gate

    def test_01_pip_audit_pass_when_no_findings(self):
        allowlist = _write(
            FIXTURES / 'test_01_allowlist.toml',
            self._allowlist(expires=_iso(datetime.timedelta(days=90))),
        )
        audit = _write(FIXTURES / 'test_01_audit.json', '{"dependencies": [], "fixes": []}')
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)

    def test_02_pip_audit_fails_on_new_vulnerability(self):
        allowlist = _write(FIXTURES / 'test_02_allowlist.toml', '')
        audit = _write(FIXTURES / 'test_02_audit.json', _vuln_json('other-pkg', '1.0.0', 'GHSA-9999-zzzz'))
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 1)
        self.assertIn('GHSA-9999-zzzz', out)
        self.assertIn('NEW', out)

    def test_03_pip_audit_fails_on_expired_exception(self):
        allowlist = _write(FIXTURES / 'test_03_allowlist.toml', self._allowlist(expires=_iso(-DAYS)))
        audit = _write(
            FIXTURES / 'test_03_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 1)
        self.assertIn('EXPIRED', out)
        self.assertNotIn('STALE', out)  # the matched (expired) entry is not also stale

    def test_04_pip_audit_allows_exception_not_expired(self):
        allowlist = _write(FIXTURES / 'test_04_allowlist.toml', self._allowlist(expires=_iso(DAYS)))
        audit = _write(
            FIXTURES / 'test_04_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('ALLOWED', out)

    def test_05_pip_audit_allows_exception_at_expiry_boundary(self):
        allowlist = _write(FIXTURES / 'test_05_allowlist.toml', self._allowlist(expires=_iso(DAYS * 0)))
        audit = _write(
            FIXTURES / 'test_05_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('ALLOWED', out)

    def test_06_pip_audit_allows_exception_matched_via_aliases(self):
        allowlist = _write(FIXTURES / 'test_06_allowlist.toml', self._allowlist(expires=_iso(DAYS)))
        audit = _write(
            FIXTURES / 'test_06_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'PYSEC-0000-0001', aliases=['GHSA-1111-aaaa-bbbb']),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('ALLOWED', out)

    def test_07_pip_audit_id_matches_but_package_differs_fails(self):
        allowlist = _write(FIXTURES / 'test_07_allowlist.toml', self._allowlist(expires=_iso(DAYS)))
        audit = _write(
            FIXTURES / 'test_07_audit.json',
            _vuln_json('different-pkg', '1.0.0', 'GHSA-2222-cccc-dddd', aliases=['GHSA-1111-aaaa-bbbb']),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 1)
        self.assertIn('NEW', out)
        self.assertIn('different-pkg', out)

    def test_08_pip_audit_allows_id_only_match_when_package_empty(self):
        allowlist = _write(
            FIXTURES / 'test_08_allowlist.toml',
            self._allowlist(expires=_iso(DAYS)).replace("package = 'vuln-pkg'", "package = ''"),
        )
        audit = _write(
            FIXTURES / 'test_08_audit.json',
            _vuln_json('any-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('ALLOWED', out)

    def _assert_invalid_allowlist_fails(self, content, detail, tag):
        allowlist = _write(FIXTURES / f'test_{tag}_allowlist.toml', content)
        audit = _write(
            FIXTURES / f'test_{tag}_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 1, out)
        self.assertIn('invalid allowlist', out)
        self.assertIn(detail, out)

    def test_09_pip_audit_fails_when_allowlist_missing_or_invalid_expires(self):
        base = self._allowlist(expires=_iso(DAYS))
        self._assert_invalid_allowlist_fails(
            base.replace(f"expires = '{_iso(DAYS)}'", "expires = 'not-a-date'"),
            'invalid expires',
            tag='09a',
        )
        self._assert_invalid_allowlist_fails(
            base.replace(f"expires = '{_iso(DAYS)}'\n", ''),
            'missing expires',
            tag='09b',
        )
        self._assert_invalid_allowlist_fails(
            base.replace(f"expires = '{_iso(DAYS)}'", "expires = '2027-01-31T00:00:00'"),
            'expected YYYY-MM-DD',
            tag='09c',
        )

    def test_10_pip_audit_fails_when_allowlist_empty_reason_or_owner(self):
        base = self._allowlist(expires=_iso(DAYS))
        self._assert_invalid_allowlist_fails(
            base.replace("reason = 'No fix available.'", "reason = ''"), 'reason', tag='10a'
        )
        self._assert_invalid_allowlist_fails(base.replace("owner = '@tradermonty'", "owner = ''"), 'owner', tag='10b')

    def test_11_pip_audit_warns_on_stale_allowlist_entry_without_failing(self):
        allowlist = _write(FIXTURES / 'test_11_allowlist.toml', self._allowlist(expires=_iso(DAYS)))
        audit = _write(FIXTURES / 'test_11_audit.json', '{"dependencies": [], "fixes": []}')
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('STALE', out)

    def test_12_pip_audit_fails_on_empty_or_invalid_json(self):
        allowlist = _write(FIXTURES / 'test_12_allowlist.toml', self._allowlist(expires=_iso(DAYS)))
        for name, content in ('test_12_invalid.json', '{not json'), ('test_12_empty.json', ''):
            audit = _write(FIXTURES / name, content)
            rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
            self.assertEqual(rc, 1, out)
            self.assertIn('MALFORMED-INPUT', out)

    def test_22_pip_audit_fails_when_allowlist_file_not_found(self):
        audit = _write(FIXTURES / 'test_22_audit.json', '{"dependencies": [], "fixes": []}')
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(FIXTURES / 'does_not_exist.toml')])
        self.assertEqual(rc, 1, out)
        self.assertIn('MALFORMED-INPUT', out)

    def test_23_pip_audit_fails_on_scanner_error_code(self):
        allowlist = _write(FIXTURES / 'test_23_allowlist.toml', '')
        audit = _write(FIXTURES / 'test_23_audit.json', '{"dependencies": [], "fixes": []}')
        rc, out, _ = _run_gate(
            [
                'pip-audit',
                str(audit),
                '--allowlist',
                str(allowlist),
                '--scanner-code',
                '2',
                '--today',
                _iso(DAYS * 0),
            ]
        )
        self.assertEqual(rc, 1, out)
        self.assertIn('SCANNER-FAILURE', out)

    def test_24_pip_audit_prefers_renewed_exception_over_expired_duplicate(self):
        renewed = self._allowlist(expires=_iso(DAYS), vid='GHSA-1111-aaaa-bbbb')
        expired = self._allowlist(expires=_iso(-DAYS), vid='GHSA-1111-aaaa-bbbb')
        allowlist = _write(FIXTURES / 'test_24_allowlist.toml', expired + '\n' + renewed)
        audit = _write(
            FIXTURES / 'test_24_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 0, out)
        self.assertIn('ALLOWED', out)
        self.assertNotIn('EXPIRED', out)

    def test_25_pip_audit_fails_when_allowlist_entry_not_a_table(self):
        allowlist = _write(FIXTURES / 'test_25_allowlist.toml', 'exceptions = ["not-a-table"]')
        audit = _write(
            FIXTURES / 'test_25_audit.json',
            _vuln_json('vuln-pkg', '1.0.0', 'GHSA-1111-aaaa-bbbb'),
        )
        rc, out, _ = _run_gate(['pip-audit', str(audit), '--allowlist', str(allowlist), '--today', _iso(DAYS * 0)])
        self.assertEqual(rc, 1, out)
        self.assertIn('invalid allowlist', out)

    @staticmethod
    def _allowlist(expires: str, vid: str = 'GHSA-1111-aaaa-bbbb') -> str:
        return (
            '[[exceptions]]\n'
            f"id = '{vid}'\n"
            "package = 'vuln-pkg'\n"
            "fix_version = '2.0.0'\n"
            "reason = 'No fix available.'\n"
            "owner = '@tradermonty'\n"
            f"expires = '{expires}'\n"
        )

    # secrets-check gate

    @staticmethod
    def _entry(secret, ftype='Secret Keyword', filename='.env.sample', line=1):
        return {
            'type': ftype,
            'filename': filename,
            'hashed_secret': security_gate.hash_secret(secret),
            'is_verified': False,
            'line_number': line,
        }

    def _run_secrets(self, name, live_results, baseline_results):
        baseline = _write(FIXTURES / f'{name}_baseline.json', json.dumps({'results': baseline_results}))
        live = _write(FIXTURES / f'{name}_live.json', json.dumps({'results': live_results}))
        return _run_gate(['secrets-check', str(live), '--baseline', str(baseline)])

    def test_13_secrets_pass_when_matches_baseline(self):
        rc, out, _ = self._run_secrets(
            'test_13', {'.env.sample': [self._entry('placeholder')]}, {'.env.sample': [self._entry('placeholder')]}
        )
        self.assertEqual(rc, 0, out)

    def test_14_secrets_passes_audited_is_secret_false_entry(self):
        entry = self._entry('placeholder')
        entry['is_secret'] = False
        entry['is_used'] = False
        entry_live = self._entry('placeholder')
        rc, out, _ = self._run_secrets('test_14', {'.env.sample': [entry_live]}, {'.env.sample': [entry]})
        self.assertEqual(rc, 0, out)
        self.assertNotIn('FAIL', out)

    def test_15_secrets_fails_on_new_finding(self):
        dummy_15 = 'new-real-secret'  # pragma: allowlist secret
        rc, out, _ = self._run_secrets('test_15', {'.env.sample': [self._entry(dummy_15)]}, {})
        self.assertEqual(rc, 1)
        self.assertIn('NEW', out)

    def test_16_secrets_report_contains_no_raw_secret_value(self):
        raw_secret = 'sup3r-s3cr3t-l1v3-v4lu3'  # pragma: allowlist secret
        rc, out, err = self._run_secrets(
            'test_16',
            {'.env.sample': [self._entry(raw_secret)]},
            {},
        )
        self.assertEqual(rc, 1)
        self.assertIn('NEW', out)  # a non-vacuous report
        prefix = security_gate.hash_secret(raw_secret)[: security_gate.REDACTED_HASH_PREFIX_LEN]
        self.assertIn(prefix, out)  # redacted hash present for triage
        self.assertNotIn(raw_secret, out)  # raw value absent from stdout
        self.assertNotIn(raw_secret, err)  # raw value absent from stderr
        self.assertNotIn(security_gate.hash_secret(raw_secret), out)  # full hash absent

    def test_17_secrets_fails_on_audited_unused_entries(self):
        # Case A: is_used explicitly false.
        entry = self._entry('placeholder')
        entry['is_secret'] = True
        entry['is_used'] = False
        rc, out, _ = self._run_secrets(
            'test_17a', {'.env.sample': [self._entry('placeholder')]}, {'.env.sample': [entry]}
        )
        self.assertEqual(rc, 1)
        self.assertIn('AUDIT-DERELICT', out)
        # Case B: is_secret true but is_used absent (partial audit) must also fail.
        entry2 = self._entry('placeholder')
        entry2['is_secret'] = True
        rc, out, _ = self._run_secrets(
            'test_17b', {'.env.sample': [self._entry('placeholder')]}, {'.env.sample': [entry2]}
        )
        self.assertEqual(rc, 1)
        self.assertIn('AUDIT-DERELICT', out)

    def test_18_secrets_allows_removed_finding_with_prune_hint_only(self):
        rc, out, _ = self._run_secrets('test_18', {}, {'.env.sample': [self._entry('placeholder')]})
        self.assertEqual(rc, 0, out)
        self.assertIn('PRUNE', out)

    def test_19_secrets_fails_on_empty_or_invalid_json(self):
        baseline = _write(FIXTURES / 'test_19_baseline.json', json.dumps({'results': {}}))
        for name, content in ('test_19_invalid.json', '{oops'), ('test_19_empty.json', ''):
            live = _write(FIXTURES / name, content)
            rc, out, _ = _run_gate(['secrets-check', str(live), '--baseline', str(baseline)])
            self.assertEqual(rc, 1, out)
            self.assertIn('MALFORMED-INPUT', out)

    def test_20_secrets_fails_on_mix_of_stale_and_new_findings(self):
        dummy_20 = 'tokento'  # pragma: allowlist secret
        rc, out, _ = self._run_secrets(
            'test_20',
            {'other.py': [self._entry(dummy_20, filename='other.py')]},
            {'.env.sample': [self._entry('placeholder')]},
        )
        self.assertEqual(rc, 1)
        self.assertIn('PRUNE', out)  # baseline entry no longer present in live scan
        self.assertIn('NEW', out)  # other.py secret is new

    def test_21_secrets_hash_drift_fails_with_review_guidance(self):
        baseline_entry = self._entry('placeholder', line=1)
        live_entry = self._entry('placeholder-changed', line=3)  # pragma: allowlist secret
        rc, out, _ = self._run_secrets('test_21', {'.env.sample': [live_entry]}, {'.env.sample': [baseline_entry]})
        self.assertEqual(rc, 1)
        self.assertIn('COMMENT', out)
        self.assertIn('baseline', out.lower())
        self.assertNotIn('PRUNE', out)  # drift consumes the superseded baseline entry

    def test_26_secrets_fails_on_non_dict_finding_entry(self):
        baseline = _write(FIXTURES / 'test_26_baseline.json', json.dumps({'results': {}}))
        live = _write(FIXTURES / 'test_26_live.json', json.dumps({'results': {'.env.sample': ['oops']}}))
        rc, out, _ = _run_gate(['secrets-check', str(live), '--baseline', str(baseline)])
        self.assertEqual(rc, 1, out)
        self.assertIn('MALFORMED-INPUT', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
