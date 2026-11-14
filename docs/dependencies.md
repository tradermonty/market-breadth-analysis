# Locked environments (Issue #17)

## Supported environments

The supported interpreter is CPython **3.11** (`>=3.11,<3.12`), pinned by
`.python-version`. CI targets Ubuntu 24.04 x86_64 and macOS 15; the local
validation is macOS Apple Silicon. CI results must be checked on each PR;
adding a matrix does not establish that those jobs have passed. Python
3.8–3.10 and 3.12+ are not supported by this lock. Windows and Intel macOS
are best-effort until tested; the optional Windows Qt profile is not a
claim of Windows validation.

Install **uv 0.12.17** using your package manager or the [official uv installation
instructions](https://docs.astral.sh/uv/getting-started/installation/).
`tool.uv.required-version` rejects a different resolver version; Actions pins
both the setup action revision and uv version. `uv.lock` records exact versions,
platform markers, distribution URLs and hashes. The source of dependency ranges
is `pyproject.toml`; never edit the lock or generated requirements by hand.

## Install and run

From the repository root:

```sh
uv sync --locked
. .venv/bin/activate
python market_breadth.py --help
python trigger_market_breadth.py --help
```

The default profile contains analysis, single/multi-ETF backtesting, interactive
HTML and CSV output. It does not install a broker SDK, Qt, PNG rendering, or tests.
`--use_saved_data` additionally requires caches covering the requested dates;
a clean checkout has no market data. Missing caches fail explicitly and do not
fetch data. For a deterministic, credential-free synthetic example:

```sh
python scripts/smoke_headless.py
```

This example requires the core-only environment. It blocks sockets and verifies
that saved-data analysis generates HTML and nonempty CSV/summary files even
when optional PNG export fails. It does not publish Pages or validate provider data.

Select a profile explicitly each time you sync; an exact sync removes packages
from a previous profile that are no longer selected:

```sh
uv sync --locked --extra trading --group test  # full API-free regression profile
SKIP_API_TESTS=true python -m pytest tests/ -q
uv sync --locked --extra png                 # analysis + optional PNG
uv sync --locked --extra windows-qt          # Windows GUI backend only
```

Modern Kaleido needs an independently installed compatible Chrome/Chromium.
Without Kaleido or Chrome, PNG export is optional and may fail; inspect the
`PNG export skipped` message. HTML/CSV remain the required outputs. Browser
installation/version is outside the Python lock. Windows Qt is optional; Linux
and macOS headless jobs do not need it. On Windows activate with
`.venv\Scripts\Activate.ps1` instead. The legacy Alpaca SDK can build msgpack
from source: a C build toolchain may be required; build-tool binaries and OS
libraries are outside the Python distribution-version reproducibility claim.
No trading commands or credentials are needed for these regression checks.

## Development tools and compatibility

Tool direct pins in `requirements-lint-tools.txt` and
`requirements-security-tools.txt` remain the source of truth shared with
pre-commit parity checks. Their corresponding dependency groups mirror those
pins; tests reject drift. Use separate environments for tool groups:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/breadth-lint uv sync --locked --only-group lint
/tmp/breadth-lint/bin/ruff check .
/tmp/breadth-lint/bin/ruff format --check .
/tmp/breadth-lint/bin/mypy --config-file pyproject.toml
/tmp/breadth-lint/bin/codespell
UV_PROJECT_ENVIRONMENT=/tmp/breadth-security uv sync --locked --only-group security
/tmp/breadth-security/bin/bandit -c pyproject.toml -r . --exclude archive,venv311,tests
```

Use tool environments outside the source tree so repository-wide scanners do
not scan installed third-party code. Lint and trading are explicitly incompatible:
`types-requests` needs urllib3 >=2 while Alpaca 3.2.0 needs urllib3 <2. The lock
resolves these separately; `--all-groups --all-extras` is intentionally rejected.

`requirements.txt` is a hash-bearing **core + all extras + test** compatibility
export, not the minimal analysis profile. For pip users, inside a fresh Python
3.11 environment:

```sh
python -m pip install --require-hashes -r requirements.txt
```

Platform markers exclude Qt outside Windows. The lock selects different urllib3
versions for core/tool versus trading profiles; installed-version comparisons
must compare the same profile on the same OS/Python. A universal lock does not
mean different optional profiles or operating systems install identical packages.

## Check and update

```sh
uv lock --check
python scripts/check_dependencies.py          # also checks exact export freshness
uv lock --upgrade-package PACKAGE            # targeted update, review the lock diff
python scripts/check_dependencies.py --write
uv sync --locked --extra trading --group test
SKIP_API_TESTS=true python -m pytest tests/ -q
```

For a tool update, change the direct pin file, matching dependency group, and
pre-commit rev together before regenerating the lock. `uv lock --upgrade` is a
broader deliberate refresh, not an installation command. Tests deliberately
corrupt an export and add a dependency to prove stale inputs fail. CI uses
`--locked` (not `--frozen`) and installs each regression profile twice in fresh
environments, comparing distribution/version manifests and uploading them.
It also runs the core-only smoke and full regressions on both OS targets.

The existing Security Scan audits both the resolved `requirements.txt` trading/test
export and the core-only `requirements-analysis.txt` production export with
pip-audit and the fail-closed wrapper, retaining the dated Alpaca msgpack/urllib3
exceptions in `security/pip-audit-allowlist.toml` for the trading profile. The
core-only export audits its urllib3 >=2 fork separately with an empty allowlist
(`security/pip-audit-analysis-allowlist.toml`). Both exports are checked/generated
by the same dependency checker. The job still rejects new
unapproved findings and malformed scanner output. Lint/security tool transitives
are locked but remain outside Gate 1's audit surface; they are not claimed
audited. Pre-commit manages its own isolated tool environments and is not a
fully locked replacement for CI. OS/browser/build-tool reproducibility and
live broker/provider behavior are not established by fixture tests.

Native lock/sync/export behavior follows the [uv locking and syncing
contract](https://docs.astral.sh/uv/concepts/projects/sync/).
