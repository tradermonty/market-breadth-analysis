"""Make SKIP_API_TESTS a process-wide contract, including collection and children."""

import os
import uuid
from pathlib import Path

import pytest

from scripts.offline_guard import install

OFFLINE = os.environ.get('SKIP_API_TESTS', '').lower() == 'true'
LOG = None
if OFFLINE:
    LOG = Path(os.environ.get('BREADTH_OFFLINE_LOG') or f'/tmp/breadth-pytest-{uuid.uuid4().hex}.log')
    LOG.touch(exist_ok=True)
    os.environ['BREADTH_OFFLINE_LOG'] = str(LOG)
    install(LOG)


def pytest_collection_modifyitems(items):
    if OFFLINE:
        for item in items:
            if item.path.name == 'test_sp500_fetch.py':
                item.add_marker(pytest.mark.skip(reason='SKIP_API_TESTS=true: real provider integration disabled'))


def pytest_sessionfinish(session, exitstatus):
    if LOG is not None and LOG.stat().st_size:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if reporter:
            reporter.write_sep('!', 'API-free contract violated: network attempt recorded')
