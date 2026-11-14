"""Activate the inherited test network guard before Python imports application code."""

import os

if os.environ.get('BREADTH_OFFLINE_LOG'):
    from scripts.offline_guard import install

    install(os.environ['BREADTH_OFFLINE_LOG'])

# Needed also when coverage is available through PYTHONPATH rather than an
# installed .pth file. process_startup is idempotent with coverage's own .pth.
if os.environ.get('COVERAGE_PROCESS_CONFIG') or os.environ.get('COVERAGE_PROCESS_START'):
    import coverage

    coverage.process_startup()
