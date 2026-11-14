"""Process-wide Python network audit hook for deterministic fixture tests.

This is a regression guard, not a sandbox for hostile code or native clients.
Only event names are recorded; hostnames, tokens and payloads are never logged.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / 'tests' / 'offline_bootstrap'
_INSTALLED = False
EVENTS = {
    'socket.connect',
    'socket.getaddrinfo',
    'socket.gethostbyname',
    'socket.gethostbyaddr',
    'socket.getnameinfo',
    'socket.sendto',
    'socket.sendmsg',
}


def offline_environment(log, environment=None):
    env = dict(os.environ if environment is None else environment)
    env['BREADTH_OFFLINE_LOG'] = str(Path(log).resolve())
    entries = [str(BOOTSTRAP), str(ROOT)]
    coverage_module = sys.modules.get('coverage')
    if coverage_module is not None and coverage_module.__file__:
        entries.append(str(Path(coverage_module.__file__).resolve().parent.parent))
    entries.extend(p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and p not in entries)
    env['PYTHONPATH'] = os.pathsep.join(entries)
    env['SKIP_API_TESTS'] = 'true'
    env['DOTENV_DISABLED'] = '1'
    return env


def install(log):
    global _INSTALLED
    os.environ.update(offline_environment(log))
    if _INSTALLED:
        return
    log = Path(log).resolve()

    def audit(event, args):
        if event == 'subprocess.Popen':
            env = args[3]
            if env is not None:
                # Explicit env/PYTHONPATH in an E2E must not discard the guard.
                # Dedicated negative probes may choose their own attempt log.
                child_log = env.get('BREADTH_OFFLINE_LOG') or str(log)
                env.update(offline_environment(child_log, env))
                for key in ('COVERAGE_PROCESS_CONFIG', 'COVERAGE_PROCESS_START'):
                    if key in os.environ:
                        env.setdefault(key, os.environ[key])
        if event in EVENTS:
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (event + '\n').encode('ascii'))
            finally:
                os.close(fd)
            raise OSError(f'Network forbidden in API-free validation: {event}')

    sys.addaudithook(audit)
    _INSTALLED = True
