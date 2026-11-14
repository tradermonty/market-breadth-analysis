"""Run Python fixture commands with an inherited network-attempt log."""

import argparse
import subprocess  # nosec B404: developer-supplied argument vector, no shell
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.offline_guard import offline_environment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a Python command is required after --')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    # Never erase evidence from an earlier invocation.
    with args.log.open('x'):
        pass
    result = subprocess.run(command, env=offline_environment(args.log))  # nosec B603
    if args.log.stat().st_size:
        print('FAIL: a network attempt occurred (see attempt log).', file=sys.stderr)
        return 1
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
