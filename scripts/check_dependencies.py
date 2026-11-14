"""Check the native uv lock and the generated pip/audit compatibility export."""

import argparse
import shutil
import subprocess  # nosec B404: invokes the pinned uv tool with fixed arguments, without a shell
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ARGS = [
    'uv',
    'export',
    '--locked',
    '--all-extras',
    '--group',
    'test',
    '--no-default-groups',
    '--no-emit-project',
    '--no-header',
]
HEADER = '# Generated from uv.lock; do not edit. Refresh: python scripts/check_dependencies.py --write\n'


def check(project_dir=ROOT, *, write=False):
    uv = shutil.which('uv')
    if uv is None:
        raise SystemExit('uv 0.12.17 is required; see docs/dependencies.md')
    subprocess.run([uv, 'lock', '--check'], cwd=project_dir, check=True)  # nosec B603
    profiles = {
        'requirements.txt': EXPORT_ARGS[1:],
        'requirements-analysis.txt': [
            'export',
            '--locked',
            '--no-default-groups',
            '--no-emit-project',
            '--no-header',
        ],
    }
    for filename, arguments in profiles.items():
        exported = subprocess.run([uv, *arguments], cwd=project_dir, check=True, capture_output=True, text=True).stdout  # nosec B603
        expected = HEADER + exported
        path = project_dir / filename
        if write:
            path.write_text(expected, encoding='utf-8')
        elif not path.exists() or path.read_text(encoding='utf-8') != expected:
            raise SystemExit(f'{filename} is stale; run python scripts/check_dependencies.py --write')
    print('Dependency lock and requirements exports match.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Regenerate requirements.txt from the checked lock')
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    check(args.project_dir, write=args.write)


if __name__ == '__main__':
    main()
