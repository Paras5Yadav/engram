"""Bootstrap a bundled Engram wheel locally, then run its JSON-capable CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import venv
from contextlib import contextmanager
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]


@contextmanager
def install_lock(path):
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            handle.write(b'0')
            handle.flush()
        else:
            import fcntl
        deadline = time.monotonic() + 60
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('another Engram setup is still running')
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def runtime():
    if sys.version_info < (3, 11):
        raise RuntimeError('Engram requires Python 3.11 or later')
    assets = SKILL / 'assets'
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    name = manifest['wheel']
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError('invalid bundled wheel name')
    wheel = assets / name
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('bundled wheel does not match its manifest')
    home = SKILL / '.runtime'
    python = home / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    stamp = home / 'engram-wheel.sha256'
    with install_lock(SKILL / '.setup.lock'):
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(home)
        if not stamp.exists() or stamp.read_text().strip() != digest:
            print('Preparing the local Engram runtime...', file=sys.stderr)
            subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                            '--no-index', '--no-deps', '--force-reinstall', str(wheel)], check=True, stdout=sys.stderr)
            stamp.write_text(digest + '\n')
    return python, wheel


def main():
    try:
        python, wheel = runtime()
        arguments = sys.argv[1:]
        if arguments[:1] == ['setup']:
            parser = argparse.ArgumentParser(description='Prepare optional provider dependencies')
            parser.add_argument('--provider', choices=('base', 'typesafe', 'semantic', 'laya'), default='base')
            options = parser.parse_args(arguments[1:])
            if options.provider != 'base':
                with install_lock(SKILL / '.setup.lock'):
                    subprocess.run([str(python), '-I', '-m', 'pip', 'install', '--disable-pip-version-check',
                                    f'{wheel}[{options.provider}]'], check=True, stdout=sys.stderr)
            print(json.dumps({'ready': True, 'provider': options.provider}))
            return 0
        return subprocess.call([str(python), '-I', '-m', 'engram', *arguments])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': {'type': type(error).__name__, 'message': str(error)}}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
