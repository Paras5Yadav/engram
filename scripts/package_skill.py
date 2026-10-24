"""Bundle the portable skill with an already built local wheel."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(wheel: Path, output: Path) -> dict:
    wheel, output = wheel.resolve(), output.resolve()
    with zipfile.ZipFile(wheel) as archive:
        if 'engram/api.py' not in archive.namelist():
            raise ValueError('wheel does not contain the Engram engine')
    folder = ROOT / 'skills' / 'engram-memory'
    output.parent.mkdir(parents=True, exist_ok=True)
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ('SKILL.md', 'scripts/run.py'):
            archive.write(folder / relative, f'engram-memory/{relative}')
        archive.write(wheel, f'engram-memory/assets/{wheel.name}')
        archive.writestr('engram-memory/assets/manifest.json', json.dumps({'wheel': wheel.name, 'sha256': wheel_hash}, indent=2) + '\n')
    return {'path': str(output), 'wheel_sha256': wheel_hash}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'engram-memory.zip')
    args = parser.parse_args()
    print(json.dumps(package(args.wheel, args.output)))
