"""Exercise the actual skill bundle outside the source checkout, without indexes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(bundle: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix='engram-skill-check-') as directory:
        directory = Path(directory)
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(directory)
        runner = directory / 'engram-memory/scripts/run.py'
        store = directory / 'user-project/.engram-memory'
        env = dict(os.environ, PIP_NO_INDEX='1', PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
        for name in ('OPENAI_API_KEY', 'ENGRAM_API_KEY', 'TYPESAFE_API_KEY', 'ENGRAM_CONTROL_API_KEY', 'ENGRAM_ANSWER_API_KEY'):
            env.pop(name, None)
        def run(*arguments, expected_code=0):
            result = subprocess.run([sys.executable, str(runner), *arguments], cwd=directory,
                                    env=env, capture_output=True, text=True, timeout=40, check=False)
            if result.returncode != expected_code:
                raise RuntimeError(result.stdout + result.stderr)
            return json.loads(result.stdout)
        common = ('--persist', str(store), '--scope', 'demo', '--offline', '--json')
        first = run('remember', *common, '--text', 'Mira prefers tea.')
        result = run('query', *common, '--question', 'What does Mira prefer?')
        if result['evidence'][0]['node']['node_id'] != first['node_id']:
            raise AssertionError('installed skill failed recall')
        exported = run('export', *common, '--output', str(directory / 'export.json'))
        if len(json.loads(Path(exported['path']).read_text(encoding='utf-8'))['nodes']) != 1:
            raise AssertionError('installed skill failed export')
        if not run('forget', *common, '--id', str(first['node_id']))['deleted']:
            raise AssertionError('installed skill failed deletion')
        if run('query', *common, '--question', 'What does Mira prefer?')['trace']['stop_reason'] != 'no_match':
            raise AssertionError('deleted memory remained queryable')
        manifest = json.loads((runner.parents[1] / 'assets/manifest.json').read_text(encoding='utf-8'))
        wheel = runner.parents[1] / 'assets' / manifest['wheel']
        wheel.write_bytes(wheel.read_bytes() + b'corrupted')
        if 'error' not in run('query', *common, '--question', 'What does Mira prefer?', expected_code=2):
            raise AssertionError('corrupt wheel was not rejected')
    return {'passed': True, 'checks': ['offline_bootstrap', 'recall', 'export', 'forget', 'corrupt_wheel_rejection']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.bundle.resolve())))
