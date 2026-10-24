"""Reproducible retrieval evaluation over explicitly labeled local JSON data.

This measures evidence retrieval; generated-answer accuracy needs separate evaluation.
Remote decisions are used only if the supplied configuration selects them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from . import __version__
from .api import Engram
from .config import EngramConfig, load_config


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))] if ordered else None


def evaluate(payload: dict, *, config: EngramConfig | None = None, backend=None, encoder=None) -> dict:
    if not isinstance(payload, dict) or set(payload) != {'observations', 'queries'}:
        raise ValueError('evaluation data must contain observations and queries')
    observations, queries = payload['observations'], payload['queries']
    if not isinstance(observations, list) or not isinstance(queries, list) or not queries:
        raise ValueError('observations and nonempty queries must be arrays')
    labels = []
    for item in observations:
        if not isinstance(item, dict) or not isinstance(item.get('label'), str) or not item['label']:
            raise ValueError('each observation needs a nonempty label')
        labels.append(item['label'])
    if len(labels) != len(set(labels)):
        raise ValueError('observation labels must be unique')
    for query in queries:
        if not isinstance(query, dict) or not isinstance(query.get('question'), str) or not query['question'].strip():
            raise ValueError('each query needs a nonempty question')
        for name in ('expected', 'forbidden'):
            values = query.get(name, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or v not in labels for v in values):
                raise ValueError(f'query {name} must contain known observation labels')
        if 'no_match' in query and type(query['no_match']) is not bool:
            raise ValueError('no_match must be boolean')
    config = config or EngramConfig()
    output, latencies, precisions = [], [], []
    expected_count = hit_count = no_match_count = no_match_correct = forbidden_count = calls = fallbacks = 0
    with tempfile.TemporaryDirectory(prefix='engram-evaluation-') as directory:
        with Engram(directory, config=config, backend=backend, encoder=encoder) as app:
            start = time.monotonic()
            writes = app.ingest_many([{k: v for k, v in item.items() if k != 'label'} for item in observations])
            build_seconds = time.monotonic() - start
            ids = {label: result.get('node_id') for label, result in zip(labels, writes)}
            for query in queries:
                expected = {ids[label] for label in query.get('expected', [])}
                forbidden = {ids[label] for label in query.get('forbidden', [])}
                if None in expected or None in forbidden:
                    raise ValueError('gold labels must identify a retained memory')
                start = time.monotonic()
                result = app.query(query['question'], top_k=query.get('top_k'))
                latency = (time.monotonic() - start) * 1000
                latencies.append(latency)
                found = set(result['trace']['evidence_ids'])
                hits = found & expected
                if expected:
                    hit_count += len(hits)
                    expected_count += len(expected)
                    precisions.append(len(hits) / len(found) if found else 0)
                if 'no_match' in query:
                    no_match_count += 1
                    no_match_correct += (not found) == query['no_match']
                forbidden_count += len(found & forbidden)
                calls += result['trace']['calls']
                fallbacks += result['trace']['fallbacks']
                output.append({'question': query['question'], 'expected_ids': sorted(expected),
                               'found_ids': sorted(found), 'missing_ids': sorted(expected - found),
                               'forbidden_ids_returned': sorted(found & forbidden),
                               'latency_ms': round(latency, 3), 'trace': result['trace']})
            identity = app.store.encoder.model_id
            actual_backend = app.backend.name
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {'kind': 'retrieval-evaluation', 'version': __version__, 'source_sha256': source.hexdigest(),
            'dataset_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            'config': config.to_dict(), 'encoder': identity, 'backend': actual_backend,
            'metrics': {'queries': len(queries), 'gold_evidence_recall': hit_count / expected_count if expected_count else None,
                        'mean_gold_precision': statistics.mean(precisions) if precisions else None,
                        'no_match_accuracy': no_match_correct / no_match_count if no_match_count else None,
                        'forbidden_evidence_count': forbidden_count, 'controller_calls': calls, 'fallbacks': fallbacks,
                        'build_seconds': round(build_seconds, 3), 'query_p50_ms': round(statistics.median(latencies), 3),
                        'query_p95_ms': round(percentile(latencies, .95), 3)},
            'queries': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    report = evaluate(payload, config=load_config(args.config))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(Path(args.output).resolve()), 'metrics': report['metrics']}))


if __name__ == '__main__':
    main()
