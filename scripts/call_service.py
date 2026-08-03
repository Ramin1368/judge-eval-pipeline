from __future__ import annotations
import csv
import json
import sys
import urllib.request
from pathlib import Path


def _rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main():
    url = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'http://localhost:8080'
    data = Path('data')
    labeled = [
        {'prompt': r['prompt'], 'response_a': r['response_a'], 'response_b': r['response_b'],
         'preferred': r['preferred'], 'annotator_id': r.get('annotator_id')}
        for r in _rows(data / 'preferences.csv')
    ]
    policy_outputs = [
        {'prompt': r['prompt'], 'policy_a': r['policy_a'], 'policy_b': r['policy_b']}
        for r in _rows(data / 'policy_outputs.csv')
    ]
    payload = {
        'judge': 'heuristic', 'policy_a': 'policy_a', 'policy_b': 'policy_b',
        'labeled': labeled, 'policy_outputs': policy_outputs,
    }
    req = urllib.request.Request(
        url + '/compare', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read().decode())
    print('winner:', d['winner'])
    print('win_rate_b:', d['win_rate_b'])
    print('ci:', d['ci'])
    print('p_value:', d['p_value'])
    print('judge_kappa:', d['cohen_kappa'])
    print('judge_trustworthy:', d['judge_trustworthy'])


if __name__ == '__main__':
    main()
