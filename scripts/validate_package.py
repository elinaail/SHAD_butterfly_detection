"""Offline integrity checks. Does not call labeling services or model APIs."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def keyed(data):
    out = {str(r['photo_id']): r for r in data}
    assert len(out) == len(data), 'Duplicate photo ID'
    return out

gold = rows(ROOT/'gold_400_for_metrics/gold_400_v3.jsonl')
old = rows(ROOT/'gold_400_for_metrics/gold_400_v2.jsonl')
assert [r['photo_id'] for r in gold] == [r['photo_id'] for r in old]
assert {str(a['photo_id']) for a,b in zip(gold,old) if a != b} == {'7775492','16391324'}
assert len(gold) == 400 and sum(len(r['boxes']) for r in gold) == 361
assert sum(not r['boxes'] for r in gold) == 122
ids = set(keyed(gold))
for p in (ROOT/'model_results_400').glob('*.jsonl'):
    data=keyed(rows(p)); assert set(data)==ids
    for r in data.values():
        assert not r.get('error')
        for b in r['boxes']:
            x,y,w,h=[b[k] for k in ['left','top','width','height']]
            assert all(math.isfinite(v) for v in [x,y,w,h])
            assert min(x,y)>=0 and min(w,h)>0 and x+w<=1+1e-6 and y+h<=1+1e-6
participants=defaultdict(set)
for phase in ['pilot','main','replacements','repair']:
    for a in json.loads((ROOT/f'crowd/responses/bbox_{phase}.json').read_text(encoding='utf-8')):
        for t in a['tasks']:
            pid=t['input_values'].get('photo_id')
            if pid is not None:participants[str(pid)].add(a['user_id'])
checked=0
for phase in ['pilot','main','replacements']:
    for a in json.loads((ROOT/f'crowd/responses/review_{phase}.json').read_text(encoding='utf-8')):
        for t in a['tasks']:
            pid=str(t['input_values'].get('photo_id'))
            if pid not in participants:continue
            assert a['user_id'] not in participants[pid], ('self-review',pid)
            if a['status']=='ACCEPTED': checked+=1
        assert a['user_id'].startswith('worker_')
budget=json.loads((ROOT/'crowd/budget.json').read_text(encoding='utf-8'))
assert abs(sum(budget['stages_including_fee'].values())-budget['spent'])<1e-6
assert abs(budget['limit']-budget['spent']-budget['remaining'])<1e-6
assert hashlib.sha256((ROOT/'contest/predictions.jsonl').read_bytes()).hexdigest()=='146014cd76a9639180aa3a8bbc4b84b470e953adc1f91a7f98c1b5c753ad98da'
print(f'OK: 400 IDs, gold v3, seven model files, {checked} accepted review votes without self-review, budget and contest checksum')
