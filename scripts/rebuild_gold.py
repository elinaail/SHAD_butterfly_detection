"""Reconstruct current gold from crowd candidates and recorded adjudications."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def boxes(values): return [{k:b[k] for k in ['left','top','width','height']} for b in values]

def main():
    result={}
    for phase in ['pilot','main']:
        for r in read(f'crowd/candidates/{phase}_2026_11_18.json'):
            result[int(r['photo_id'])]=boxes(r['boxes'])
    for r in read('crowd/adjudications/pilot_corrections.json'):
        result[int(r['photo_id'])]=boxes(r['after'])
    for phase in ['batch01','batch02','batch03','batch04','meow_adjudications']:
        for r in read(f'crowd/adjudications/{phase}.json'):
            result[int(r['photo_id'])]=boxes(r['boxes'])
    for r in read('gold_400_for_metrics/replacements.json')['selected']:
        del result[int(r['replaces'])]
    for r in read('crowd/candidates/replacements_2026_11_18.json'):
        result[int(r['photo_id'])]=boxes(r['boxes'])
    for r in read('gold_400_for_metrics/corrections_v3.json'):
        assert result[r['photo_id']]==r['before']
        result[r['photo_id']]=r['after']
    stored=[json.loads(s) for s in (ROOT/'gold_400_for_metrics/gold_400_v3.jsonl').read_text(encoding='utf-8').splitlines()]
    assert result=={r['photo_id']:r['boxes'] for r in stored}, 'Reconstructed gold differs'
    print('OK: gold v3 reproduced from 400 original candidates, adjudications, three replacements and two class corrections')

if __name__=='__main__':main()
