import json
from pathlib import Path
from urllib.parse import urlencode

SKILLS = {'score': '21393', 'gate': '21394'}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def get_value(api_call, user_id, skill_id):
    response = api_call('GET', 'user-skills?' + urlencode({'user_id':user_id,'skill_id':skill_id,'limit':100}))
    items = response.get('items', [])
    if response.get('has_more') or len(items) > 1:
        raise RuntimeError('Ambiguous existing skill records')
    if not items:
        return None
    row = items[0]
    if row['user_id'] != user_id or row['skill_id'] != skill_id:
        raise RuntimeError('Skill response belongs to another worker or skill')
    return float(row['exact_value'])

def issue_skills(report, api_call, journal_path):
    if report.get('decision') not in ('pass', 'fail'):
        return {'status':'held_for_review','writes':0}
    if not report.get('user_id') or not report.get('assignment_id') or not report.get('gold_hash') or not report.get('assignment_hash'):
        raise ValueError('Missing grading provenance')
    expected_gate = 100 if report['decision'] == 'pass' else 0
    if report['gate'] != expected_gate or not 0 <= report['score'] <= 100:
        raise ValueError('Inconsistent skill plan')
    path = Path(journal_path)
    journal = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    key = str(report['assignment_id'])
    signature = {k:report[k] for k in ('user_id','gold_hash','assignment_hash','score','gate')}
    record = journal.get(key, {'signature': signature, 'steps': {}})
    if record['signature'] != signature:
        raise RuntimeError('Assignment or gold changed after skill operation began')
    existing = {field:get_value(api_call, report['user_id'], skill_id) for field,skill_id in SKILLS.items()}
    for field in SKILLS:
        if existing[field] is not None and abs(existing[field] - report[field]) > 0.005:
            raise RuntimeError('Existing skill conflicts with grading; review before overwriting')
    writes = 0
    for field, skill_id in SKILLS.items():
        value = report[field]
        if existing[field] is not None:
            record['steps'][field] = 'verified'
            journal[key] = record
            atomic_json(path, journal)
            continue
        if field in record['steps']:
            raise RuntimeError('Earlier write unresolved or skill removed; no blind retry')
        record['steps'][field] = 'intent'
        journal[key] = record
        atomic_json(path, journal)
        error = None
        try:
            api_call('PUT', 'user-skills', {'skill_id':skill_id,'user_id':report['user_id'],'value':value})
            writes += 1
        except Exception as exc:
            error = type(exc).__name__
        confirmed = get_value(api_call, report['user_id'], skill_id)
        if confirmed is None or abs(confirmed - value) > 0.005:
            raise RuntimeError('Skill write not confirmed; stopped before further writes. ' + str(error or 'readback mismatch'))
        record['steps'][field] = 'verified'
        atomic_json(path, journal)
    return {'status':'verified','writes':writes,'score':report['score'],'gate':report['gate']}
