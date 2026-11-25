import json
import sys
import re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import requests
from network_get import safe_get

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'private'
import os
EXPECTED = os.environ.get('YATASKS_REQUESTER_ID')

def request(method, path, payload=None, allow_exam_skill_write=False):
    if method not in ('GET', 'POST', 'PUT') or any(x in path for x in ('/open', '/moderate', '/close', '/archive')):
        raise ValueError('Only closed preparation is authorized')
    if method == 'POST' and path not in ('projects', 'skills', 'pools'):
        raise ValueError('Creation restricted to projects, skills and closed pools')
    skill_write = method == 'PUT' and path == 'user-skills' and allow_exam_skill_write
    if skill_write and (not payload or payload.get('skill_id') not in ('21393','21394')):
        raise ValueError('Only bbox exam skills may be assigned')
    if method == 'PUT' and not skill_write and not re.fullmatch(r'(projects|pools)/\d+', path):
        raise ValueError('Only editing existing projects and pools is authorized')
    values = dict(line.split('=', 1) for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines() if '=' in line and not line.startswith('#'))
    url = 'https://tasks.yandex.ru/api/v1/' + path
    headers = {'Authorization': 'OAuth ' + values['YATASKS_TOKEN'].strip()}
    if method == 'GET':
        response = safe_get(url, headers=headers)
    else:
        response = requests.request(method, url, headers=headers, json=payload,
                                    timeout=(15, 30), allow_redirects=False)
    if 300 <= response.status_code < 400:
        raise RuntimeError('Unexpected redirect: request was not repeated')
    if not response.ok:
        raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2500]}')
    return response.json()

def save(name, data):
    PRIVATE.mkdir(exist_ok=True)
    (PRIVATE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def identity():
    if not EXPECTED:
        raise ValueError('Set YATASKS_REQUESTER_ID before accessing a live account')
    account = request('GET', 'requester')
    assert account['id'] == EXPECTED, 'Unexpected requester'
    safe = {k: account.get(k) for k in ('id', 'balance', 'public_name', 'company')}
    save('team_account.json', safe)
    return safe

if __name__ == '__main__':
    print(json.dumps(identity(), ensure_ascii=False))
    for endpoint in ('projects', 'skills'):
        data = request('GET', endpoint + '?limit=100')
        save('existing_' + endpoint + '.json', data)
        print(endpoint, [(x['id'], x.get('public_name', x.get('name')), x.get('status')) for x in data['items']])
