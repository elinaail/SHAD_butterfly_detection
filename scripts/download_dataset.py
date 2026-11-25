"""Download final CC0 originals; verify the hashes of model inputs."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = [json.loads(s) for s in (ROOT/'data/main.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        target = ROOT/'models'/row['local_path']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']:
            continue
        if row['license'].lower() != 'cc0': raise ValueError('Expected CC0')
        with urlopen(row['url'], timeout=90) as response: content = response.read()
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError(f"Source changed: {row['photo_id']}")
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+'.tmp');tmp.write_bytes(content);tmp.replace(target)
        print(row['photo_id'])
    print(f'{len(rows)} originals verified')

if __name__ == '__main__': main()
