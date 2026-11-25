import ipaddress
import socket
import sys
from urllib.parse import urlsplit

import requests
import urllib3

ALLOWED_HOSTS = {'tasks.yandex.ru', 'inaturalist-open-data.s3.amazonaws.com'}


def is_dns_failure(error):
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (socket.gaierror, urllib3.exceptions.NameResolutionError)):
            return True
        for child in (getattr(item, '__cause__', None), getattr(item, '__context__', None),
                      getattr(item, 'reason', None), *getattr(item, 'args', ())):
            if isinstance(child, BaseException):
                pending.append(child)
    return False


def doh_get(url, headers=None, max_bytes=40_000_000):
    target = urlsplit(url)
    host = target.hostname
    if (target.scheme != 'https' or host not in ALLOWED_HOSTS or target.username
            or target.password or target.port not in (None, 443) or target.fragment):
        raise ValueError('DNS fallback destination is not allowed')
    headers = dict(headers or {})
    if any(k.lower() == 'host' for k in headers):
        raise ValueError('Caller cannot override Host')
    if host != 'tasks.yandex.ru' and any(k.lower() == 'authorization' for k in headers):
        raise ValueError('Credentials may only be sent to the task API')
    dns = requests.get('https://dns.google/resolve', params={'name': host, 'type': 'A'},
                       timeout=(10, 15), allow_redirects=False)
    dns.raise_for_status()
    record = dns.json()
    if record.get('Status') != 0:
        raise RuntimeError('Public DNS returned no successful answer')
    addresses = [r['data'] for r in record.get('Answer', []) if r.get('type') == 1]
    if not addresses or not ipaddress.ip_address(addresses[0]).is_global:
        raise RuntimeError('Public DNS did not return a public address')
    pool = urllib3.HTTPSConnectionPool(
        addresses[0], server_hostname=host, assert_hostname=host,
        cert_reqs='CERT_REQUIRED', ca_certs=requests.certs.where(),
        timeout=urllib3.Timeout(connect=10, read=30))
    raw = None
    try:
        path = target.path or '/'
        if target.query:
            path += '?' + target.query
        raw = pool.urlopen('GET', path, headers={**headers, 'Host': host},
                           preload_content=False, retries=False, redirect=False)
        body = raw.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError('Response exceeded the download limit')
        response = requests.Response()
        response.status_code = raw.status
        response.headers.update(raw.headers)
        response._content = body
        response.url = url
        response.encoding = requests.utils.get_encoding_from_headers(response.headers)
        response.dns_fallback_used = True
        return response
    finally:
        if raw is not None:
            raw.close()
        pool.close()


def safe_get(url, headers=None, timeout=(15, 30)):
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False)
            response.dns_fallback_used = False
            return response
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
        except requests.exceptions.ConnectionError as error:
            if not is_dns_failure(error):
                raise
            return doh_get(url, headers)
