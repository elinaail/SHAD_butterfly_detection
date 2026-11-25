from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch, Mock

from review_transfer import *
from network_get import safe_get, doh_get, is_dns_failure, requests


def fixture(count=20):
    source, assignments, controls = {}, [], []
    for i in range(count):
        photo = str(i)
        url = 'https://example.org/' + photo + '.jpg'
        source[photo] = {'photo_id': photo, 'url': url, 'split': 'main'}
        assignments.append({'id': 'a' + photo, 'pool_id': BBOX_POOL, 'user_id': 'author' + str(i % 3),
                            'status': 'ACCEPTED', 'tasks': [{'id': 't' + photo, 'input_values': {'photo_id': photo, 'image': url}}],
                            'solutions': [{'output_values': {'status': 'no_butterflies', 'result': []}}]})
    for i in range(12):
        photo = 'g' + str(i)
        url = 'https://example.org/' + photo + '.jpg'
        source[photo] = {'photo_id': photo, 'url': url, 'split': 'train'}
        controls.append({'photo_id': photo, 'image': url, 'boxes': [], 'verdict': 'accept' if i < 6 else 'fix',
                         'verified': True, 'annotator': 'human1', 'verified_by': 'human2', 'rationale': 'synthetic test only'})
    return source, assignments, controls


def make_plan(count=20, previous_reviews=(), existing_suites=()):
    source, assignments, controls = fixture(count)
    candidates, participants, held = candidates_from_assignments(assignments, source)
    assert not held
    return build_plan(candidates, participants, controls, source, expected_count=count,
                      previous_reviews=previous_reviews, existing_suites=existing_suites)


def review_answers(plan, worker='reviewer1', verdict='accept'):
    return [{'id': worker + str(i), 'pool_id': REVIEW_POOL, 'user_id': worker, 'status': 'ACCEPTED',
             'tasks': deepcopy(s['tasks']), 'solutions': [{'output_values': {'verdict': verdict}} for _ in s['tasks']]}
            for i, s in enumerate(plan['review_suites'])]


class TransferTests(unittest.TestCase):
    def test_budget_and_mixed_page_union(self):
        plan = make_plan()
        self.assertEqual(plan['budget']['maximum_rub_under_assumption'], '14.00')
        self.assertEqual(len(plan['suites']), 5)
        authors = {c['candidate_id']: c['author_id'] for c in plan['candidates']}
        mixed = False
        for page in plan['suites']:
            expected = {authors[t['input_values']['candidate_id']] for t in page['tasks'] if not t.get('known_solutions')}
            self.assertEqual(set(page['unavailable_for']), expected)
            self.assertEqual(sum(bool(t.get('known_solutions')) for t in page['tasks']), 1)
            self.assertEqual(page['overlap'], 2)
            mixed |= len(expected) > 1
        self.assertTrue(mixed)

    def test_positive_native_rectangle(self):
        source, rows, _ = fixture(1)
        rows[0]['solutions'][0]['output_values'] = {'status': 'has_butterflies', 'result': [
            {'shape': 'rectangle', 'left': 0.1, 'top': 0.2, 'width': 0.3, 'height': 0.4}]}
        candidates, _, held = candidates_from_assignments(rows, source)
        self.assertFalse(held)
        self.assertEqual(candidates[0]['boxes'][0]['shape'], 'rectangle')

    def test_load_error_is_held(self):
        source, rows, _ = fixture(1)
        rows[0]['solutions'][0]['output_values']['status'] = 'load_error'
        candidates, _, held = candidates_from_assignments(rows, source)
        self.assertFalse(candidates)
        self.assertTrue(held)

    def test_invalid_box_held(self):
        source, rows, _ = fixture(1)
        rows[0]['solutions'][0]['output_values'] = {'status': 'has_butterflies', 'result': [{'left': 2}]}
        self.assertTrue(candidates_from_assignments(rows, source)[2])

    def test_source_url_mismatch_held(self):
        source, rows, _ = fixture(1)
        rows[0]['tasks'][0]['input_values']['image'] += '?changed'
        self.assertTrue(candidates_from_assignments(rows, source)[2])

    def test_duplicate_snapshot_deduplicated(self):
        source, rows, _ = fixture(1)
        self.assertEqual(len(candidates_from_assignments(rows + rows, source)[0]), 1)

    def test_conflicting_snapshot_rejected(self):
        source, rows, _ = fixture(1)
        other = deepcopy(rows[0]); other['status'] = 'SUBMITTED'
        with self.assertRaises(ValueError):
            candidates_from_assignments(rows + [other], source)

    def test_two_accepted_candidates_held(self):
        source, rows, _ = fixture(1)
        other = deepcopy(rows[0]); other['id'] = 'other'
        candidates, _, held = candidates_from_assignments(rows + [other], source)
        self.assertFalse(candidates)
        self.assertTrue(held)

    def test_expired_bbox_participant_excluded(self):
        source, rows, controls = fixture(1)
        other = deepcopy(rows[0]); other.update(id='expired', user_id='old_author', status='EXPIRED')
        candidates, participants, held = candidates_from_assignments(rows + [other], source)
        plan = build_plan(candidates, participants, controls, source, expected_count=1)
        self.assertIn('old_author', plan['suites'][0]['unavailable_for'])

    def test_previous_reviewer_excluded_across_revisions(self):
        prior = [{'user_id': 'old_reviewer', 'status': 'EXPIRED', 'tasks': [{'input_values': {'photo_id': '0'}}]}]
        self.assertIn('old_reviewer', make_plan(1, prior)['suites'][0]['unavailable_for'])

    def test_missing_controls_rejected(self):
        source, rows, _ = fixture(1)
        candidates, participants, _ = candidates_from_assignments(rows, source)
        with self.assertRaises(ValueError):
            build_plan(candidates, participants, [], source, expected_count=1)

    def test_unverified_controls_rejected(self):
        source, rows, controls = fixture(1)
        controls[0]['verified_by'] = 'human1'
        candidates, participants, _ = candidates_from_assignments(rows, source)
        with self.assertRaises(ValueError):
            build_plan(candidates, participants, controls, source, expected_count=1)

    def test_repeat_plan_has_same_operation(self):
        self.assertEqual(make_plan()['operation_id'], make_plan()['operation_id'])

    def test_existing_suites_not_recreated(self):
        original = make_plan()
        repeated = make_plan(existing_suites=original['suites'])
        self.assertFalse(repeated['suites'])
        self.assertEqual(len(repeated['skipped_candidate_ids']), 20)
        answers = review_answers(repeated) + review_answers(repeated, 'reviewer2')
        self.assertTrue(audit_reviews(repeated, answers)['review_consensus_complete'])

    def test_changed_existing_candidate_rejected(self):
        suites = make_plan()['suites']
        next(t for t in suites[0]['tasks'] if not t.get('known_solutions'))['input_values']['candidate_revision'] = 2
        with self.assertRaises(ValueError):
            make_plan(existing_suites=suites)

    def test_readback_exclusions_required(self):
        plan = make_plan()
        self.assertTrue(audit_uploaded(plan, deepcopy(plan['suites'])))
        pages = deepcopy(plan['suites']); pages[0]['unavailable_for'] = []
        with self.assertRaises(ValueError):
            audit_uploaded(plan, pages)

    def test_self_review_detected_even_active(self):
        plan = make_plan(1)
        rows = review_answers(plan, 'author0')
        rows[0]['status'] = 'ACTIVE'; rows[0]['solutions'] = []
        self.assertFalse(audit_reviews(plan, rows)['passed'])

    def test_two_independent_accepts(self):
        plan = make_plan()
        report = audit_reviews(plan, review_answers(plan) + review_answers(plan, 'reviewer2'))
        self.assertTrue(report['review_consensus_complete'])

    def test_duplicate_reviewer_not_two_votes(self):
        plan = make_plan(1)
        rows = review_answers(plan)
        other = deepcopy(rows); other[0]['id'] = 'different_assignment'
        self.assertFalse(audit_reviews(plan, rows + other)['passed'])

    def test_disagreement_or_load_error_needs_manual_review(self):
        for verdict in ('fix', 'load_error'):
            plan = make_plan(1)
            report = audit_reviews(plan, review_answers(plan) + review_answers(plan, 'reviewer2', verdict))
            self.assertEqual(report['decisions'][0]['decision'], 'manual_review')
            self.assertFalse(report['review_consensus_complete'])

    def test_partial_page_cost_rounded_up(self):
        self.assertEqual(make_plan(5)['budget']['maximum_rub_under_assumption'], '5.60')

    def test_missing_pilot_candidates_rejected(self):
        source, rows, controls = fixture(1)
        candidates, participants, _ = candidates_from_assignments(rows, source)
        with self.assertRaises(ValueError):
            build_plan(candidates, participants, controls, source)


class NetworkTests(unittest.TestCase):
    def test_tls_hostname_and_no_credential_leak_to_dns(self):
        dns = Mock()
        dns.json.return_value = {'Status': 0, 'Answer': [{'type': 1, 'data': '8.8.8.8'}]}
        pool, raw = Mock(), Mock()
        raw.status = 200; raw.headers = {}; raw.read.return_value = b'{}'
        pool.urlopen.return_value = raw
        with patch('network_get.requests.get', return_value=dns) as lookup, patch('network_get.urllib3.HTTPSConnectionPool', return_value=pool) as factory:
            doh_get('https://tasks.yandex.ru/api/v1/requester', {'Authorization': 'test-only'})
            self.assertNotIn('headers', lookup.call_args.kwargs)
            self.assertEqual(factory.call_args.kwargs['server_hostname'], 'tasks.yandex.ru')
            self.assertEqual(factory.call_args.kwargs['assert_hostname'], 'tasks.yandex.ru')
            self.assertEqual(factory.call_args.kwargs['cert_reqs'], 'CERT_REQUIRED')
            self.assertEqual(pool.urlopen.call_args.kwargs['headers']['Host'], 'tasks.yandex.ru')
            self.assertFalse(pool.urlopen.call_args.kwargs['redirect'])
            self.assertFalse(pool.urlopen.call_args.kwargs['retries'])

    def test_api_mutation_not_retried(self):
        import api
        error = requests.exceptions.ConnectionError(socket.gaierror(11001, 'synthetic DNS failure'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.env').write_text('YATASKS_TOKEN=test-only', encoding='utf-8')
            with patch.object(api, 'ROOT', root), patch('api.requests.request', side_effect=error) as write, patch('api.safe_get') as fallback:
                with self.assertRaises(requests.exceptions.ConnectionError):
                    api.request('PUT', 'pools/' + REVIEW_POOL, {})
                write.assert_called_once()
                fallback.assert_not_called()

    def test_normal_get_no_fallback(self):
        with patch('network_get.requests.get', return_value=Mock()) as get, patch('network_get.doh_get') as fallback:
            safe_get('https://tasks.yandex.ru/api/v1/requester')
            fallback.assert_not_called()
            self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_only_dns_error_gets_fallback(self):
        error = requests.exceptions.ConnectionError(socket.gaierror(11001, 'test DNS failure'))
        with patch('network_get.requests.get', side_effect=error), patch('network_get.doh_get') as fallback:
            safe_get('https://tasks.yandex.ru/api/v1/requester')
            fallback.assert_called_once()

    def test_timeout_not_retried(self):
        with patch('network_get.requests.get', side_effect=requests.exceptions.ConnectTimeout()), patch('network_get.doh_get') as fallback:
            with self.assertRaises(requests.exceptions.ConnectTimeout):
                safe_get('https://tasks.yandex.ru/api/v1/requester')
            fallback.assert_not_called()

    def test_credentials_not_allowed_on_photo_host(self):
        with self.assertRaises(ValueError):
            doh_get('https://inaturalist-open-data.s3.amazonaws.com/photo.jpg', {'Authorization': 'test-only'})

    def test_arbitrary_host_not_allowed(self):
        for url in ('https://example.org', 'http://tasks.yandex.ru', 'https://user:pass@tasks.yandex.ru'):
            with self.assertRaises(ValueError):
                doh_get(url)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('review_transfer_test_report.json').write_text(json.dumps(
        {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
         'passed': result.wasSuccessful(), 'synthetic_data_only': True}, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
