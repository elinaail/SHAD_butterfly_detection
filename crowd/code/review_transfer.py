from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
import uuid

from bbox_exam import fingerprint, normalize_boxes, validate_verification

BBOX_POOL = '6866118'
REVIEW_POOL = '6866120'


def required_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing identifier')
    return value


def native_boxes(boxes):
    return [dict(shape='rectangle', **{key: float(value) for key, value in box.items()}) for box in normalize_boxes(boxes)]


def candidates_from_assignments(assignments, source, bbox_pool=BBOX_POOL):
    by_photo, participants, held, assignment_hashes = defaultdict(list), defaultdict(set), [], {}
    for assignment in assignments:
        aid = required_text(assignment.get('id'))
        if assignment.get('pool_id') != bbox_pool:
            raise ValueError('Unexpected bbox pool')
        digest = fingerprint(assignment)
        if aid in assignment_hashes:
            if assignment_hashes[aid] != digest:
                raise ValueError('Conflicting snapshots of one assignment')
            continue
        assignment_hashes[aid] = digest
        author = required_text(assignment.get('user_id'))
        tasks = assignment.get('tasks', [])
        for task in tasks:
            photo = str(task.get('input_values', {}).get('photo_id', ''))
            if photo:
                participants[photo].add(author)
        if assignment.get('status') != 'ACCEPTED':
            continue
        solutions = assignment.get('solutions', [])
        if not tasks or len(tasks) != len(solutions):
            held.append({'assignment_id': aid, 'reason': 'incomplete_page'})
            continue
        seen = set()
        for task, solution in zip(tasks, solutions):
            inputs = task.get('input_values', {})
            photo = str(inputs.get('photo_id', ''))
            try:
                if photo in seen or photo not in source:
                    raise ValueError('Duplicate or unknown photo')
                seen.add(photo)
                if source[photo]['split'] != 'main' or inputs.get('image') != source[photo]['url']:
                    raise ValueError('Source split or URL mismatch')
                output = solution.get('output_values', {})
                status = output.get('status')
                if status not in ('has_butterflies', 'no_butterflies'):
                    raise ValueError('Image error or unknown status')
                boxes = native_boxes(output.get('result', []))
                if bool(boxes) != (status == 'has_butterflies'):
                    raise ValueError('Status and boxes disagree')
                candidate = {
                    'photo_id': photo, 'image': source[photo]['url'], 'boxes': boxes,
                    'candidate_id': 'bbox-' + fingerprint([bbox_pool, aid, photo])[:32],
                    'candidate_revision': 1, 'author_id': author, 'source_assignment_id': aid,
                    'source_task_id': required_text(task.get('id')), 'source_pool_id': bbox_pool,
                    'source_assignment_hash': digest, 'status': status}
                candidate['content_hash'] = fingerprint(candidate)
                by_photo[photo].append(candidate)
            except (ValueError, TypeError, AttributeError) as error:
                held.append({'assignment_id': aid, 'photo_id': photo, 'reason': str(error)})
    candidates = []
    for photo, values in sorted(by_photo.items()):
        if len(values) != 1:
            held.append({'photo_id': photo, 'reason': 'multiple_accepted_candidates_require_selection'})
        else:
            candidates.append(values[0])
    return candidates, {k: sorted(v) for k, v in participants.items()}, held


def prior_reviewers(assignments):
    result = defaultdict(set)
    for assignment in assignments:
        worker = required_text(assignment.get('user_id'))
        for task in assignment.get('tasks', []):
            photo = str(task.get('input_values', {}).get('photo_id', ''))
            if photo:
                result[photo].add(worker)
    return {k: sorted(v) for k, v in result.items()}


def input_values(candidate):
    return {key: candidate[key] for key in ('photo_id', 'image', 'candidate_id', 'candidate_revision', 'boxes')}


def validate_controls(controls, source):
    if len(controls) != 12 or sum(c.get('verdict') == 'accept' for c in controls) != 6:
        raise ValueError('Pilot requires twelve controls, six accept and six fix')
    result, seen = [], set()
    for control in controls:
        photo = str(control.get('photo_id', ''))
        if photo in seen or photo not in source or source[photo]['split'] != 'train':
            raise ValueError('Controls must be distinct reserved train photos')
        seen.add(photo)
        validate_verification(control)
        if not control.get('rationale') or control.get('verdict') not in ('accept', 'fix'):
            raise ValueError('Missing control rationale or verdict')
        if control.get('image') != source[photo]['url']:
            raise ValueError('Control image mismatch')
        normalized = {**control, 'photo_id': photo, 'boxes': native_boxes(control.get('boxes'))}
        normalized['candidate_id'] = 'control-' + fingerprint(normalized)[:32]
        normalized['candidate_revision'] = 1
        for worker in normalized.get('author_worker_ids', []):
            required_text(worker)
        result.append(normalized)
    return sorted(result, key=lambda c: c['photo_id'])


def build_plan(candidates, participants, controls, source, previous_reviews=(), existing_suites=(),
               expected_count=20, commission='0.40', max_cost='14.00',
               bbox_pool=BBOX_POOL, review_pool=REVIEW_POOL, max_photos=20, additional_bbox_pools=()):
    if len(candidates) != expected_count or expected_count < 1 or expected_count > max_photos:
        raise ValueError('Expected the complete selected batch within its explicit photo limit')
    if any(c['source_pool_id'] not in {bbox_pool, *additional_bbox_pools} for c in candidates):
        raise ValueError('Candidate comes from a different bbox pool')
    if len({c['photo_id'] for c in candidates}) != len(candidates):
        raise ValueError('Multiple revisions of the same photo require manual selection')
    controls = validate_controls(controls, source)
    already, existing_pages = {}, {}
    for suite in existing_suites:
        if suite.get('pool_id') != review_pool:
            raise ValueError('Unexpected existing review pool')
        for task in suite.get('tasks', []):
            if not task.get('known_solutions'):
                values = task['input_values']
                photo = str(values['photo_id'])
                if photo in already:
                    raise ValueError('Duplicate existing review photo')
                already[photo] = values
                existing_pages[photo] = suite
    previous = prior_reviewers(previous_reviews)
    skipped, pending = [], []
    for candidate in sorted(candidates, key=lambda c: (c['author_id'], c['photo_id'])):
        photo = candidate['photo_id']
        if photo in already:
            if already[photo] != input_values(candidate):
                raise ValueError('Previously loaded photo has a different candidate; no automatic re-review')
            page = existing_pages[photo]
            excluded = {candidate['author_id'], *participants.get(photo, [])}
            if (not excluded.issubset(page.get('unavailable_for', [])) or page.get('overlap') != 2
                    or page.get('infinite_overlap') is not False or page.get('reserved_for')):
                raise ValueError('Existing page has unsafe exclusions or overlap')
            skipped.append(candidate['candidate_id'])
        else:
            pending.append(candidate)
    suites = []
    for index in range(0, len(pending), 4):
        group = pending[index:index + 4]
        control = controls[(index // 4) % len(controls)]
        excluded = set(control.get('author_worker_ids', []))
        for candidate in [*group, control]:
            photo = candidate['photo_id']
            excluded.update(participants.get(photo, []))
            excluded.update(previous.get(photo, []))
            if 'author_id' in candidate:
                excluded.add(candidate['author_id'])
        for worker in excluded:
            required_text(worker)
        tasks = [{'input_values': input_values(c)} for c in group]
        gold = {'input_values': input_values(control), 'known_solutions': [
            {'output_values': {'verdict': control['verdict']}, 'correctness_weight': 1.0}]}
        tasks.insert((index // 4) % (len(tasks) + 1), gold)
        suites.append({'pool_id': review_pool, 'tasks': tasks, 'overlap': 2,
                       'infinite_overlap': False, 'unavailable_for': sorted(excluded), 'reserved_for': []})
    fee = Decimal(commission)
    if fee < 0 or fee > 1:
        raise ValueError('Invalid commission')
    cost = (Decimal(len(suites) * 2) * (1 + fee)).quantize(Decimal('0.01'))
    if cost > Decimal(max_cost):
        raise ValueError('Review transfer exceeds the approved planning ceiling')
    payload_hash = fingerprint(suites)
    return {'version': 1, 'mode': 'PREPARED_ONLY', 'bbox_pool': bbox_pool, 'review_pool': review_pool,
            'candidates': deepcopy(candidates), 'suites': suites,
            'review_suites': [*deepcopy(existing_suites), *suites], 'skipped_candidate_ids': skipped,
            'payload_hash': payload_hash,
            'operation_id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'review:' + review_pool + ':' + payload_hash)),
            'post_query': {'async_mode': True, 'operation_id': str(uuid.uuid5(uuid.NAMESPACE_URL,
                           'review:' + review_pool + ':' + payload_hash)), 'open_pool': False,
                           'allow_defaults': False, 'skip_invalid_items': False},
            'budget': {'new_suites': len(suites), 'paid_pages_max': len(suites) * 2,
                       'price_per_page_rub': 1, 'assumed_commission': str(fee),
                       'maximum_rub_under_assumption': str(cost), 'commission_confirmed': False},
            'upload_enabled': False}


def audit_uploaded(plan, uploaded_suites):
    expected = {fingerprint(s['tasks']): s for s in plan['suites']}
    if len(uploaded_suites) != len(expected):
        raise ValueError('Missing or duplicated uploaded suites')
    seen = set()
    for suite in uploaded_suites:
        tasks = [{k: t[k] for k in ('input_values', 'known_solutions') if k in t} for t in suite['tasks']]
        key = fingerprint(tasks)
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicated uploaded content')
        seen.add(key)
        wanted = expected[key]
        if (suite.get('pool_id') != plan['review_pool'] or suite.get('overlap') != 2
                or suite.get('infinite_overlap') is not False or suite.get('reserved_for')
                or not set(wanted['unavailable_for']).issubset(suite.get('unavailable_for', []))):
            raise ValueError('Uploaded page does not preserve author exclusions or overlap')
    return True


def audit_reviews(plan, assignments):
    candidates = {c['candidate_id']: c for c in plan['candidates']}
    votes, issues, seen_assignments = defaultdict(list), [], {}
    planned = {}
    controls = set()
    for suite in plan['review_suites']:
        for task in suite['tasks']:
            if not task.get('known_solutions'):
                planned[task['input_values']['candidate_id']] = suite
            else:
                controls.add(fingerprint(task['input_values']))
    for assignment in assignments:
        aid = required_text(assignment.get('id'))
        digest = fingerprint(assignment)
        if aid in seen_assignments:
            if seen_assignments[aid] != digest:
                issues.append({'assignment_id': aid, 'reason': 'conflicting_snapshots'})
            continue
        seen_assignments[aid] = digest
        worker = required_text(assignment.get('user_id'))
        if assignment.get('pool_id') != plan['review_pool']:
            raise ValueError('Unexpected review pool')
        tasks, solutions = assignment.get('tasks', []), assignment.get('solutions', [])
        if assignment.get('status') == 'ACCEPTED' and len(tasks) != len(solutions):
            issues.append({'assignment_id': aid, 'reason': 'incomplete_page'})
            continue
        seen_candidates = set()
        for index, task in enumerate(tasks):
            values = task.get('input_values', {})
            cid = values.get('candidate_id')
            if fingerprint(values) in controls:
                continue
            if (cid not in planned or cid in seen_candidates or
                    cid not in candidates or values != input_values(candidates[cid])):
                issues.append({'assignment_id': aid, 'reason': 'unknown_duplicate_or_changed_candidate'})
                continue
            seen_candidates.add(cid)
            if worker in planned[cid]['unavailable_for']:
                issues.append({'assignment_id': aid, 'candidate_id': cid, 'reason': 'excluded_worker_received_page'})
                continue
            if assignment.get('status') == 'ACCEPTED':
                verdict = solutions[index].get('output_values', {}).get('verdict')
                if verdict not in ('accept', 'fix', 'load_error'):
                    issues.append({'assignment_id': aid, 'candidate_id': cid, 'reason': 'invalid_verdict'})
                    continue
                votes[cid].append({'worker_id': worker, 'verdict': verdict, 'assignment_id': aid})
    decisions = []
    for cid, candidate in candidates.items():
        rows = votes[cid]
        distinct = len({r['worker_id'] for r in rows})
        if len(rows) > 2 or distinct != len(rows):
            decision = 'manual_review'
            issues.append({'candidate_id': cid, 'reason': 'duplicate_worker_or_excess_votes'})
        elif len(rows) < 2:
            decision = 'pending'
        else:
            decision = 'accepted_by_review' if all(r['verdict'] == 'accept' for r in rows) else 'manual_review'
        decisions.append({'candidate_id': cid, 'photo_id': candidate['photo_id'], 'decision': decision, 'votes': rows})
    return {'passed': not issues, 'issues': issues, 'decisions': decisions,
            'review_consensus_complete': not issues and all(d['decision'] == 'accepted_by_review' for d in decisions),
            'quality_gate_checked': False}
