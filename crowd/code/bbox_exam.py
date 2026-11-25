import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP

IOU_THRESHOLD = 0.7

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def review_content_hash(record):
    excluded = {'verified', 'verified_by', 'verification', 'candidate_id'}
    return fingerprint({k: v for k, v in record.items() if k not in excluded})

def validate_verification(record):
    if (record.get('verified') is not True or not record.get('annotator')
            or not record.get('verified_by') or record['annotator'] == record['verified_by']):
        raise ValueError('A documented reviewer different from the original annotator is required')
    review = record.get('verification', {})
    if review.get('type') == 'model_review' or str(record['verified_by']).startswith('captain_review'):
        if (review.get('type') != 'model_review' or review.get('user_authorized') is not True
                or review.get('independent_human_verification') is not False
                or review.get('reviewer') != record['verified_by']
                or not review.get('evidence_file') or not review.get('reason')
                or review.get('content_hash') != review_content_hash(record)):
            raise ValueError('Model review requires explicit authorization, provenance and unchanged content')

def normalize_boxes(boxes, coordinate_system='normalized_xywh', width=None, height=None):
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError('Expected a list of at most 100 rectangles')
    if coordinate_system not in ('normalized_xywh', 'pixel_xywh'):
        raise ValueError('Unknown coordinate system')
    if coordinate_system == 'pixel_xywh' and (not width or not height or width <= 0 or height <= 0):
        raise ValueError('Image dimensions required for pixel coordinates')
    result = []
    for box in boxes:
        if not isinstance(box, dict) or box.get('shape', 'rectangle') != 'rectangle':
            raise ValueError('Only rectangles are supported')
        values = [box.get(key) for key in ('left', 'top', 'width', 'height')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('Rectangle coordinates must be finite numbers')
        x, y, w, h = values
        if coordinate_system == 'pixel_xywh':
            x, y, w, h = x / width, y / height, w / width, h / height
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
            raise ValueError('Invalid rectangle bounds')
        result.append({'left': x, 'top': y, 'width': w, 'height': h})
    return result

def iou(a, b):
    width = max(0, min(a['left'] + a['width'], b['left'] + b['width']) - max(a['left'], b['left']))
    height = max(0, min(a['top'] + a['height'], b['top'] + b['height']) - max(a['top'], b['top']))
    intersection = width * height
    return intersection / (a['width'] * a['height'] + b['width'] * b['height'] - intersection)

def match_boxes(predicted, reference, threshold=IOU_THRESHOLD):
    matrix = [[iou(pred, gold) for gold in reference] for pred in predicted]
    edges = [sorted((j for j, value in enumerate(row) if value + 1e-12 >= threshold), key=lambda j: -row[j]) for row in matrix]
    matched = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False
    for i in range(len(predicted)):
        augment(i, set())
    return [{'predicted': i, 'gold': j, 'iou': matrix[i][j]} for j, i in sorted(matched.items())], matrix

def validate_gold(manifest):
    if manifest.get('version') != 1 or manifest.get('pool_id') != '6866117':
        raise ValueError('Wrong gold manifest version or exam pool')
    photos = manifest.get('photos', [])
    if len(photos) != 6 or len({str(p.get('photo_id')) for p in photos}) != 6:
        raise ValueError('Exam requires exactly six distinct photos on one page')
    normalized = {}
    for photo in photos:
        if photo.get('split') != 'exam':
            raise ValueError('Use the reserved exam split')
        validate_verification(photo)
        if photo.get('verified') is not True or not photo.get('image_url'):
            raise ValueError('Gold not verified or image URL missing')
        boxes = normalize_boxes(photo.get('boxes'), photo.get('coordinate_system'), photo.get('image_width'), photo.get('image_height'))
        if len({tuple(b.values()) for b in boxes}) != len(boxes):
            raise ValueError('Duplicate gold rectangles')
        status = 'has_butterflies' if boxes else 'no_butterflies'
        if photo.get('status') != status:
            raise ValueError('Inconsistent gold status and boxes')
        normalized[str(photo['photo_id'])] = {**photo, 'boxes': boxes}
    if sum(bool(p['boxes']) for p in normalized.values()) != 4:
        raise ValueError('Exam requires four positive and two negative photos')
    return normalized

def grade_photo(output, gold):
    try:
        if not isinstance(output, dict):
            raise ValueError('Missing output values')
        status = output.get('status')
        if status == 'load_error':
            return {'outcome': 'manual_review', 'reason': 'image_load_error'}
        if status not in ('has_butterflies', 'no_butterflies'):
            raise ValueError('Missing or unknown status')
        boxes = normalize_boxes(output.get('result', []))
        if (status == 'has_butterflies') != bool(boxes):
            raise ValueError('Status conflicts with rectangles')
    except ValueError as error:
        return {'outcome':'manual_review','reason':str(error)}
    if status != gold['status']:
        return {'outcome':'fail','reason':'wrong_object_presence'}
    pairs, matrix = match_boxes(boxes, gold['boxes'])
    complete = len(pairs) == len(boxes) == len(gold['boxes'])
    result = {'outcome':'pass' if complete else 'fail', 'reason':'all_objects_matched' if complete else 'missing_extra_or_inaccurate_boxes',
              'pairs':pairs,'predicted_count':len(boxes),'gold_count':len(gold['boxes'])}
    if len(boxes) == len(gold['boxes']) and boxes:
        loose, _ = match_boxes(boxes, gold['boxes'], threshold=0.65)
        strict, _ = match_boxes(boxes, gold['boxes'], threshold=0.75)
        if len(loose) == len(boxes) and len(strict) != len(boxes):
            result.update(outcome='manual_review',reason='borderline_geometry',automatic_threshold_result=complete)
    return result

def grade_assignment(assignment, manifest):
    gold = validate_gold(manifest)
    result = {'assignment_id':assignment.get('id'), 'user_id':assignment.get('user_id'),
              'gold_hash':fingerprint(manifest), 'assignment_hash':fingerprint(assignment), 'decision':'manual_review'}
    if not assignment.get('id') or not assignment.get('user_id'):
        return {**result,'reason':'missing_assignment_or_worker_id'}
    if assignment.get('pool_id') != manifest['pool_id'] or assignment.get('status') != 'ACCEPTED':
        return {**result,'reason':'wrong_pool_or_not_accepted'}
    tasks, solutions = assignment.get('tasks', []), assignment.get('solutions', [])
    if len(tasks) != 6 or len(solutions) != 6:
        return {**result,'reason':'incomplete_exam_page'}
    seen, grades = set(), {}
    for task, solution in zip(tasks, solutions):
        inputs = task.get('input_values', {})
        key = str(inputs.get('photo_id'))
        if key in seen or key not in gold or inputs.get('image') != gold[key]['image_url']:
            return {**result,'reason':'unexpected_duplicate_or_changed_photo'}
        seen.add(key)
        grades[key] = grade_photo(solution.get('output_values'), gold[key])
    result['photos'] = grades
    if any(g['outcome'] == 'manual_review' for g in grades.values()):
        return {**result,'reason':'technical_or_borderline_answer'}
    correct = sum(g['outcome'] == 'pass' for g in grades.values())
    negatives_ok = all(grades[key]['outcome'] == 'pass' for key,p in gold.items() if not p['boxes'])
    passed = correct >= 5 and negatives_ok
    score = (Decimal(correct) * 100 / 6).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return {**result,'decision':'pass' if passed else 'fail', 'correct':correct, 'total':6,
            'negatives_ok':negatives_ok, 'score':float(score), 'gate':100 if passed else 0}
