import copy
import itertools
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from bbox_exam import iou, normalize_boxes, match_boxes, grade_assignment, validate_gold, review_content_hash
from skill_writer import issue_skills

BOX = {'shape':'rectangle','left':0.1,'top':0.2,'width':0.3,'height':0.4}

def fixture():
    photos = [{'photo_id':str(i),'split':'exam','image_url':'https://example.invalid/'+str(i),
               'coordinate_system':'normalized_xywh','image_width':1000,'image_height':1000,
               'boxes':[copy.deepcopy(BOX)] if i < 4 else [],
               'status':'has_butterflies' if i < 4 else 'no_butterflies',
               'annotator':'author','verified_by':'reviewer','verified':True} for i in range(6)]
    gold = {'version':1,'pool_id':'6866117','photos':photos}
    assignment = {'id':'synthetic-assignment','user_id':'synthetic-worker','pool_id':'6866117','status':'ACCEPTED',
                  'tasks':[{'input_values':{'photo_id':p['photo_id'],'image':p['image_url']}} for p in photos],
                  'solutions':[{'output_values':{'status':p['status'],'result':copy.deepcopy(p['boxes'])}} for p in photos]}
    return gold, assignment

class GeometryTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou(BOX, BOX),1)
        self.assertEqual(iou(BOX,dict(BOX,left=.7,width=.2)),0)

    def test_pixel_conversion(self):
        result=normalize_boxes([{'left':100,'top':200,'width':300,'height':400}],'pixel_xywh',1000,1000)
        self.assertEqual(result[0],{k:BOX[k] for k in ('left','top','width','height')})

    def test_invalid_coordinates(self):
        for value in (math.nan,math.inf,-.1,True,'0.1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalize_boxes([dict(BOX,left=value)])
        with self.assertRaises(ValueError): normalize_boxes([dict(BOX,width=0)])
        with self.assertRaises(ValueError): normalize_boxes([dict(BOX,left=.9)])

    def test_matching_against_exhaustive_oracle(self):
        rng=random.Random(113)
        for _ in range(80):
            a=[{'left':rng.random()*.4,'top':rng.random()*.4,'width':.4,'height':.4} for _ in range(4)]
            b=[{'left':rng.random()*.4,'top':rng.random()*.4,'width':.4,'height':.4} for _ in range(4)]
            pairs,matrix=match_boxes(a,b,.3)
            best=max(sum(matrix[i][j]>=.3 for i,j in enumerate(perm)) for perm in itertools.permutations(range(4)))
            self.assertEqual(len(pairs),best)
            self.assertEqual(len({p['gold'] for p in pairs}),len(pairs))

class ExamTests(unittest.TestCase):
    def setUp(self): self.gold,self.assignment=fixture()
    def grade(self): return grade_assignment(self.assignment,self.gold)
    def test_perfect(self):
        result=self.grade()
        self.assertEqual((result['decision'],result['score'],result['gate']),('pass',100,100))
    def test_five_of_six(self):
        self.assignment['solutions'][0]['output_values']={'status':'no_butterflies','result':[]}
        self.assertEqual((self.grade()['decision'],self.grade()['score']),('pass',83.33))
    def test_negative_error_blocks_despite_five_correct(self):
        self.assignment['solutions'][4]['output_values']={'status':'has_butterflies','result':[BOX]}
        self.assertEqual((self.grade()['correct'],self.grade()['gate']),(5,0))
    def test_duplicate_box_fails_photo(self):
        self.assignment['solutions'][0]['output_values']['result']=[BOX,BOX]
        self.assertEqual(self.grade()['photos']['0']['outcome'],'fail')
    def test_box_order_irrelevant(self):
        second=dict(BOX,left=.6,width=.2)
        self.gold['photos'][0]['boxes']=[BOX,second]
        self.assignment['solutions'][0]['output_values']['result']=[second,BOX]
        self.assertEqual(self.grade()['decision'],'pass')
    def test_technical_and_borderline_held(self):
        self.assignment['solutions'][0]['output_values']={'status':'load_error'}
        self.assertEqual(self.grade()['decision'],'manual_review')
        self.assignment['solutions'][0]['output_values']={'status':'has_butterflies','result':[dict(BOX,width=.3/.7)]}
        self.assertEqual(self.grade()['decision'],'manual_review')
    def test_incomplete_and_duplicate_page(self):
        self.assignment['solutions'].pop()
        self.assertEqual(self.grade()['decision'],'manual_review')
        self.gold,self.assignment=fixture()
        self.assignment['tasks'][1]=self.assignment['tasks'][0]
        self.assertEqual(self.grade()['decision'],'manual_review')
    def test_wrong_image_or_pending_submission(self):
        self.assignment['tasks'][0]['input_values']['image']='https://example.invalid/other'
        self.assertEqual(self.grade()['decision'],'manual_review')
        self.gold,self.assignment=fixture()
        self.assignment['status']='SUBMITTED'
        self.assertEqual(self.grade()['decision'],'manual_review')
    def test_gold_needs_second_person(self):
        self.gold['photos'][0]['verified_by']='author'
        with self.assertRaises(ValueError): validate_gold(self.gold)
    def model_review(self):
        row = self.gold['photos'][0]
        row['verified_by'] = 'captain_review'
        row['verification'] = {'type': 'model_review', 'reviewer': 'captain_review', 'user_authorized': True,
            'independent_human_verification': False, 'evidence_file': 'synthetic.json',
            'reason': 'Synthetic test', 'content_hash': review_content_hash(row)}
        return row
    def test_authorized_model_review(self):
        self.model_review()
        self.assertEqual(self.grade()['decision'], 'pass')
    def test_model_review_rejects_changed_coordinates(self):
        row = self.model_review()
        row['boxes'][0]['left'] = .12
        with self.assertRaises(ValueError): validate_gold(self.gold)
    def test_model_review_requires_authorization_and_honest_provenance(self):
        for key, value in [('user_authorized', False), ('independent_human_verification', True)]:
            row = self.model_review()
            row['verification'][key] = value
            with self.assertRaises(ValueError): validate_gold(self.gold)
        row = self.model_review()
        del row['verification']
        with self.assertRaises(ValueError): validate_gold(self.gold)
    def test_missing_empty_result_is_valid(self):
        self.assignment['solutions'][4]['output_values'].pop('result')
        self.assertEqual(self.grade()['decision'],'pass')

class FakeAPI:
    def __init__(self, timeout=None):
        self.values={}
        self.writes=[]
        self.timeout=timeout
    def __call__(self,method,path,payload=None):
        from urllib.parse import parse_qs
        if method=='GET':
            query=parse_qs(path.split('?')[1]);key=query['skill_id'][0]
            return {'items':[] if key not in self.values else [{'skill_id':key,'user_id':'synthetic-worker','exact_value':self.values[key]}], 'has_more':False}
        self.writes.append(payload)
        if set(payload) != {'skill_id','user_id','value'}:
            raise ValueError('PUT schema requires value; exact_value is a response field')
        if self.timeout!='before': self.values[payload['skill_id']]=payload['value']
        if self.timeout: raise TimeoutError('synthetic')
        return {'ok':True}

class SkillTests(unittest.TestCase):
    def setUp(self):
        gold,assignment=fixture();self.report=grade_assignment(assignment,gold)
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'journal.json'
    def tearDown(self): self.temp.cleanup()
    def test_issue_and_repeat_are_idempotent(self):
        api=FakeAPI()
        self.assertEqual(issue_skills(self.report,api,self.path)['status'],'verified')
        issue_skills(self.report,api,self.path)
        self.assertEqual([p['skill_id'] for p in api.writes],['21393','21394'])
    def test_timeout_after_server_write_resolved_by_get(self):
        api=FakeAPI('after')
        self.assertEqual(issue_skills(self.report,api,self.path)['status'],'verified')
    def test_unknown_write_not_repeated(self):
        api=FakeAPI('before')
        with self.assertRaises(RuntimeError): issue_skills(self.report,api,self.path)
        with self.assertRaises(RuntimeError): issue_skills(self.report,api,self.path)
        self.assertEqual(len(api.writes),1)
    def test_existing_conflict_prevents_all_writes(self):
        api=FakeAPI();api.values['21394']=0
        with self.assertRaises(RuntimeError): issue_skills(self.report,api,self.path)
        self.assertFalse(api.writes)
    def test_manual_review_never_writes(self):
        api=FakeAPI()
        issue_skills({**self.report,'decision':'manual_review'},api,self.path)
        self.assertFalse(api.writes)

if __name__=='__main__': unittest.main(verbosity=2)
