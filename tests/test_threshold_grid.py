"""Cache/direct inference parity and COCOeval sweep equivalence, no weights/network."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'detection'), str(ROOT/'evaluation_runs')]
from waste_cascade import WasteCascade
from evaluate_cascade import coco_metrics
from threshold_cache import DiskCache, ensure_image, predictions_for_image, cascade_score, build_predictions
from grid_metrics import metric_sweep, best_pr


def model_fixture(outside=False):
    model = WasteCascade.__new__(WasteCascade)
    model.device = torch.device('cpu')
    model.cls_names = ['paper','plastic']+[f'other{i}' for i in range(8)]
    model.det_categories = {0:'paper',1:'plastic'}
    model.det_name_to_id = {'paper':None if outside else 0,'plastic':1}
    model.det_tf = Mock(return_value=torch.zeros(3,14,14))
    model.cls_tf = Mock(return_value=torch.zeros(3,14,14))
    probabilities = torch.tensor([[[.9,.1],[.1,.8],[.7,.1]]])
    model.detector = Mock(return_value={'pred_logits':torch.logit(probabilities),
        'pred_boxes':torch.tensor([[[.25,.25,.3,.3],[.25,.25,.3,.3],[.75,.75,.3,.3]]])})
    model.classifier = Mock(side_effect=lambda x: torch.tensor([[3.,1.]+[-2.]*8]).repeat(len(x),1))
    return model


def coco_fixture(crowd=False, empty=False):
    data={'images':[{'id':1,'width':100,'height':100,'file_name':'sample.png'}],
          'categories':[{'id':0,'name':'paper'},{'id':1,'name':'plastic'}],
          'annotations':[] if empty else [{'id':1,'image_id':1,'category_id':0,'bbox':[10,10,30,30],'area':900,'iscrowd':0}],
          'info':{},'licenses':[]}
    if crowd:
        data['annotations'].append({'id':2,'image_id':1,'category_id':0,'bbox':[60,60,30,30],'area':900,'iscrowd':1})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=data;gt.createIndex()
    return gt


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'image.png'
        Image.new('RGB',(100,100),'white').save(self.path)
        self.image={'id':1,'file_name':'image.png','width':100,'height':100}

    def test_all_nms_parity_and_single_inference(self):
        model=model_fixture();cache=DiskCache(self.root/'cache.sqlite',{'version':1})
        self.addCleanup(cache.close)
        stats=ensure_image(cache,model,self.image,self.path,[.3,.4,.5,.6,.7],cls_batch_size=2)
        self.assertEqual(stats['detector_inferences'],1)
        self.assertEqual(stats['new_unique_crops'],2)  # Three queries, two identical coordinates.
        self.assertEqual(model.classifier.call_count,1)
        for iou in [None,.3,.4,.5,.6,.7]:
            actual=model.predict(Image.open(self.path),nms_iou=iou,cls_batch_size=2,proposal_threshold=0.)
            for classify in [False,True]:
                cached=predictions_for_image(cache,1,model.cls_names,model.det_name_to_id,{0:0,1:1},classify=classify,nms_iou=iou)
                direct=actual['cascade' if classify else 'baseline']
                self.assertEqual([r['bbox_xyxy'] for r in cached],[r['bbox_xyxy'] for r in direct])
                self.assertEqual([r['category_id'] for r in cached],[r['category_id'] for r in direct])
                np.testing.assert_allclose([r['score'] for r in cached],[r['score'] for r in direct],rtol=1e-6)

    def test_disk_resume_calls_neither_model(self):
        model=model_fixture();path=self.root/'cache.sqlite'
        cache=DiskCache(path,{'version':1})
        ensure_image(cache,model,self.image,self.path,[.5]);cache.close()
        model.detector.reset_mock();model.classifier.reset_mock()
        cache=DiskCache(path,{'version':1});self.addCleanup(cache.close)
        stats=ensure_image(cache,model,self.image,self.path,[.3,.5,.7])
        self.assertEqual(stats['detector_inferences'],0);self.assertEqual(stats['new_unique_crops'],0)
        model.detector.assert_not_called();model.classifier.assert_not_called()

    def test_fingerprint_and_image_change_rejected(self):
        model=model_fixture();path=self.root/'cache.sqlite'
        cache=DiskCache(path,{'version':1});ensure_image(cache,model,self.image,self.path,[.5]);cache.close()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            DiskCache(path,{'version':2})
        Image.new('RGB',(100,100),'black').save(self.path)
        cache=DiskCache(path,{'version':1});self.addCleanup(cache.close)
        with self.assertRaisesRegex(ValueError,'content changed'):
            ensure_image(cache,model,self.image,self.path,[.5])

    def test_nms_before_cap_union_includes_lower_query(self):
        cache=DiskCache(self.root/'cache.sqlite',{});self.addCleanup(cache.close)
        model=model_fixture();ensure_image(cache,model,self.image,self.path,[.5],max_proposals=2)
        rows=predictions_for_image(cache,1,model.cls_names,model.det_name_to_id,{0:10,1:20},
                                  classify=False,nms_iou=.5,max_proposals=2)
        self.assertEqual([r['bbox_xyxy'] for r in rows],[[10,10,40,40],[60,60,90,90]])
        self.assertEqual(rows[0]['category_id'],10)

    def test_empty_proposals_skip_classifier(self):
        model=model_fixture()
        model.detector.return_value['pred_boxes'].zero_()
        cache=DiskCache(self.root/'cache.sqlite',{});self.addCleanup(cache.close)
        ensure_image(cache,model,self.image,self.path,[.5])
        exported,outside=build_predictions(cache,[1],model.cls_names,model.det_name_to_id,{0:0,1:1},classify=True,nms_iou=.5)
        self.assertEqual(exported,[]);self.assertEqual(outside,[]);model.classifier.assert_not_called()

    def test_outside_scope_stays_counted(self):
        model=model_fixture(outside=True);cache=DiskCache(self.root/'cache.sqlite',{});self.addCleanup(cache.close)
        ensure_image(cache,model,self.image,self.path,[.5])
        exported,outside=build_predictions(cache,[1],model.cls_names,model.det_name_to_id,{0:0,1:1},classify=True,nms_iou=.5)
        self.assertEqual(exported,[]);self.assertEqual(len(outside),2)

    def test_score_matches_float32_tensor_product(self):
        for a,b in [(.234567891,.876543211),(.05,.99999994),(.33333334,.3)]:
            self.assertEqual(cascade_score(a,b),float(a*torch.tensor(b,dtype=torch.float32)))


class MetricTests(unittest.TestCase):
    def test_one_pass_sweep_equals_reference_matching_with_crowd_and_duplicates(self):
        predictions=[{'image_id':1,'category_id':0,'bbox':box,'score':score}
                     for box,score in [([10,10,30,30],.9),([10,10,30,30],.2),([65,65,10,10],.1)]]
        thresholds=[.05,.15,.3,.5]
        stats,sweep=metric_sweep(coco_fixture(crowd=True),predictions,[1],thresholds)
        for threshold,row in zip(thresholds,sweep):
            reference=coco_metrics(coco_fixture(crowd=True),predictions,[1],threshold)
            for key in ['mAP50','mAP50_95','AR100']:self.assertAlmostEqual(stats[key],reference[key])
            for key in ['precision_iou50','recall_iou50','TP','FP','FN','ignored_predictions']:
                self.assertEqual(row[key],reference[key])

    def test_empty_predictions_and_absent_gt(self):
        stats,sweep=metric_sweep(coco_fixture(),[],[1],[.05,.5])
        self.assertEqual(stats['mAP50'],0.);self.assertEqual(sweep[0]['FN'],1)
        stats,sweep=metric_sweep(coco_fixture(empty=True),[],[1],[.05])
        self.assertIsNone(stats['mAP50']);self.assertIsNone(sweep[0]['f1_iou50'])
        with self.assertRaises(ValueError):best_pr(sweep)

    def test_best_pr_ties_are_deterministic(self):
        points=[{'f1_iou50':.8,'precision_iou50':.9,'pr_score_threshold':thr} for thr in [.2,.1,.3]]
        self.assertEqual(best_pr(points)['pr_score_threshold'],.1)


if __name__=='__main__':unittest.main()
