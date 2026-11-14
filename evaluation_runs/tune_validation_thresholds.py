"""Resumable VALIDATION-only threshold sweep with an inference-equivalent disk cache."""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import copy
import csv
import gc
import io
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'detection'))
import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO
from evaluate_cascade import (compile_label_policy, detector_categories, detector_index_map,
    read_json, sha256_file, validate_coco)
from waste_cascade import WasteCascade, synchronize
from threshold_cache import DiskCache, ensure_image, build_predictions, predictions_for_image
from grid_metrics import metric_sweep, best_pr


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def as_coco(metadata):
    with contextlib.redirect_stdout(io.StringIO()):
        result = COCO()
        result.dataset = copy.deepcopy(metadata)
        result.createIndex()
        return result


def parse_grid(value):
    result = [float(x.strip()) for x in value.split(',') if x.strip()]
    if not result or len(set(result)) != len(result) or any(not 0 <= x <= 1 for x in result):
        raise ValueError('Grid must contain unique values in [0,1]')
    return result


def smoke_equivalence(cache, model, images, root, index_map, nms_grid, args):
    checks, latency = [], []
    for image in images:
        with Image.open(root/image['file_name']) as pil:
            rgb = pil.convert('RGB')
        for iou in [None]+nms_grid:
            synchronize(model.device)
            start = time.perf_counter()
            actual = model.predict(rgb, proposal_threshold=args.proposal_threshold,
                max_proposals=args.max_proposals, cls_batch_size=args.cls_batch_size, nms_iou=iou)
            synchronize(model.device)
            latency.append({'image_id':image['id'], 'nms_iou':iou, 'cascade_predict_wall_ms':(time.perf_counter()-start)*1000})
            for classify in [False, True]:
                cached = predictions_for_image(cache, image['id'], model.cls_names, model.det_name_to_id, index_map,
                    classify=classify, nms_iou=iou, proposal_threshold=args.proposal_threshold,max_proposals=args.max_proposals)
                direct = actual['cascade' if classify else 'baseline']
                if len(cached) != len(direct):
                    raise AssertionError('Cached/direct prediction count differs')
                max_delta = 0.
                for a,b in zip(cached,direct):
                    external = None if b['category_id'] is None else index_map[b['category_id']]
                    if a['bbox_xyxy'] != b['bbox_xyxy'] or a['category_id'] != external:
                        raise AssertionError('Cached/direct coordinates or mapped labels differ')
                    if classify and a['class_name'] != b['class_name']:
                        raise AssertionError('Cached/direct classifier labels differ')
                    for score_key in (['score','cls_score','det_score'] if classify else ['score']):
                        delta=abs(a[score_key]-b[score_key]); max_delta=max(max_delta,delta)
                        if not np.isclose(a[score_key],b[score_key],rtol=1e-5,atol=1e-6):
                            raise AssertionError(f'Cached/direct score differs: {score_key} {a[score_key]} {b[score_key]}')
                checks.append({'image_id':image['id'],'nms_iou':iou,'classify':classify,
                               'predictions':len(cached),'max_abs_score_difference':max_delta,'passed':True})
    return {'checks':checks,'all_passed':True,'direct_pipeline_latency_observations':latency,
            'latency_limitations':'Smoke observations only; tiny sample, no controlled benchmark. Not production latency.'}


def grid_search(cache, model, metadata, image_ids, index_map, nms_grid, pr_grid, args, output):
    gt=as_coco(metadata)
    entries, rows = [], []
    for iou in [None]+nms_grid:
        for classify in [False,True]:
            name=('cascade' if classify else 'detector')+('_original' if iou is None else '_nms')
            label=name if iou is None else name+f'_iou_{iou:.2f}'
            print(f'COCOeval {label}: one matching pass for all {len(pr_grid)} PR thresholds',flush=True)
            exported,outside=build_predictions(cache,image_ids,model.cls_names,model.det_name_to_id,index_map,
                classify=classify,nms_iou=iou,proposal_threshold=args.proposal_threshold,max_proposals=args.max_proposals)
            metrics,sweep=metric_sweep(gt,exported,image_ids,pr_grid)
            for point in sweep:
                point['outside_scope_dropped_at_pr_threshold']=sum(p['score'] >= point['pr_score_threshold'] for p in outside)
                rows.append({'experiment':name,'nms_iou':iou,**metrics,**point,'outside_scope_dropped':len(outside)})
            entry={'experiment':name,'nms_iou':iou,**metrics,'exported_prediction_count':len(exported),
                   'outside_scope_dropped':len(outside),'outside_scope_by_class':dict(Counter(p['class_name'] for p in outside)),
                   'best_pr_operating_point':best_pr(sweep),'pr_threshold_sweep':sweep,'latency_ms':None}
            entries.append(entry)
            folder=output/'predictions'/label
            save_json(folder/'predictions.json',exported)
            save_json(folder/'out_of_scope_predictions.json',outside)
            save_json(folder/'metrics.json',entry)
            save_json(output/'grid_progress.json',{'completed_settings':len(entries),'settings':entries})
            del exported,outside
            gc.collect()
    no_nms={e['experiment']:e for e in entries if e['nms_iou'] is None}
    sweep={f'{iou:.2f}':{e['experiment']:e for e in entries if e['nms_iou']==iou} for iou in nms_grid}
    chosen=max(nms_grid,key=lambda iou:(
        sum(sweep[f'{iou:.2f}'][name]['mAP50_95'] or 0. for name in ['detector_nms','cascade_nms']),
        sweep[f'{iou:.2f}']['cascade_nms']['mAP50_95'] or 0.,-iou))
    shared=sweep[f'{chosen:.2f}']
    recommended={'proposal_threshold':args.proposal_threshold,'nms_iou':chosen,
        'baseline_pr_threshold':shared['detector_nms']['best_pr_operating_point']['pr_score_threshold'],
        'cascade_pr_threshold':shared['cascade_nms']['best_pr_operating_point']['pr_score_threshold'],
        'selection_rule':'Predeclared: shared NMS maximizes mean AP50:95 of detector_nms/cascade_nms; tie cascade AP then lower IoU. PR maximizes micro-F1@IoU0.5, tie precision then lower threshold.',
        'independent':False,'eligible_for_final_test_lock':args.mode!='smoke' and not args.allow_provisional_validation,
        'selection_status':'smoke_only' if args.mode=='smoke' else ('provisional_pending_data_review' if args.allow_provisional_validation else 'validation_selected')}
    family_best={name:max([e for e in entries if e['experiment']==name],key=lambda e:((e['mAP50_95'] or 0.),-(e['nms_iou'] or 0.)))
                 for name in ['detector_original','detector_nms','cascade_original','cascade_nms']}
    with (output/'grid_results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for name in ['detector_original','detector_nms','cascade_original','cascade_nms']:
        entry=no_nms[name] if name.endswith('original') else shared[name]
        summary.append({'experiment':name,'nms_iou':entry['nms_iou'],**{k:entry[k] for k in ['mAP50','mAP50_95','AR100','outside_scope_dropped']},**entry['best_pr_operating_point']})
    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return {'no_nms':no_nms,'nms_sweep':sweep,'recommended_locked_parameters':recommended,
            'best_nms_per_family_by_AP50_95':family_best,'summary_at_shared_nms':summary,
            'cocoeval_matching_passes':len(entries),'grid_row_count':len(rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','grid_search','single_run'],default='grid_search')
    parser.add_argument('--det_ckpt',default=str(ROOT/'detection/checkpoints_wastedetect/best_model.pt'))
    parser.add_argument('--cls_ckpt',default=str(ROOT/'checkpoints/dinov2_garbage_final.pth'))
    parser.add_argument('--valid_coco',default=str(ROOT/'WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json'))
    parser.add_argument('--valid_images',default=str(ROOT/'WasteDetect_Dataset/valid_clean_candidate/images'))
    parser.add_argument('--class_mapping',default=str(ROOT/'detection/class_mapping.eval.json'))
    parser.add_argument('--detector_class_map')
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--proposal_threshold',type=float,default=0.)
    parser.add_argument('--nms_grid',default='0.3,0.4,0.5,0.6,0.7')
    parser.add_argument('--pr_threshold_grid',default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50')
    parser.add_argument('--nms_iou',type=float,default=.5)
    parser.add_argument('--baseline_pr_threshold',type=float,default=.05)
    parser.add_argument('--cascade_pr_threshold',type=float,default=.05)
    parser.add_argument('--max_proposals',type=int,default=100)
    parser.add_argument('--cls_batch_size',type=int,default=2)
    parser.add_argument('--smoke_images',type=int,default=3)
    parser.add_argument('--warmup_images',type=int,default=0)
    parser.add_argument('--cpu_threads',type=int,default=2)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--allow_provisional_validation',action='store_true',help='Explicitly permit unresolved data warnings; no final Test lock is claimed')
    args=parser.parse_args()
    if args.proposal_threshold != 0. or args.cls_batch_size < 1 or args.max_proposals < 1 or args.smoke_images < 1 or args.warmup_images < 0:
        parser.error('COCO proposal threshold must be 0; batch/cap/smoke positive; warmup nonnegative')
    if any(p.lower().startswith('test') for p in Path(args.valid_coco).parts+Path(args.valid_images).parts):
        parser.error('This script accepts VALIDATION only; Test paths are forbidden')
    nms_grid,pr_grid=parse_grid(args.nms_grid),parse_grid(args.pr_threshold_grid)
    if args.mode=='single_run':
        nms_grid=[args.nms_iou];pr_grid=sorted(set([args.baseline_pr_threshold,args.cascade_pr_threshold]))
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    output=Path(args.out_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    metadata=read_json(args.valid_coco)
    categories=detector_categories(metadata);validate_coco(metadata,categories)
    full_count,full_boxes=len(metadata['images']),len(metadata['annotations'])
    audit_path=output/'data_audit/validation_data_audit.json'
    data_audit=read_json(audit_path) if audit_path.exists() else None
    if args.mode!='smoke':
        if data_audit is None or data_audit['validation_coco_sha256'] != sha256_file(args.valid_coco):
            raise ValueError('A current validation data audit is required before the full sweep')
        if any(p['cls_split'] in {'cls_train','cls_val'} for p in data_audit['exact_pairs']):
            raise ValueError('Confirmed exact classification train/val overlap: full sweep is blocked')
        warnings=bool(data_audit['phash_train_val_pending'] or data_audit['duplicate_annotation_geometry'])
        if warnings and not args.allow_provisional_validation:
            raise ValueError('Unreviewed classification pHash candidates / duplicate GT remain. Full run requires curator resolution or explicit provisional authorization.')
    index_map=detector_index_map(categories,read_json(args.detector_class_map) if args.detector_class_map else None)
    cls_meta=torch.load(args.cls_ckpt,map_location='cpu',weights_only=True)
    class_names=list(cls_meta['classes']);del cls_meta
    resolved=compile_label_policy(read_json(args.class_mapping),class_names,categories)
    reverse={v:k for k,v in index_map.items()}
    internal={k:None if v is None else reverse[v] for k,v in resolved.items()}
    checkpoints={name:sha256_file(getattr(args,name)) for name in ['det_ckpt','cls_ckpt']}
    model=WasteCascade(args.det_ckpt,args.cls_ckpt,{i:categories[cid] for i,cid in index_map.items()},args.device,classifier_to_detector=internal)
    fingerprint={'version':2,'validation_coco_sha256':sha256_file(args.valid_coco),'checkpoints':checkpoints,
        'class_mapping_sha256':sha256_file(args.class_mapping),'classes':class_names,'index_map':index_map,
        'classifier_tf':repr(model.cls_tf),'detector_tf':repr(model.det_tf),'device':str(model.device),
        'device_name':torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else 'cpu',
        'torch':torch.__version__,'cpu_threads':args.cpu_threads,
        'code':{name:sha256_file(ROOT/name) for name in ['detection/waste_cascade.py','detection/Train.py','evaluation_runs/threshold_cache.py']}}
    cache=DiskCache(output/'cache/predictions.sqlite3',fingerprint)
    save_json(output/'cache/fingerprint.json',fingerprint)
    images=metadata['images'][:args.smoke_images] if args.mode=='smoke' else metadata['images']
    if args.mode=='smoke':
        ids={im['id'] for im in images};metadata=copy.deepcopy(metadata)
        metadata['images']=images;metadata['annotations']=[ann for ann in metadata['annotations'] if ann['image_id'] in ids]
    audit_hashes={im['image_id']:im['sha256'] for im in data_audit['validation_image_manifest']} if data_audit else {}
    counters=Counter();start=time.perf_counter()
    if model.device.type=='cuda':torch.cuda.reset_peak_memory_stats(model.device)
    try:
        for image in images[:args.warmup_images]:
            with Image.open(Path(args.valid_images)/image['file_name']) as pil:
                model.predict(pil,proposal_threshold=args.proposal_threshold,max_proposals=args.max_proposals,
                              cls_batch_size=args.cls_batch_size,nms_iou=None)
        for index,image in enumerate(images,1):
            stats=ensure_image(cache,model,image,Path(args.valid_images)/image['file_name'],nms_grid,
                proposal_threshold=args.proposal_threshold,max_proposals=args.max_proposals,
                cls_batch_size=args.cls_batch_size,expected_sha=audit_hashes.get(image['id']))
            counters.update(stats)
            if index%10==0 or index==len(images):
                progress={'completed_images':index,'total_images':len(images),'elapsed_seconds':time.perf_counter()-start,'cache_work_this_run':dict(counters)}
                print(json.dumps(progress),flush=True);save_json(output/'inference_progress.json',progress)
        if args.mode=='smoke':
            smoke=smoke_equivalence(cache,model,images,Path(args.valid_images),index_map,nms_grid,args)
            save_json(output/'smoke/equivalence.json',smoke)
        target=output/'smoke' if args.mode=='smoke' else output
        results=grid_search(cache,model,metadata,[im['id'] for im in images],index_map,nms_grid,pr_grid,args,target)
        report={'split':'valid_clean_candidate','mode':args.mode,'validation_images_used':len(images),
            'validation_boxes_used':len(metadata['annotations']),'full_validation_images':full_count,'full_validation_boxes':full_boxes,
            'settings':vars(args),'cache_work_this_run':dict(counters),'cache_fingerprint':fingerprint,
            'independent':False,'data_warnings':{'classification_phash_pending':data_audit.get('phash_train_val_pending') if data_audit else 'audit_pending',
                'duplicate_annotation_geometry':data_audit.get('duplicate_annotation_geometry') if data_audit else 'audit_pending'},
            'protocol':{'AP':'All exported predictions at proposal_threshold=0, independent of PR thresholds.',
                'PR':'COCOeval raw matches; micro P/R/F1 at IoU .5; crowd/maxDets=100 per category respected.',
                'score':'Same float32 det_score * cls_score as WasteCascade.predict.',
                'latency':'Per-setting latency unavailable: shared cached inference is not an end-to-end benchmark. Smoke timings are tiny-sample observations only.',
                'cache':'Raw detector scores/boxes and crop probabilities persisted transactionally on CPU in SQLite. Selection uses the actual inference device. Resume never recomputes completed cache items; explicit warmup/smoke equivalence calls are separate.',
                'test':'No Test images used for inference, matching or parameter selection.'},
            'hardware':{'gpu':fingerprint['device_name'],'peak_allocated_MB':torch.cuda.max_memory_allocated(model.device)/1024**2 if model.device.type=='cuda' else None,
                        'peak_reserved_MB':torch.cuda.max_memory_reserved(model.device)/1024**2 if model.device.type=='cuda' else None},
            **results}
        report['checkpoints_unchanged']=all(sha256_file(getattr(args,name))==digest for name,digest in checkpoints.items())
        if not report['checkpoints_unchanged']:raise RuntimeError('Checkpoint bytes changed during run')
        save_json(target/'validation_grid_search_report.json',report)
        parameter_file='smoke_parameters_not_for_test.json' if args.mode=='smoke' else 'locked_parameters.json'
        save_json(target/parameter_file,report['recommended_locked_parameters'])
        print(json.dumps({'mode':args.mode,'images':len(images),'boxes':len(metadata['annotations']),
              'recommended':report['recommended_locked_parameters'],'summary':report['summary_at_shared_nms'],
              'hardware':report['hardware'],'checkpoints_unchanged':report['checkpoints_unchanged']},indent=2,ensure_ascii=False),flush=True)
    except Exception as error:
        save_json(output/('smoke_failure.json' if args.mode=='smoke' else 'run_failure.json'),{'error':type(error).__name__,'message':str(error),'cache_preserved':True})
        raise
    finally:
        cache.close()


if __name__=='__main__':main()
