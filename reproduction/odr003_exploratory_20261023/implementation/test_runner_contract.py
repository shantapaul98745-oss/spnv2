"""Real-artifact CPU contracts for runner preflight; never calls run/model/CUDA."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import unittest

import run_exploration as runner
from data_boundary import load_image_manifest


PLAN = runner.PLAN


def bind(path):
    return {'path': str(path), 'sha256': runner.sha(path)}


def task_for(method='O', domain='lightbox', stream='offline'):
    receipt_path=PLAN/'preflight/official_source_receipt.json'
    receipt=runner.read_json(receipt_path)
    core_doc=runner.read_json(receipt['source_core_manifest_path'])
    prepared_path=PLAN/'preflight/data_boundary_prepared.json'
    prepared=runner.read_json(prepared_path)
    def prepared_binding(key):
        item=prepared['artifacts'][domain+'/'+key]
        return {'path':item['path'],'sha256':item['sha256']}
    bindings={
        'official_receipt':bind(receipt_path),
        'source':{'path':receipt['source_path'],'sha256':receipt['source_sha256']},
        'config':{'path':receipt['config_path'],'sha256':receipt['config_sha256']},
        'source_core_manifest':{'path':receipt['source_core_manifest_path'],'sha256':receipt['source_core_manifest_sha256']},
        'data_prepared':bind(prepared_path),
        'data_verified':bind(PLAN/'preflight/data_boundary_verified.json'),
        'developer_manifest':prepared_binding('developer'),
        'developer_labels':prepared_binding('developer_labels'),
        'camera':{'path':core_doc['camera_path'],'sha256':core_doc['camera_sha256']},
        'keypoints':{'path':core_doc['keypoint_path'],'sha256':core_doc['keypoint_sha256']},
    }
    if method!='O': bindings['stream_manifest']=prepared_binding('stream_'+stream)
    name='contract_only_'+method+'_'+domain
    return {'schema':'odr003-task-v1','run_id':name,'method':method,'domain':domain,'stream':stream,
        'seed':2021 if method=='O' else int(stream),'output_dir':str(PLAN/'runs'/name),
        'bindings':bindings,'code_sha256':{name:runner.sha(Path(runner.__file__).parent/name) for name in runner.REQUIRED_CODE},
        'checkpoint_every_updates':64}


class RunnerContractTests(unittest.TestCase):
    def test_01_actual_offline_preflight_passes(self):
        task=task_for()
        self.assertEqual(runner.verify_task(task),Path(task['output_dir']))
        paths,receipt,core,prepared=runner.verify_bindings(task)
        self.assertEqual(receipt['status'],'passed')
        self.assertEqual(prepared['counts']['lightbox']['developer'],512)
        self.assertEqual(paths['source'],Path(receipt['source_path']))
        self.assertTrue(core.is_dir())
        self.assertFalse(Path(task['output_dir']).exists())

    def test_02_actual_original_a_preflight_passes(self):
        task=task_for('A','sunlamp','2023')
        runner.verify_task(task)
        paths,_,_,_=runner.verify_bindings(task)
        manifest=load_image_manifest(paths['stream_manifest'])
        self.assertEqual(manifest['seed'],2023)
        self.assertEqual(len(manifest['items']),1024)
        self.assertEqual({x['domain'] for x in manifest['items']},{'sunlamp'})
        self.assertFalse(manifest['contains_gt'])

    def test_03_wrong_source_hash_rejected(self):
        task=task_for()
        task['bindings']['source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Bound file changed'):
            runner.verify_bindings(task)

    def test_04_cross_domain_stream_rejected(self):
        task=task_for('A','lightbox','2021')
        other=task_for('A','sunlamp','2021')
        task['bindings']['stream_manifest']=other['bindings']['stream_manifest']
        with self.assertRaisesRegex(ValueError,'Adaptation stream differs'):
            runner.verify_bindings(task)

    def test_05_unknown_method_rejected(self):
        task=task_for();task['method']='B2'
        with self.assertRaisesRegex(ValueError,'Unknown method'):
            runner.verify_task(task)

    def test_06_code_drift_rejected(self):
        task=task_for();task['code_sha256']['prediction.py']='f'*64
        with self.assertRaisesRegex(ValueError,'Implementation hash mismatch'):
            runner.verify_task(task)

    def test_07_real_cfg_path_overrides_only(self):
        task=task_for('A','sunlamp','2022')
        paths,_,core,_=runner.verify_bindings(task)
        effective,changes=runner.configure(task,paths,core)
        from config import cfg as defaults
        from config.efficientdet import EFFICIENTDET_EXTRAS
        original=defaults.clone();original.defrost();original.merge_from_file(str(paths['config']))
        original.MODEL.BACKBONE.EXTRA=EFFICIENTDET_EXTRAS[original.MODEL.BACKBONE.NAME]
        def flatten(tree,prefix=''):
            result={}
            for key,value in tree.items():
                name=prefix+key
                if hasattr(value,'items'):result.update(flatten(value,name+'.'))
                else:result[name]=value
            return result
        before,after=flatten(original),flatten(effective)
        self.assertEqual(set(before),set(after))
        for key in before:
            if key in changes:self.assertEqual(after[key],changes[key],key)
            else:self.assertEqual(before[key],after[key],key)
        self.assertEqual(effective.TRAIN.LR,1e-4)
        self.assertEqual(effective.TRAIN.WD,0.)
        self.assertEqual(effective.TRAIN.ACCUM_STEPS,1)
        self.assertEqual(effective.ODR.IMAGES_PER_BATCH,4)
        self.assertFalse(effective.FP16)
        self.assertEqual(list(effective.MODEL.HEAD.LOSS_HEADS),[])
        self.assertEqual(effective.DATASET.CAMERA,str(paths['camera']))
        self.assertEqual(effective.DATASET.KEYPOINTS,str(paths['keypoints']))
        self.assertEqual(effective.MODEL.PRETRAIN_FILE,str(paths['source']))
        self.assertEqual(effective.TRAIN.TRAIN_CSV,'sunlamp/labels/forbidden.csv')
        self.assertTrue(effective.is_frozen())

    def test_08_image_manifests_have_no_labels(self):
        for domain in ('lightbox','sunlamp'):
            for seed in (2021,2022,2023):
                manifest=load_image_manifest(PLAN/'data'/domain/f'stream_{seed}.json')
                self.assertFalse(manifest['contains_gt'])
                self.assertEqual(len(manifest['items']),1024)
                for row in manifest['items']:
                    self.assertEqual(set(row),{'sample_id','domain','filename','image_sha256'})

    def test_99_gt_audit_boundary_actual_paths(self):
        # Install exactly once, last: sys.addaudithook has no removal API.
        developer=PLAN/'data/lightbox/dev_labels.json'
        other=PLAN/'data/sunlamp/dev_labels.json'
        boundary=runner.GTBoundary(developer)
        boundary.phase='adaptation'
        with self.assertRaisesRegex(PermissionError,'Developer GT forbidden'):
            with developer.open('rb'):pass
        boundary.phase='evaluation'
        # Opening allowed, but this contract reads zero label bytes.
        with developer.open('rb') as f:self.assertEqual(f.read(0),b'')
        with self.assertRaisesRegex(PermissionError,'Developer GT forbidden'):
            with other.open('rb'):pass
        for original in (runner.DATA_ROOT/'lightbox/labels/test.csv',runner.DATA_ROOT/'sunlamp/test.json'):
            with self.assertRaisesRegex(PermissionError,'Full target GT source forbidden'):
                with original.open('rb'):pass
        self.assertEqual(boundary.developer_reads,1)
        self.assertEqual(len(boundary.denied),4)


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU contract requires CUDA_VISIBLE_DEVICES empty')
    unittest.main(verbosity=2)
