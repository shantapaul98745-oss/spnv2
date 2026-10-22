"""CPU-only strict compatibility check for the verified official offline source."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import hashlib
import json
import os
import sys
import traceback

PLAN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
OLD = PLAN.parent / 'rebaseline_20261017'
CORE = OLD / 'odr/source_A/core'
CONFIG = OLD / 'odr/configs/fixed_phi3_A_lightbox_s2021_N1024_B4.yaml'
SOURCE = PLAN / 'assets/spnv2_efficientnetb3_fullconfig_offline.pth.tar'
EXPECTED_SHA = 'd8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc'
MANIFEST = PLAN / 'preflight/source_core_manifest.json'
REPORT = PLAN / 'preflight/official_strict_load.json'
RECEIPT = PLAN / 'preflight/official_source_receipt.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    assert os.environ.get('STY'), 'Run this verification in a unique guarded screen'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only verification required'
    assert not REPORT.exists() and not RECEIPT.exists(), 'Do not overwrite previous verification'
    assert SOURCE.stat().st_size == 55320971
    assert sha(SOURCE) == EXPECTED_SHA
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest['files'].items():
        assert sha(Path(manifest['root']) / relative) == expected, relative
    assert sha(CONFIG) == manifest['config_sha256']
    assert sha(manifest['keypoint_path']) == manifest['keypoint_sha256']
    sys.path.insert(0, str(CORE))
    import torch
    from config import cfg, update_config
    from nets import build_spnv2
    torch.set_num_threads(2)
    torch.manual_seed(2021)
    assert not torch.cuda.is_initialized()
    cached = Path.home() / '.cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth'
    assert cached.is_file(), 'Required cached ImageNet build dependency is unavailable'
    cached_sha = sha(cached)
    # Build the unmodified architecture; skip the old self-trained PRETRAIN_FILE.
    # The original file remains byte-identical, and every tensor is overwritten
    # below with the official offline state using a strict, weights-only load.
    update_config(cfg, SimpleNamespace(cfg=str(CONFIG), opts=['CUDA', 'False', 'MODEL.PRETRAIN_FILE', '']))
    assert cfg.MODEL.EFFICIENTDET_PHI == 3
    assert list(cfg.MODEL.HEAD.NAMES) == ['heatmap', 'efficientpose', 'segmentation']
    assert not cfg.MODEL.USE_GROUPNORM_BACKBONE
    assert cfg.MODEL.USE_GROUPNORM_HEADS
    assert list(cfg.DATASET.INPUT_SIZE) == [768, 512]
    assert not cfg.FP16
    assert cfg.ODR.NUM_TRAIN_SAMPLES == 1024 and cfg.ODR.IMAGES_PER_BATCH == 4
    model = build_spnv2(cfg).cpu().eval()
    state = torch.load(SOURCE, map_location='cpu', weights_only=True)
    assert isinstance(state, dict) and state
    assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()), 'Official file must be a direct state_dict'
    before = model.state_dict()
    assert set(before) == set(state), {'missing': sorted(set(before)-set(state)), 'unexpected': sorted(set(state)-set(before))}
    metadata = []
    for k, tensor in state.items():
        assert before[k].shape == tensor.shape and before[k].dtype == tensor.dtype, k
        assert torch.isfinite(tensor).all(), k
        metadata.append({'key': k, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': tensor.numel()})
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    head_counts = {name: sum(k.startswith('heads.'+str(i)+'.') for k in state) for i, name in enumerate(model.head_names)}
    assert all(n > 0 for n in head_counts.values())
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert not torch.cuda.is_initialized()
    assert sha(SOURCE) == EXPECTED_SHA and sha(CONFIG) == manifest['config_sha256'] and sha(cached) == cached_sha
    for relative, expected in manifest['files'].items():
        assert sha(Path(manifest['root']) / relative) == expected, relative
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = dict(status='passed', checked_at=now, strict_model_load_performed=True,
                  source_path=str(SOURCE), source_sha256=EXPECTED_SHA,
                  config_path=str(CONFIG), config_sha256=sha(CONFIG),
                  source_core_manifest_path=str(MANIFEST), source_core_manifest_sha256=sha(MANIFEST),
                  strict_load=True, weights_only=True, direct_state_dict=True,
                  missing_keys=[], unexpected_keys=[], tensor_count=len(state),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  tensor_metadata=metadata, head_tensor_counts=head_counts,
                  all_tensors_finite=True, all_shapes_and_dtypes_match=True,
                  all_loaded_tensors_equal_source=True, device='cpu', cuda_initialized=False,
                  cached_imagenet_build_dependency={'path': str(cached), 'sha256': cached_sha, 'used_only_for_architecture_build_then_fully_overwritten': True},
                  config_runtime_overrides={'CUDA':False, 'MODEL.PRETRAIN_FILE':''},
                  forward_passes=0, gt_read=False, source_or_frozen_code_modified=False,
                  screen=os.environ['STY'], torch_version=torch.__version__,
                  verifier_path=str(Path(__file__).resolve()), verifier_sha256=sha(__file__))
    save(REPORT, report)
    copy_record = json.loads((PLAN/'preflight/official_asset_copy.json').read_text())
    receipt = dict(schema_version=1, status='passed', checked_at=now,
                   source_offline_identity_verified=True, strict_model_load_performed=True,
                   source='author_official', model_identity='SPNv2 phi3-BN full EHS offline/source',
                   author_repository='https://github.com/tpark94/spnv2',
                   author_weights_url='https://1drv.ms/f/c/fa28139a835eeb46/EgaEWqai0j5JkYLdAt0B8msBtg_-Q3UpmrWhHUbnSqMX_Q',
                   author_filename=SOURCE.name, acquisition='verified_local_official_asset_copy',
                   identity_basis=copy_record['identity_basis'],
                   author_published_checksum_claimed=False,
                   source_path=str(SOURCE), source_sha256=EXPECTED_SHA, source_bytes=SOURCE.stat().st_size,
                   config_path=str(CONFIG), config_sha256=sha(CONFIG),
                   source_core_manifest_path=str(MANIFEST), source_core_manifest_sha256=sha(MANIFEST),
                   keypoint_path=manifest['keypoint_path'], keypoint_sha256=manifest['keypoint_sha256'],
                   strict_load_report_path=str(REPORT), strict_load_report_sha256=sha(REPORT),
                   copy_record_path=str(PLAN/'preflight/official_asset_copy.json'),
                   copy_record_sha256=sha(PLAN/'preflight/official_asset_copy.json'),
                   scope='Compatibility verification only; no prediction, adaptation, or accuracy result')
    save(RECEIPT, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('identity_basis',)}))
    print('PASS: all official offline tensors loaded strictly and identically on CPU; no GT or forward')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(PLAN/'preflight/official_strict_load_failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
