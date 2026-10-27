"""Explicit streaming migration handoff: w16 ready first, archives in background.

Original complete transfer is retained as an archive/reproduction objective,
not a prerequisite for scientific windows whose exact dependencies are ready.
Staging bytes and patch bases never move; verified immutable inodes are linked
into the admitted canonical namespace, with original gates and ledger intact.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import uuid

try:
    from . import namespace_runner as ns
    from . import dispatch_config as dc
except ImportError:
    import namespace_runner as ns
    import dispatch_config as dc

U = Path('/home/ps/Downloads/UsrSourceCode/LuXinghai')
M = U / 'spnv2_runs/migrations/sysu_20261027'
S = M / 'source_snapshot'
D = U / ns.DIAGNOSTIC
J = Path(ns.LOGICAL_ROOT)
WAIT_SECONDS = 300
import importlib.util
INPUTS = {'asset_manifest', 'candidate_index', 'candidate_build_receipt', 'source_fence',
          'reconciled_state', 'source_index', 'original_candidate_index', 'source_budget_identity',
          'source_budget_journal', 'image_manifest', 'image_cache_receipt', 'storage_plan', 'screen_probe',
          'metadata_transfer', 'supplemental_closure','source_controller_status','source_fence_intent','streaming_dependencies'}
INPUT_RELATIVE_PATHS=dict(asset_manifest='asset_transfer_manifest.json',candidate_index='runtime_candidate/index_candidate.json',
    candidate_build_receipt='runtime_candidate/build_receipt.json',source_fence='source_fence_receipt.json',
    reconciled_state='source_handoff_snapshot_20261027/reconciled_state.json',source_index='source_handoff_snapshot_20261027/index.json',
    original_candidate_index='source_handoff_snapshot_20261027/index_candidate.json',
    source_budget_identity='source_handoff_snapshot_20261027/budget/identity.json',
    source_budget_journal='source_handoff_snapshot_20261027/budget/journal.jsonl',
    image_manifest='required_images_spec_20261027.json',image_cache_receipt='required_image_cache_receipt.json',
    storage_plan='storage_plan_20261027.json',screen_probe='namespace_screen_probe.json',
    metadata_transfer='metadata_transfer_verification.json',supplemental_closure='supplemental_asset_closure.json',
    source_controller_status='source_handoff_snapshot_20261027/controller_status.json',source_fence_intent='source_fence_intent.json',streaming_dependencies='pipeline_optimization_20261027/window_dependencies.json')


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def signature(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def owned(path, root=None):
    path = Path(path)
    ns.require(path.absolute() == path.resolve(), 'Refusing symlink/alias path: ' + str(path))
    if root is not None:
        ns.require(path.is_relative_to(Path(root)), 'Mutation escapes its owned root')
    if path.exists():
        ns.require(path.stat().st_uid == os.getuid(), 'Unexpected file owner: ' + str(path))
    return path


def immutable_bytes(path, data):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ns.require(path.is_file() and path.read_bytes() == data, 'Immutable evidence already differs: ' + str(path))
        return path
    pending = path.with_name('.pending-' + uuid.uuid4().hex)
    fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    try: os.link(pending, path)
    except FileExistsError:
        ns.require(path.is_file() and not path.is_symlink() and path.read_bytes() == data, 'Immutable publication race differs')
    pending.unlink(); fsync_dir(path.parent)
    return path


def immutable(path, value):
    return immutable_bytes(path, raw_json(value))


def atomic(path, value):
    path = owned(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.pending-' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw_json(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path); fsync_dir(path.parent)


def bind(path, logical=False):
    path = Path(path)
    target = str(J / path.relative_to(U)) if logical else str(path)
    return dict(path=target, sha256=ns.sha(path))


def mapped(path):
    path = Path(path)
    ns.require(path == J or path.is_relative_to(J), 'Only preserved project logical paths are mapped')
    return U / path.relative_to(J)


def checked(item):
    path = owned(item['path'])
    ns.require(ns.sha(path) == item['sha256'], 'Launch-spec input changed: ' + str(path))
    return ns.metadata(path)


def identity(nonce):
    proc = Path('/proc') / str(os.getpid()); stat = (proc/'stat').read_text()
    command = (proc/'cmdline').read_bytes()
    return dict(pid=os.getpid(), uid=os.getuid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cmdline_sha256=sha_bytes(command), argv=[x.decode() for x in command.split(b'\0') if x],
        nonce=nonce, sty=os.environ.get('STY'))


def exact_live(record, nonce_name, nonce):
    try:
        proc = Path('/proc') / str(record['pid']); stat = (proc/'stat').read_text()
        return (proc.stat().st_uid == os.getuid()
            and record['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and record['starttime'] == int(stat[stat.rfind(')')+2:].split()[19])
            and record['cmdline_sha256'] == sha_bytes((proc/'cmdline').read_bytes())
            and (nonce_name + '=' + nonce).encode() in (proc/'environ').read_bytes().split(b'\0'))
    except (OSError, KeyError, ValueError):
        return False


@contextmanager
def exclusive(path):
    path = owned(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally: os.close(fd)


class TreeMover:
    """Same-filesystem, no-overwrite rename with durable per-path intents."""
    def __init__(self, root, records, archived_locks, lock_paths=()):
        self.root = Path(root)
        self.records,self.archived=owned(records,self.root),owned(archived_locks,self.root)
        self.locks = {str(Path(x)) for x in lock_paths}

    def _record(self, source):
        return self.records / hashlib.sha256(str(source).encode()).hexdigest()

    def move(self, source, destination):
        source, destination = owned(source, self.root), owned(destination, self.root)
        base = self._record(source); intent_path, done = base.with_suffix('.intent.json'), base.with_suffix('.done.json')
        if done.exists():
            proof = ns.metadata(owned(done,self.root))
            ns.require(proof['source']==str(source) and proof['destination']==str(destination), 'Completed move receipt changed')
            ns.require(not source.exists() and destination.exists(), 'Completed rename endpoints changed')
            if proof['mode'] == 'rename':
                ns.require(destination.stat().st_ino == proof['source_stat']['inode']
                    and destination.stat().st_dev == proof['source_stat']['device'], 'Completed rename inode changed')
            if proof['mode'] == 'lock_archive':
                archived=owned(self.archived/source.relative_to(self.root),self.root)
                ns.require(archived.is_file() and archived.stat().st_ino==proof['source_stat']['inode']
                    and archived.stat().st_dev==proof['source_stat']['device'],'Completed lock archive inode changed')
            return
        if intent_path.exists():
            intent = ns.metadata(owned(intent_path,self.root))
            ns.require(intent['source'] == str(source) and intent['destination'] == str(destination), 'Rename intent mismatch')
            ns.require(intent['mode'] in ('rename','merge','lock_archive')
                and all(PurePosixPath(name).name==name and name not in ('.','..') for name in intent['children']),
                'Invalid rename intent mode or child path')
        else:
            ns.require(source.exists(), 'Unexplained missing staging source')
            ns.require(source.stat().st_dev == destination.parent.stat().st_dev, 'Cross-filesystem moves are forbidden')
            if not destination.exists(): mode = 'rename'
            elif source.is_dir() and destination.is_dir(): mode = 'merge'
            elif str(destination) in self.locks and source.is_file() and destination.is_file(): mode = 'lock_archive'
            else: raise ns.MigrationConfigError('Refusing an existing destination: ' + str(destination))
            intent = dict(source=str(source), destination=str(destination), source_stat=signature(source), mode=mode,
                          children=sorted(x.name for x in source.iterdir()) if mode == 'merge' else [])
            immutable(intent_path, intent)
        mode = intent['mode']
        if mode == 'rename':
            if source.exists():
                ns.require(not destination.exists(), 'Never overwrite an existing rename destination')
                ns.require(signature(source) == intent['source_stat'], 'Staging source changed after rename intent')
                os.rename(source, destination); fsync_dir(source.parent); fsync_dir(destination.parent)
            ns.require(destination.exists() and destination.stat().st_ino == intent['source_stat']['inode']
                and destination.stat().st_dev == intent['source_stat']['device'], 'Rename recovery cannot prove preserved inode')
        elif mode == 'merge':
            for name in intent['children']:
                self.move(source/name, destination/name)
            if source.exists():
                ns.require(not list(source.iterdir()), 'Unplanned files appeared during staging merge')
                source.rmdir(); fsync_dir(source.parent)
        else:
            ns.require(str(destination) in self.locks,'Only an explicitly held coordination lock can be archived')
            archived = owned(self.archived / source.relative_to(self.root),self.root)
            archived.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                ns.require(source.stat().st_size <= 4096 and destination.stat().st_size <= 4096
                    and ns.sha(source) == ns.sha(destination), 'Different coordination locks cannot be replaced')
                ns.require(not archived.exists(), 'Lock archive already exists unexpectedly')
                os.rename(source, archived); fsync_dir(source.parent); fsync_dir(archived.parent)
            ns.require(archived.exists() and archived.stat().st_ino == intent['source_stat']['inode'], 'Lock archive inode proof missing')
        immutable(done, intent)


class Controller:
    def __init__(self, spec, spec_binding):
        self.spec, self.spec_binding = spec, spec_binding
        self.nonce = spec['nonce']; self.run = owned(M/'deployment_controller'/self.nonce,U)
        self.run.mkdir(parents=True, exist_ok=True)
        self.birth=identity(self.nonce)
        self.attempt_path=self.run/'controller_attempts'/(str(self.birth['pid'])+'_'+str(self.birth['starttime'])+'.json')
        immutable(self.attempt_path,dict(schema='spnv2-sysu-controller-attempt-v1',identity=self.birth,
            spec=spec_binding,started_at=time.time(),controller_sha256=spec['controller']['sha256']))
        self.rollback = self.run/'rollback'; self.control = owned(M/'runtime_control',U)
        self.inputs = {name: checked(item) for name, item in spec['inputs'].items()
                       if name != 'source_budget_journal'}
        ns.require(ns.sha(spec['inputs']['source_budget_journal']['path']) == spec['inputs']['source_budget_journal']['sha256'],
                   'Final source budget journal binding changed')
        source_wait=self.inputs['source_controller_status'];fence_intent=self.inputs['source_fence_intent']
        immutable(self.run/'pending_costs.json',dict(schema='spnv2-sysu-pending-migration-costs-v1',
            source_completed_repair_resource_wait_seconds=source_wait['resource_wait_seconds'],
            source_repair_resource_wait_evidence=spec['inputs']['source_controller_status'],
            source_resource_wait_tail_wall_upper_seconds=max(0.,fence_intent['created_at']-source_wait['updated_at']),
            source_tail_evidence=spec['inputs']['source_fence_intent'],tail_is_model_time=False,
            migration_source_io_probe_costs_pending_target_receipts=True,
            target_controller_attempt_receipts=str(self.run/'controller_attempts'),
            source_budget_is_immutable=True,source_retention_settlement_not_repeated=True,model_forwards_added=0))

    def status(self, phase, **values):
        atomic(self.run/'status.json', dict(schema='spnv2-sysu-deployment-controller-status-v1',phase=phase,
            identity=identity(self.nonce),spec=self.spec_binding,updated_at=time.time(),model_forwards=0,**values))

    def common_gate(self):
        i = self.inputs; fence = i['source_fence']; state = i['reconciled_state']; build = i['candidate_build_receipt']
        ns.require(ns.sha(__file__) == self.spec['controller']['sha256'], 'Controller code changed after launch')
        for item in self.spec['inputs'].values():
            ns.require(ns.sha(item['path']) == item['sha256'], 'Bound preparation input changed while waiting')
        ns.require(fence['schema'] == 'spnv2-source-dispatch-fenced-v1' and fence['status'] == 'source_dispatch_fenced'
            and fence['no_scientific_workers_interrupted'] is True and fence['actual_controller_wrapper_exit_code'] == 143
            and fence['source_budget']['unresolved_attempt_ids'] == []
            and fence['source_budget']['time_hard_stop_enabled'] is False
            and fence['source_budget']['max_active_seconds'] is None
            and fence['source_budget_journal_sha256'] == self.spec['inputs']['source_budget_journal']['sha256']
            and fence['reconciled_state']['sha256'] == self.spec['inputs']['reconciled_state']['sha256']
            and fence['guardian']['status'] == 'holding_source_dispatch_fence'
            and set(fence['guardian']['holds']) == {'diagnostic_pipeline','diagnostic_manager'}, 'Source fence/final budget gate failed')
        ns.require(state['active'] is None and len(state['main']) == 27 and len(state['accepted']) == 26
            and state['pilot_admitted'] is True and state['budget_policy'] == i['candidate_index']['budget_policy'], 'Reconciled source state is incomplete')
        ns.require(build['schema'] == 'sysu-runtime-candidate-build-v1' and build['not_deployed'] is True
            and build['candidate_index_sha256'] == self.spec['inputs']['candidate_index']['sha256']
            and build['preserved_accepted_count'] == 26 and build['preserved_main_count'] == 27
            and build['modified_runtime_files'] == ['manager.py'], 'Candidate build binding failed')
        index, original = i['candidate_index'], i['original_candidate_index']
        ns.require(dc.same_scientific_code(index['implementation_sha256'], original['implementation_sha256'])
            and index['protocol'] == original['protocol'] and index['budget_policy'] == original['budget_policy']
            and index['frozen_inputs'] == original['frozen_inputs'], 'Candidate changes science/policy/inputs')
        ns.require(index['sysu_task_revision']=='sysu_migration_20261027'
            and set(index['migration_extensions'])==ns.CODE_NAMES,'Unexpected task revision or extension set')
        extension_paths={name:str(J/'spnv2/reproduction/sysu_migration_20261027'/(name+'.py'))
            for name in ('namespace_runner','bootstrap','dispatch_config')}
        extension_paths.update(storage_adapter=str(J/'spnv2/reproduction/sysu_migration_20261027/storage_adapter_v1.py'),
            storage_policy=str(J/'spnv2/reproduction/storage_management_20261027/storage_policy.py'))
        for name, item in index['migration_extensions'].items():
            ns.require(item['path']==extension_paths[name] and ns.sha(mapped(item['path'])) == item['sha256'],
                'Static extension path/bytes changed: ' + name)
        manager = U/'spnv2/reproduction/sysu_migration_20261027/manager_sysu.py'
        ns.require(ns.sha(manager) == index['implementation_sha256']['manager.py'] == build['new_manager_sha256'], 'Prepared manager differs')
        cache, images = i['image_cache_receipt'], i['image_manifest']
        ns.require(cache['schema'] == 'spnv2-required-image-cache-v1' and cache['status'] == 'verified_exact_frozen_bytes'
            and cache['manifest_sha256'] == self.spec['inputs']['image_manifest']['sha256']
            and cache['image_count'] == 2171 and cache['total_bytes'] == 61849173 and cache['mismatch_count'] == 0
            and images['image_count'] == 2171 and cache['shared_dataset_unchanged'] is True, 'Exact image cache missing')
        storage = i['storage_plan']; remaining = set(indexrow['run_id'] for indexrow in index['tasks']) - set(state['main'])
        ns.require(storage['schema'] == 'spnv2-experiment-storage-plan-v1'
            and set(storage['remaining_scientific_run_ids']) == remaining and len(remaining) == 21
            and storage['per_isolated_store_peak_bytes'] == 8*1024**3
            and storage['ordinary_recovery_keep_latest'] == 2
            and storage['temporary_root']=='/tmp/spnv2_sysu_20261026'
            and storage['cache_root']=='/home/ps/.cache/spnv2_sysu_20261026'
            and storage['no_scientific_protocol_or_forward_budget_change'] is True, 'Explicit storage plan is not the frozen21-store plan')
        probe = i['screen_probe']
        ns.require(probe['status'] == 'passed' and probe['worker_exit_code'] == 0 and probe['fixed_device_bind'] is True
            and probe['worker']['uid'] == 1000 and probe['worker']['sty'] and probe['worker']['torch_imported'] is False,
            'Existing zero-model nested-screen acceptance is missing')
        metadata=i['metadata_transfer'];closure=i['supplemental_closure']
        ns.require(metadata['matching_files']==metadata['expected_files'] and metadata['mismatches']==[]
            and metadata['models_loaded'] is False and closure['schema']=='sysu-required-supplemental-assets-v1',
            'Initial metadata/dependency closure acceptance is missing')
        manifest={row['relative_path']:row for row in i['asset_manifest']['files']}
        for row in closure['entries']:
            if not row['required_for_current_paths']: continue
            actual=manifest.get(row['relative_staging_path'])
            ns.require(actual and actual['sha256']==row['expected_sha256'] and actual['bytes']==row['bytes'],
                'Required supplemental dependency is absent from the exact transfer manifest')
        delta_path=self.run/'candidate_science_delta.json'
        expected_delta_inputs={name:self.spec['inputs'][name] for name in ('candidate_index','source_index')}
        if delta_path.exists():
            delta=ns.metadata(delta_path)
            ns.require(delta['inputs']==expected_delta_inputs and delta['scientific_fields_unchanged'] is True
                and delta['model_forwards']==0 and len(delta['checked_rows'])==21
                and {row['run_id'] for row in delta['checked_rows']}==remaining,
                'Previously accepted candidate delta binding/scope differs')
        else:
            old_rows={row['run_id']:row for row in i['source_index']['tasks']}
            ns.require(len(index['tasks'])==48 and len(old_rows)==48
                and {row['run_id'] for row in index['tasks']}==set(old_rows),'Frozen forty-eight run identities changed')
            checked_rows=[]
            for row in index['tasks']:
                old=old_rows[row['run_id']]
                if row['run_id'] in state['main']:
                    ns.require(row==old,'A completed/accepted index row was changed')
                    continue
                old_logical=Path(old['task_path'])
                ns.require(old_logical.is_relative_to(J/ns.DIAGNOSTIC/'tasks') and old_logical.suffix=='.json',
                    'Original task metadata escapes the fixed diagnostic task tree')
                original_path=S/old_logical.relative_to(J)
                if not original_path.exists(): original_path=mapped(old_logical)
                ns.require(Path(row['task_path']).parent==J/ns.DIAGNOSTIC/'tasks'/index['sysu_task_revision']
                    and Path(row['task_path']).name==row['run_id']+'.json','Prepared main task path escapes its new technical revision')
                new_path=M/'runtime_candidate/tasks'/index['sysu_task_revision']/Path(row['task_path']).name
                ns.require(ns.sha(original_path)==old['task_sha256'] and ns.sha(new_path)==row['task_sha256'],
                    'Original/prepared task metadata binding changed')
                old_task=ns.metadata(original_path);new_task=ns.metadata(new_path)
                ns.require({k:v for k,v in old_task.items() if k!='implementation_sha256'}==
                           {k:v for k,v in new_task.items() if k!='implementation_sha256'}
                    and new_task['implementation_sha256']==index['implementation_sha256']
                    and dc.same_scientific_code(old_task['implementation_sha256'],original['implementation_sha256']),
                    'A prepared main changed scientific task fields instead of only the metadata-manager binding')
                checked_rows.append(dict(run_id=row['run_id'],original_sha256=old['task_sha256'],new_sha256=row['task_sha256']))
            ns.require(len(checked_rows)==21,'Candidate delta is not exactly the twenty-one unstarted mains')
            immutable(delta_path,dict(inputs=expected_delta_inputs,checked_rows=checked_rows,
                scientific_fields_unchanged=True,model_forwards=0))

    def verify_streaming_plan(self):
        index=self.inputs['candidate_index'];plan=self.inputs['streaming_dependencies']
        helper=U/'spnv2/reproduction/sysu_migration_20261027/streaming_assets_v1.py'
        expected=dict(path=str(J/'spnv2/reproduction/sysu_migration_20261027/streaming_assets_v1.py'),sha256=ns.sha(helper))
        ns.require(index.get('streaming_migration')==dict(code=expected,dependencies=bind(self.spec['inputs']['streaming_dependencies']['path'],True)),
            'Candidate does not bind the exact streaming code/dependency plan')
        module_spec=importlib.util.spec_from_file_location('spnv2_streaming_controller',helper)
        self.stream=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(self.stream)
        ns.require(plan['schema']==self.stream.SCHEMA and plan['frozen_order']==[r['run_id'] for r in index['tasks']]
            and set(plan['windows'])==set(plan['frozen_order']) and plan['asset_manifest']==bind(self.spec['inputs']['asset_manifest']['path'],True),
            'Streaming plan escapes original manifest/frozen order')
        missing_main=set(self.inputs['reconciled_state']['main'])-set(self.inputs['reconciled_state']['accepted'])
        ns.require(missing_main=={'g50p_diag_v1_sunlamp_s2021_w0016'},'Initial streaming task must be saved w16 verify')
        return plan

    def transfer_status(self):
        status=ns.metadata(M/'asset_receive_status.json')
        ns.require(status.get('phase') in ('receiving','complete') and status.get('error') is None
            and status.get('model_forwards')==0,'Background transfer failed; preserve evidence')
        return status

    def publish_assets(self):
        self.transfer_status();self.verify_streaming_plan()
        with exclusive(M/'streaming_publish.lock'):
            return self.stream.publish_verified(self.inputs['asset_manifest'],M,S,U)

    def verified_assets(self):
        self.transfer_status();plan=self.verify_streaming_plan()
        required=set(plan['common_relative_paths'])|set(plan['windows']['g50p_diag_v1_sunlamp_s2021_w0016']['verify_pair'])
        rows={row['relative_path']:row for row in self.inputs['asset_manifest']['files']}
        ns.require(required<=set(rows),'Initial dependencies escape original asset manifest')
        return all(self.stream.checked_receipt(M,rows[rel],staging_root=S) is not None for rel in required)

    def layout(self):
        receipt_path=M/'streaming_canonical_asset_layout_receipt.json'
        records=self.run/'metadata_copy_records';records.mkdir(parents=True,exist_ok=True)
        manifest_paths={row['relative_path'] for row in self.inputs['asset_manifest']['files']}
        locks={str(U/'spnv2_runs/plans'/name/'pipeline.lock') for name in ns.LOCK_PLANS}|{str(D/'manager.lock')}
        # Initial metadata is copied, never hardlinked: future append/write cannot
        # modify immutable staging history. Scientific blobs are separate links.
        for source in sorted(S.rglob('*')):
            owned(source,U)
            if not source.is_file():continue
            rel=source.relative_to(S)
            if rel.parts[:2] not in (('spnv2_runs','plans'),('spnv2_runs','scripts'),('spnv2_runs','frozen')):continue
            if str(rel) in manifest_paths or str(U/rel) in locks:continue
            ns.require(source.stat().st_size <= 128*1024**2,'Unexpected nonmanifest binary in metadata copy')
            destination=U/rel;marker=records/(sha_bytes(str(rel).encode())+'.json')
            if marker.exists():
                proof=ns.metadata(marker)
                ns.require(proof['source_sha256']==ns.sha(source),'Initial metadata source changed')
                continue
            ns.require(not destination.exists(),'Unexpected existing canonical metadata: '+str(rel))
            immutable_bytes(destination,source.read_bytes())
            immutable(marker,dict(source=str(source),destination=str(destination),source_sha256=ns.sha(source),bytes=source.stat().st_size))
        self.publish_assets()
        for rid in self.inputs['reconciled_state']['accepted']:
            (D/'pairs'/rid).mkdir(parents=True,exist_ok=True)
        immutable(receipt_path,dict(schema='spnv2-streaming-canonical-asset-layout-v1',
            status='verified_metadata_with_incremental_immutable_assets',target_root=str(U),source_staging=str(S),
            transfer_manifest_sha256=self.spec['inputs']['asset_manifest']['sha256'],
            dependencies=bind(self.spec['inputs']['streaming_dependencies']['path'],True),
            no_file_content_changes=True,staging_paths_preserved=True,source_dispatch_fenced=True,
            publication='receiptverified immutable hardlinks; original staging/bases remain intact',
            metadata_copy_records=str(records),controller=self.spec_binding))
        return receipt_path

    def publish_until_transfer_complete(self):
        while True:
            published=self.publish_assets();status=self.transfer_status()
            if status['phase']=='complete':
                ns.require(not published['missing'] and len(published['published'])==len(self.inputs['asset_manifest']['files']),
                    'Completed transfer lacks canonical receiptverified publications')
                self.status('streaming_transfer_complete_manager_continues',verified_files=status['verified_files'],
                    scientific_acceptance_pending=True,automatic_manager_restart=False);return
            self.status('manager_launched_background_transfer',verified_files=status['verified_files'],
                expected_files=status['expected_files'],next_check_at=time.time()+WAIT_SECONDS,
                scientific_acceptance_pending=True,automatic_manager_restart=False)
            time.sleep(WAIT_SECONDS)

    def archive_before(self, path):
        path = Path(path)
        if not path.exists(): return None
        ns.require(path.stat().st_size <= 128*1024**2, 'Rollback capture must be bounded metadata')
        archived = self.rollback/'before'/path.relative_to(U)
        marker = archived.with_name(archived.name+'.binding.json')
        if marker.exists(): return ns.metadata(marker)
        immutable_bytes(archived, path.read_bytes())
        value = dict(path=str(path),sha256=ns.sha(path),archive=str(archived))
        immutable(marker, value)
        return value

    def replace_metadata(self, path, data):
        path = owned(path, U); before = self.archive_before(path)
        if path.exists() and sha_bytes(path.read_bytes()) == sha_bytes(data): return
        ns.require(before is None or ns.sha(path) == before['sha256'], 'Unexpected metadata mutation; no overwrite allowed')
        temp = path.with_name('.pending-'+uuid.uuid4().hex); temp.parent.mkdir(parents=True,exist_ok=True)
        fd = os.open(temp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as h: h.write(data);h.flush();os.fsync(h.fileno())
        os.replace(temp,path);fsync_dir(path.parent)

    def install_runtime(self):
        index = self.inputs['candidate_index']; state = copy.deepcopy(self.inputs['reconciled_state'])
        for path in (D/'index.json',D/'status.json',D/'launches/registry.json',D/'implementation/manager.py',
                     D/'budget/identity.json',D/'budget/journal.jsonl'):
            self.archive_before(path)
        before_status=self.rollback/'before'/D.relative_to(U)/'status.json'
        old_state=ns.metadata(before_status)
        ns.require(ns.sha(before_status)==self.inputs['source_fence']['source_status_sha256'],
            'Staged source status differs from its final source-fence binding')
        ns.require(ns.sha(self.rollback/'before'/D.relative_to(U)/'index.json')==self.inputs['source_fence']['source_index_sha256'],
            'Staged original index differs from its final source-fence binding')
        original_registry=ns.metadata(self.rollback/'before'/D.relative_to(U)/'launches/registry.json')
        target_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        ns.require(all(row.get('expected_record',{}).get('boot_id')!=target_boot for row in original_registry.get('records',[])),
            'Unknown target allocation exists in the registry; refuse to clear it')
        if old_state.get('active'):
            active=old_state['active'];saved=state['main'].get(active['run_id'])
            ns.require(saved and saved['task']['sha256']==active['task_sha256'], 'Old active record lacks its exact completed main')
            exit_path=mapped(saved['exit_code_path']);outcome=ns.metadata(mapped(saved['outcome_path']))
            ns.require(exit_path.stat().st_size<=32 and exit_path.read_text().strip()=='0'
                and outcome['exit_code']==0 and outcome['status']=='pair_main_complete'
                and outcome['report_sha256']==saved['report']['sha256'], 'Old active main has no actual successful exit proof')
        for rid in self.inputs['storage_plan']['remaining_scientific_run_ids']:
            store=D/'pairs'/rid/'isolated_durable'
            ns.require(not store.exists() or (store.is_dir() and not list(store.iterdir())),
                'A supposedly unstarted output store already has state; never adopt/prune unknown allocations')
        final_journal = Path(self.spec['inputs']['source_budget_journal']['path']).read_bytes()
        old = D/'budget/journal.jsonl'
        ns.require(not old.exists() or final_journal.startswith(old.read_bytes()), 'Unknown target budget suffix cannot be refunded')
        self.replace_metadata(D/'budget/journal.jsonl',final_journal)
        identity_raw = Path(self.spec['inputs']['source_budget_identity']['path']).read_bytes()
        ns.require(not (D/'budget/identity.json').exists() or (D/'budget/identity.json').read_bytes() == identity_raw,
                   'Stable budget identity cannot change')
        self.replace_metadata(D/'budget/identity.json',identity_raw)
        for row in index['tasks']:
            if row['run_id'] in state['main']: continue
            logical = Path(row['task_path']); destination = mapped(logical)
            source = M/'runtime_candidate/tasks'/index['sysu_task_revision']/logical.name
            ns.require(ns.sha(source) == row['task_sha256'], 'Prepared21 task binding changed')
            immutable_bytes(destination,source.read_bytes())
        manager = U/'spnv2/reproduction/sysu_migration_20261027/manager_sysu.py'
        self.replace_metadata(D/'implementation/manager.py',manager.read_bytes())
        self.replace_metadata(D/'index.json',Path(self.spec['inputs']['candidate_index']['path']).read_bytes())
        new_binding = bind(D/'index.json',True)
        state.update(index=new_binding,active=None,manager=None,status='between_tasks',reason='verified_sysu_migration_handoff',
                     waiting_since=None,next_resource_check=0)
        state.setdefault('technical_revisions',[]).append(dict(name='sysu_migration_20261027',controller=self.spec_binding,
            source_fence=self.spec['inputs']['source_fence'],previous_state=self.spec['inputs']['reconciled_state'],
            new_index=new_binding,budget_preserved=True,main_reused=27,accepted_reused=26))
        self.replace_metadata(D/'status.json',raw_json(state))
        self.replace_metadata(D/'launches/registry.json',raw_json(dict(schema='spnv2.round2.registry.v1',
            project_id='g50p_risk_diagnostic_20261025',records=[])))
        return state

    def prepare_bundle(self, state, layout_receipt):
        self.control.mkdir(parents=True,exist_ok=True);(self.control/'config_versions').mkdir(exist_ok=True)
        index = self.inputs['candidate_index']; fence = self.inputs['source_fence']; storage = self.inputs['storage_plan']
        accepted = sorted(state['accepted']); current_index = bind(D/'index.json',True)
        handoff = dict(schema='spnv2-sysu-budget-handoff-v1',migration_id=ns.MIGRATION,status='final',
            source_fence_receipt=bind(self.spec['inputs']['source_fence']['path'],True),
            budget_identity=bind(D/'budget/identity.json',True),budget_journal=bind(D/'budget/journal.jsonl',True),
            diagnostic_index=current_index,budget_journal_bytes=fence['source_budget_journal_bytes'],
            accepted_run_ids=accepted,snapshot=fence['source_budget'])
        handoff_path=immutable(self.control/'budget_handoff.json',handoff)
        init_rel='spnv2_runs/frozen/sysu_external_cache/'+ns.INITIALIZATION_NAME
        init_receipt=M/'asset_receipts'/(sha_bytes(init_rel.encode())+'.json')
        frozen_path=mapped(index['frozen_inputs']['path'])
        ns.require(ns.sha(frozen_path)==index['frozen_inputs']['sha256'],'Frozen inputs changed')
        producer=ns.metadata(frozen_path)['producer']
        paths=dict(target_root=str(U),source_snapshot=str(U),repository=str(U/'spnv2'),
            environment='/home/ps/miniconda3/envs/SPNv2',logical_environment='/home/ps/anaconda3/envs/SPNv2',
            dataset='/home/ps/Downloads/DataSet/speedplus',logical_dataset='/home/DATA/Downloads/DataSet/speedplus',
            tmp=storage['temporary_root'],cache=storage['cache_root'],logical_root=str(J),control_root=str(self.control),
            logical_control_root=str(J/'spnv2_runs/migrations/sysu_20261027/runtime_control'))
        for directory in (Path(paths['tmp']),Path(paths['cache']),Path(paths['tmp'])/'screen'):
            owned(directory);directory.mkdir(parents=True,mode=0o700,exist_ok=True)
        ns.require((Path(paths['tmp'])/'screen').stat().st_mode&0o777 == 0o700,'Existing screen socket permissions differ')
        template=dict(schema=ns.SCHEMA,migration_id=ns.MIGRATION,paths=paths,screen_socket_dir=paths['tmp']+'/screen',
            source_dispatch_fenced=True,target_scientific_dispatch_authorized=True,source_fence_nonce=fence['nonce'],
            source_fence_receipt=bind(self.spec['inputs']['source_fence']['path'],True),budget_handoff_receipt=bind(handoff_path,True),
            diagnostic_index=current_index,effective_protocol=index['protocol'],budget_policy=index['budget_policy'],
            budget_identity=handoff['budget_identity'],budget_journal=handoff['budget_journal'],accepted_run_ids=accepted,
            new_task_revision=index['sysu_task_revision'],approved_tasks=[],approved_stores=[],code_bindings=index['migration_extensions'],
            original_durable_state=dict(path=producer['implementation_dir']+'/durable_state.py',
                sha256=producer['implementation_sha256']['durable_state.py']),
            original_implementation_sha256=self.inputs['original_candidate_index']['implementation_sha256'],
            original_run_ids=[row['run_id'] for row in index['tasks']],frozen_inputs=index['frozen_inputs'],
            original_producer=producer,
            store_policies={rid:dict(schema='spnv2-storage-policy-v1',retain_latest=2,estimated_peak_bytes=storage['per_isolated_store_peak_bytes'],
                protected_generations={}) for rid in storage['remaining_scientific_run_ids']},
            asset_layout_receipt=bind(layout_receipt,True),asset_transfer_manifest=bind(self.spec['inputs']['asset_manifest']['path'],True),
            streaming_migration=index['streaming_migration'],
            initialization_cache=dict(source=str(U/init_rel),target='/home/ps/.cache/torch/hub/checkpoints/'+ns.INITIALIZATION_NAME,
                sha256=ns.INITIALIZATION_SHA256,receipt=bind(init_receipt,True)),
            frozen_image_cache=dict(source=self.inputs['image_cache_receipt']['cache_root'],
                target=paths['logical_dataset']+'/sunlamp/images_768x512_RGB',
                manifest=bind(self.spec['inputs']['image_manifest']['path'],True),receipt=bind(self.spec['inputs']['image_cache_receipt']['path'],True)))
        template_path=immutable(self.control/'template.json',template)
        deployment=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=bind(template_path,True),
            index=current_index,controller=self.spec_binding,source_fence=template['source_fence_receipt'])
        deployment_path=immutable(self.control/'deployment.json',deployment)
        # Identical to the bound manager.verification_task: saved w16 main only.
        remaining_main=set(state['main'])-set(state['accepted'])
        ns.require(len(remaining_main)==1,'First task must be exactly the one saved unverified main')
        rid=next(iter(remaining_main));row=next(row for row in index['tasks'] if row['run_id']==rid)
        receipt=state['main'][rid];science=ns.metadata(mapped(row['task_path']))
        task=dict(schema='g50p-risk-diagnostic-task-v1',action='verify_pair',run_id=rid+'_verify_'+current_index['sha256'][:12],
            implementation_sha256=index['implementation_sha256'],run_task=receipt['task'],execution_receipt=receipt,
            all_attempts=receipt['attempts'],proof_path=science['output_dir']+'/proof.json',index_path=current_index['path'],
            index_binding=science['index_binding'],budget_policy=copy.deepcopy(index['budget_policy']))
        task_path=immutable(D/'tasks'/index['sysu_task_revision']/(task['run_id']+'.json'),task)
        config=dc.compose_single_task(bind(deployment_path,True),bind(task_path,True),resolve=mapped)
        version=dc.write_config_version(config,resolve=mapped)
        return config,bind(deployment_path,True),version,bind(task_path,True)

    def guarded_validation(self, config, deployment, version, task):
        result=self.control/'config_versions'/('validation_'+self.nonce+'.json')
        request=immutable(self.control/'validation_request.json',dict(schema='spnv2-sysu-metadata-validation-request-v1',
            controller=self.spec['controller'],nonce=self.nonce,deployment=deployment,config=version,task=task,
            output=str(J/result.relative_to(U))))
        python=config['paths']['logical_environment']+'/bin/python'
        base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True)
        argv=base[:base.index(python)]+[python,'-B','-s','-u',str(J/Path(__file__).resolve().relative_to(U)),
            '--validate-bundle',str(J/request.relative_to(U)),'--request-sha256',ns.sha(request)]
        log=self.run/'metadata_validation.log';intent=self.run/'validation_intent.json'
        command=['bash',str(U/'spnv2_runs/scripts/run_in_screen_guarded.sh'),'sysu_validate_'+self.nonce[:12],str(log),*argv]
        if not intent.exists():
            immutable(intent,dict(argv=command,nonce=self.nonce,started_at=time.time()))
            run=subprocess.run(command,cwd=U/'spnv2',env={**os.environ,'SCREENDIR':config['screen_socket_dir'],
                'SPNV2_RUN_NONCE':self.nonce},capture_output=True,text=True,timeout=30)
            immutable(self.run/'validation_wrapper.json',dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
            ns.require(run.returncode==0,'Metadata validation wrapper failed')
        deadline=time.monotonic()+120
        while not Path(str(log)+'.exitcode').exists():
            ns.require(time.monotonic()<deadline,'Metadata validation actual exit unresolved; no manager launch')
            time.sleep(.2)
        ns.require(Path(str(log)+'.exitcode').read_text().strip()=='0','Metadata validation actual exit failed')
        receipt=ns.metadata(result)
        ns.require(receipt['status']=='passed' and receipt['nonce']==self.nonce and receipt['torch_imported'] is False
            and receipt['config']==version and receipt['deployment']==deployment and receipt['model_forwards']==0,
            'Zero-model metadata validation receipt mismatch')
        ns.require(receipt['identity']['uid']==1000 and receipt['identity']['sty'], 'Validation lacks its owned screen identity')
        return bind(result,True)

    def existing_manager(self):
        intent=self.run/'manager_launch_intent.json'
        if not intent.exists(): return False
        row=ns.metadata(intent);state=ns.metadata(D/'status.json');record=state.get('manager')
        if record and record.get('manager_nonce')==row['nonce'] and exact_live(record,'SPNV2_MANAGER_NONCE',row['nonce']):
            self.status('manager_already_running',manager=record,scientific_acceptance_pending=True);return True
        exitfile=Path(row['log_path']+'.exitcode')
        if exitfile.exists():
            self.status('manager_terminal',actual_exit_code=exitfile.read_text().strip(),automatic_restart=False)
            return True
        raise ns.MigrationConfigError('Unresolved prior manager launch intent; refuse duplicate launch')

    def execute(self):
        self.common_gate();self.verify_streaming_plan()
        if self.existing_manager():
            if ns.metadata(self.run/'status.json')['phase']=='manager_already_running':
                self.publish_until_transfer_complete()
            return
        while True:
            if not self.verified_assets():
                self.status('waiting_asset_transfer',next_check_at=time.time()+WAIT_SECONDS);time.sleep(WAIT_SECONDS);continue
            try:
                with exclusive(M/'streaming_handoff.lock'):
                    with ExitStack() as stack:
                        for path in [U/'spnv2_runs/plans'/name/'pipeline.lock' for name in ns.LOCK_PLANS]+[D/'manager.lock']:
                            stack.enter_context(exclusive(path))
                        self.common_gate()
                        self.status('preparing_owned_handoff')
                        layout=self.layout();state=self.install_runtime()
                        config,deployment,version,task=self.prepare_bundle(state,layout)
                        proof=self.guarded_validation(config,deployment,version,task)
                        argv=ns.build_manager_bwrap(config,deployment,version)
                        nonce=self.spec['manager_nonce'];log=self.run/'generic_manager.log'
                        command=['bash',str(U/'spnv2_runs/scripts/run_in_screen_guarded.sh'),'sysu_manager_'+nonce[:12],str(log),*argv]
                        immutable(self.run/'manager_launch_intent.json',dict(nonce=nonce,argv=command,log_path=str(log),
                            deployment=deployment,validation=proof,config=version,created_at=time.time()))
                    # Release target project locks before the original manager
                    # acquires its own manager lock and admits workers normally.
                    launched=subprocess.run(command,cwd=U/'spnv2',env={**os.environ,'SCREENDIR':config['screen_socket_dir'],
                        'SPNV2_MANAGER_NONCE':nonce},capture_output=True,text=True,timeout=30)
                    immutable(self.run/'manager_wrapper.json',dict(returncode=launched.returncode,stdout=launched.stdout,stderr=launched.stderr))
                    ns.require(launched.returncode==0,'Generic manager wrapper failed; no automatic retry')
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        record=ns.metadata(D/'status.json').get('manager')
                        if record and record.get('manager_nonce')==nonce and exact_live(record,'SPNV2_MANAGER_NONCE',nonce):
                            self.status('manager_launched',manager=record,deployment=deployment,validation=proof,
                                scientific_acceptance_pending=True,first_gpu_proof_passed=False)
                            self.publish_until_transfer_complete();return
                        time.sleep(.2)
                    raise ns.MigrationConfigError('Manager exact identity unresolved; never launch a duplicate')
            except BlockingIOError:
                self.status('waiting_receiver_or_project_lock',next_check_at=time.time()+WAIT_SECONDS)
                time.sleep(WAIT_SECONDS)


def validate_bundle(request_path, expected_sha):
    ns.require(ns.sha(request_path)==expected_sha,'Metadata validation request changed')
    request=ns.metadata(request_path)
    ns.require(request['schema']=='spnv2-sysu-metadata-validation-request-v1'
        and ns.sha(Path(__file__))==request['controller']['sha256'],'Controller validation identity differs')
    ns.require(os.getuid()==1000 and os.environ.get('STY')
        and os.environ.get('SPNV2_RUN_NONCE')==request['nonce'],'Validation requires its independent owned screen/nonce')
    config=dc.read_bound(request['config'])
    again=dc.compose_single_task(request['deployment'],request['task'])
    ns.require(again==config,'Namespace recomposition differs from the native metadata preparation')
    ns.config_location(config,request['output'])
    ns.require('torch' not in sys.modules,'Metadata validation unexpectedly imported Torch')
    immutable(request['output'],dict(schema='spnv2-sysu-metadata-validation-v1',status='passed',nonce=request['nonce'],
        identity=identity(request['nonce']),config=request['config'],deployment=request['deployment'],
        torch_imported=False,model_forwards=0,scientific_acceptance=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-spec');parser.add_argument('--launch-sha256');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--validate-bundle');parser.add_argument('--request-sha256')
    args=parser.parse_args()
    if args.validate_bundle:
        validate_bundle(args.validate_bundle,args.request_sha256);return
    ns.require(args.launch_spec and args.launch_sha256,'Explicit launch spec and SHA are required')
    spec_binding=dict(path=str(Path(args.launch_spec).resolve()),sha256=args.launch_sha256)
    spec=checked(spec_binding)
    ns.require(spec['schema']=='spnv2-sysu-deployment-controller-spec-v1' and spec['migration_id']==ns.MIGRATION
        and set(spec['inputs'])==INPUTS and spec['controller']['path']==str(Path(__file__).resolve())
        and ns.sha(__file__)==spec['controller']['sha256'],'Wrong controller launch binding')
    ns.require(all(item['path']==str(M/INPUT_RELATIVE_PATHS[name]) for name,item in spec['inputs'].items()),
        'Controller inputs must be the fixed project metadata paths, never model/state/GT files')
    ns.require(Path(spec_binding['path']).is_relative_to(M) and U.resolve()==U and os.getuid()==1000,'Wrong target host/root/owner')
    if not args.execute:
        print(json.dumps(dict(status='spec_inspected_only',experimental_dispatch=False,controller=spec['controller'])));return
    ns.require(spec.get('target_scientific_dispatch_authorized') is True and os.environ.get('STY')
        and os.environ.get('SPNV2_DEPLOYMENT_NONCE')==spec['nonce']
        and len(spec['nonce'])==len(spec['manager_nonce'])==32
        and spec['nonce']!=spec['manager_nonce']
        and all(ch in '0123456789abcdef' for ch in spec['nonce']+spec['manager_nonce']),
        'Explicit owned controller screen/authorization/nonce is required')
    controller=Controller(spec,spec_binding)
    with exclusive(M/'deployment_controller.lock'):
        started=time.monotonic();exit_code=0
        try: controller.execute()
        except Exception as error:
            exit_code=1
            controller.status('failed_preserve_and_block',error_type=type(error).__name__,error=str(error));raise
        finally:
            immutable(controller.attempt_path.with_suffix('.exit.json'),dict(exit_code=exit_code,finished_at=time.time(),
                elapsed_seconds=time.monotonic()-started,model_forwards=0,identity=controller.birth))


if __name__=='__main__':
    main()
