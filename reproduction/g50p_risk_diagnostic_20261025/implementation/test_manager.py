import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import manager
from budget import atomic_json


class Clock:
    def __init__(self):self.now=10000.;self.sleeps=0;self.maximum=1000
    def time(self):return self.now
    def monotonic(self):return self.now
    def sleep(self,seconds):
        self.now+=seconds;self.sleeps+=1
        if self.sleeps>self.maximum:raise TimeoutError('bounded mock loop')


class DiagnosticManagerContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.plan=Path(self.tmp.name);self.clock=Clock()
        for name in ('tasks','logs','launches'): (self.plan/name).mkdir()
        self.rows=[]
        for i in range(48):
            task=dict(schema='g50p-risk-diagnostic-task-v1',action='run_pair',run_id=f'window{i}',
                      implementation_sha256={},output_dir=str(self.plan/'runs'/f'window{i}'))
            path=self.plan/'tasks'/f'window{i}.json';atomic_json(path,task)
            self.rows.append(dict(run_id=task['run_id'],task_path=str(path),task_sha256=manager.sha(path),pilot=i<2))
        a_task=dict(schema='g50p-risk-diagnostic-task-v1',action='extract_a',run_id='extract_A',implementation_sha256={})
        a_path=self.plan/'tasks/extract_A.json';atomic_json(a_path,a_task)
        self.index=dict(schema='g50p-risk-diagnostic-index-v1',implementation_sha256={},complete_ODR_runs=0,
             frozen_inputs=dict(path=str(self.plan/'frozen_inputs.json'),sha256='f'*64),
             tasks=self.rows,preprocessing=[dict(run_id='extract_A',task_path=str(a_path),task_sha256=manager.sha(a_path))],
             pilot_admission_template=dict(schema='g50p-risk-diagnostic-task-v1',action='verify_pilots',run_id='verify_pilots',implementation_sha256={}),
             summary_template=dict(schema='g50p-risk-diagnostic-task-v1',action='summarize_diagnostic',run_id='summary',implementation_sha256={}))
        self.index_path=self.plan/'index.json';atomic_json(self.index_path,self.index)
        self.launches=[]
        self.patches=[patch.object(manager,'PLAN',self.plan),patch.object(manager,'time',self.clock),
            patch.object(manager,'os',SimpleNamespace(name='posix',environ={'STY':'mock','SPNV2_MANAGER_NONCE':'n'*32})),
            patch.object(manager,'project_lock',side_effect=lambda path:nullcontext()),
            patch.object(manager,'validate_index',return_value=None),
            patch.object(manager,'manager_identity',return_value={'pid':1,'starttime':2}),
            patch.object(manager,'host_modules',return_value=(SimpleNamespace(assess_snapshot=lambda *a,**k:SimpleNamespace(allow_start=True,reasons=[])),
                SimpleNamespace(collect_snapshot=lambda *a,**k:{'captured_at':self.clock.time()}))),
            patch.object(manager.subprocess,'run',side_effect=AssertionError('No process launch in mock tests'))]
        for item in self.patches:item.start()
    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.tmp.cleanup()

    def fake_launch(self,state,task_path,run_id,index_path):
        self.launches.append(run_id);task=manager.read(task_path);nonce=f'{len(self.launches):032x}'
        log=self.plan/'logs'/f'{nonce}.log';log.write_text('mock')
        directory=self.plan/'launches'/nonce;directory.mkdir()
        record=dict(run_id=run_id,nonce=nonce,log_path=str(log),launch_directory=str(directory),
                    task_path=str(task_path),task_sha256=manager.sha(task_path),queued_at=self.clock.time())
        state['active']=record;state['attempts'].append(record);state['status']='running'
        action=task['action']
        if action=='run_pair':
            path=Path(task['output_dir'])/'report.json';atomic_json(path,{'run_id':run_id,'mock':True})
            outcome=dict(exit_code=0,status='pair_main_complete',report=str(path))
        elif action=='verify_pair':
            source=manager.checked(task['run_task']);science=source['run_id']
            proof=Path(task['proof_path']);atomic_json(proof,dict(status='passed',report_sha256=state['main'][science]['report']['sha256']))
            outcome=dict(exit_code=0,status='pair_verified',proof=str(proof))
        elif action=='extract_a':
            proof=self.plan/'a_proof.json';atomic_json(proof,{'status':'passed','model_forwards':0})
            outcome=dict(exit_code=0,status='a_extracted',proof=str(proof))
        elif action=='verify_pilots':
            self.assertEqual(len(state['accepted']),2)
            proof=self.plan/'pilot_proof.json';atomic_json(proof,{'status':'passed','allow_remaining_46':True})
            outcome=dict(exit_code=0,status='pilot_admitted',proof=str(proof))
        else:
            self.assertEqual(len(state['accepted']),48)
            summary=self.plan/'summary.json';atomic_json(summary,{'completed_windows':48})
            proof=self.plan/'delivery_proof.json';atomic_json(proof,dict(status='passed',summary_sha256=manager.sha(summary),
                complete_scope=True,completed_windows=48))
            atomic_json(directory/'budget_closed.json',dict(schema='g50p-diagnostic-budget-snapshot-v1',
                unresolved_attempt_ids=[],active_time_bound_violation_attempt_ids=[],
                active_seconds_charged_upper=123.,physical_forwards_reserved_upper=456.))
            outcome=dict(exit_code=0,status='diagnostic_delivered',proof=str(proof),summary=str(summary))
        Path(str(log)+'.exitcode').write_text('0');atomic_json(directory/'outcome.json',outcome)
        manager.write_state(state)

    def test_A_then_two_included_pilots_then_independent_admission_then_46_delivery_once(self):
        with patch.object(manager,'launch',side_effect=self.fake_launch):
            self.assertEqual(manager.main(self.index_path),0)
            launches=list(self.launches)
            self.assertEqual(manager.main(self.index_path),0)
            self.assertEqual(self.launches,launches)
        self.assertEqual(self.launches[:6],['extract_A','window0','window0_verify','window1','window1_verify','verify_pilots'])
        self.assertEqual(len([key for key in self.launches if key.startswith('window') and not key.endswith('_verify')]),48)
        self.assertEqual(len(self.launches),99)
        state=manager.read(self.plan/'status.json');self.assertEqual(state['status'],'complete')
        self.assertEqual(state['complete_ODR_runs'],0)
        final=manager.checked(state['delivery']['final_manifest'])
        self.assertEqual(final['active_seconds_charged_upper'],123.)
        self.assertTrue(final['scientific_scope_complete'])

    def test_budget_failure_blocks_without_launching_remaining_windows(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        active=state['active'];directory=Path(active['launch_directory'])
        Path(active['log_path']+'.exitcode').write_text('1')
        atomic_json(directory/'outcome.json',dict(status='budget_stopped',exit_code=1))
        manager.finish(state)
        self.assertEqual(state['status'],'budget_stopped')
        self.assertEqual(state['accepted'],{})

    def test_75_is_waiting_not_success_or_blind_failure_retry(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        active=state['active'];directory=Path(active['launch_directory'])
        Path(active['log_path']+'.exitcode').write_text('75')
        atomic_json(directory/'outcome.json',dict(status='waiting_resources',exit_code=75))
        manager.finish(state)
        self.assertIsNone(state['active']);self.assertEqual(state['main'],{})
        self.assertEqual(state['next_resource_check']-self.clock.time(),300)

    def test_missing_outcome_is_not_success(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        (Path(state['active']['launch_directory'])/'outcome.json').unlink()
        manager.finish(state)
        self.assertEqual(state['status'],'blocked');self.assertEqual(state['accepted'],{})

    def test_failed_pilot_admission_holds_remaining_46(self):
        state=manager.new_state(self.index_path)
        task=self.plan/'tasks/held.json';atomic_json(task,{'action':'verify_pilots'})
        log=self.plan/'logs/held.log';Path(str(log)+'.exitcode').write_text('0')
        directory=self.plan/'launches/held';directory.mkdir()
        atomic_json(directory/'outcome.json',dict(status='diagnostic_held',exit_code=0,reason='pilot_cost_exceeds_remaining_budget'))
        state['active']=dict(task_path=str(task),log_path=str(log),launch_directory=str(directory))
        manager.finish(state)
        self.assertEqual(state['status'],'held');self.assertFalse(state['pilot_admitted'])

    def test_budget_stop_delivers_partial_subset_without_any_next_launch(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        manager.finish(state)
        state.update(status='budget_stopped',reason='global_forward_or_time_limit')
        manager.write_state(state)
        with patch.object(manager,'launch',side_effect=AssertionError('Budget stop cannot dispatch')):
            self.assertEqual(manager.main(self.index_path),0)
        saved=manager.read(self.plan/'status.json')
        self.assertEqual(saved['status'],'partial_delivery_required')
        manifest=manager.checked(saved['partial_delivery'])
        self.assertFalse(manifest['scientific_scope_complete'])
        self.assertEqual(manifest['accepted_window_count'],0)
        self.assertEqual(manifest['main_window_count'],1)
        receipts=manager.checked(manifest['receipts'])
        self.assertEqual(receipts['main_without_independent_acceptance'],['window0'])
        self.assertEqual(len(receipts['not_executed']),47)
        self.assertEqual(receipts['new_model_forwards_in_delivery'],0)

    def test_partial_audit_receipts_state_missing_runs_instead_of_keyerror(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        manager.finish(state)
        path,_=manager.audit_task(self.index,state,'summary')
        task=manager.read(path);receipts=manager.checked(task['execution_receipts'])
        self.assertEqual(len(receipts['runs']),1)
        self.assertEqual(len(receipts['missing_main_run_ids']),47)
        self.assertEqual(len(receipts['missing_independent_acceptance_run_ids']),48)

    def test_failed_started_window_is_preserved_not_labelled_never_executed(self):
        state=manager.new_state(self.index_path)
        self.fake_launch(state,Path(self.rows[0]['task_path']),'window0',self.index_path)
        state.update(status='budget_stopped',reason='deadline_after_partial_panel')
        path=manager.partial_delivery(self.index,state)
        receipts=manager.checked(manager.read(path)['receipts'])
        self.assertEqual(receipts['attempted_without_complete_main'],['window0'])
        self.assertNotIn('window0',receipts['not_executed'])
        self.assertEqual(len(receipts['not_executed']),47)
        self.assertEqual(receipts['preserved_own_output_locations'][0]['run_id'],'window0')
        self.assertFalse(receipts['preserved_own_output_locations'][0]['independent_acceptance'])

    def test_pure_technical_index_revision_keeps_same_existing_budget_identity(self):
        state=manager.new_state(self.index_path);state['status']='budget_stopped'
        manager.partial_delivery(self.index,state)
        identity_path=self.plan/'budget/identity.json'
        before=identity_path.read_bytes()
        budget_snapshot=manager.read(self.plan/'delivery/partial_receipts.json')['budget']
        self.assertIsNotNone(budget_snapshot)
        # A separately authorized, no-worker technical migration changes the
        # dispatch index, never the frozen research inputs or cost directory.
        for path in (self.plan/'delivery').glob('*.json'):path.unlink()
        revised=copy.deepcopy(self.index);revised['implementation_sha256']={'technical_fix.py':'a'*64}
        state['index']['sha256']='b'*64
        manager.partial_delivery(revised,state)
        self.assertEqual(identity_path.read_bytes(),before)
        self.assertEqual(manager.read(identity_path)['identity'],
                         dict(project_id=manager.PROJECT,frozen_inputs_sha256='f'*64))

    def preserved_A_fixture(self):
        implementation=self.plan/'implementation';implementation.mkdir(exist_ok=True)
        code={}
        for name in ('host_entry.py','budget.py','extract_A.py'):
            path=implementation/name;path.write_text('# same accepted A module\n');code[name]=manager.sha(path)
        old_code={**code,'manager.py':'1'*64};new_code={**code,'manager.py':'2'*64}
        index=copy.deepcopy(self.index);index['implementation_sha256']=new_code;index['host_modules']={}
        row=index['preprocessing'][0]
        task=manager.read(row['task_path']);task['implementation_sha256']=old_code
        atomic_json(row['task_path'],task);row['task_sha256']=manager.sha(row['task_path'])
        original=copy.deepcopy(index);original['implementation_sha256']=old_code
        old_index=self.plan/'original_index.json';atomic_json(old_index,original)
        nonce='a'*32;directory=self.plan/'launches'/nonce;directory.mkdir(exist_ok=True)
        log=self.plan/'logs/accepted_A.log';log.write_text('accepted A')
        exit_path=Path(str(log)+'.exitcode');exit_path.write_text('0')
        proof_path=self.plan/'A_proof.json'
        atomic_json(proof_path,dict(status='passed',model_forwards=0,frozen_inputs=index['frozen_inputs']))
        outcome=dict(status='a_extracted',exit_code=0,proof=str(proof_path),proof_sha256=manager.sha(proof_path))
        atomic_json(directory/'outcome.json',outcome)
        argv=['python',str(implementation/'host_entry.py'),'--task-json',row['task_path']]
        launch=dict(project_id=manager.PROJECT,run_id=row['run_id'],run_nonce=nonce,pid=7,starttime=9,
                    boot_id='original-boot',argv=argv,log_path=str(log),
                    cmdline_sha256=manager.hashlib.sha256(('\0'.join(argv)+'\0').encode()).hexdigest())
        atomic_json(directory/'launch.json',launch)
        attempt=dict(run_id=row['run_id'],task_path=row['task_path'],task_sha256=row['task_sha256'],
                     nonce=nonce,exit_code=0,outcome=outcome,log_path=str(log),launch_directory=str(directory))
        state_record=dict(proof=manager.binding(proof_path),attempt=attempt)
        spec=dict(original_index=manager.binding(old_index),task=dict(path=row['task_path'],sha256=row['task_sha256']),
                  proof=manager.binding(proof_path),outcome=manager.binding(directory/'outcome.json'),
                  exit_code=manager.binding(exit_path),launch_record=manager.binding(directory/'launch.json'),state_record=state_record)
        return index,row,task,spec

    def test_preserved_A_requires_exact_actual_proof_and_unchanged_runtime_modules(self):
        index,row,task,spec=self.preserved_A_fixture()
        self.assertTrue(manager.validate_preserved_A(index,row,task,spec))
        changed=copy.deepcopy(index);changed['implementation_sha256']['extract_A.py']='c'*64
        with self.assertRaises(ValueError):manager.validate_preserved_A(changed,row,task,spec)
        with self.assertRaises(ValueError):manager.validate_preserved_A(index,row,{**task,'action':'run_pair'},spec)
        Path(spec['exit_code']['path']).write_text('1');spec['exit_code']=manager.binding(spec['exit_code']['path'])
        with self.assertRaises(ValueError):manager.validate_preserved_A(index,row,task,spec)

    def test_revision_partial_delivery_keeps_old_manifest_and_new_revision_separate(self):
        state=manager.new_state(self.index_path);state['status']='blocked'
        original=manager.partial_delivery(self.index,state);original_bytes=original.read_bytes()
        revised=copy.deepcopy(self.index);revised['technical_revision']='import_bootstrap_20261025'
        state['index']['sha256']='c'*64
        fresh=manager.partial_delivery(revised,state)
        self.assertNotEqual(original,fresh)
        self.assertIn('c'*12,fresh.name)
        self.assertEqual(original.read_bytes(),original_bytes)
        self.assertEqual(state['partial_delivery'],manager.binding(fresh))


if __name__=='__main__':unittest.main(verbosity=2)
