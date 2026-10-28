"""In-memory contract tests only. No project runtime, ledger, files or models."""
import copy

def run(c,e=None):
 count=0
 def expect_error(fn):
  try:fn()
  except ValueError:return
  raise AssertionError('Expected fail-closed rejection')
 binding=lambda label:dict(path=c.PLAN+'/revisions/final_cost_accounting_20261028/'+label+'.json',sha256='a'*64)
 item=dict(item_id='closed_control_scope',coverage_key='unpaid_control_epoch',host='sysu_target',kind='control_scope',covered_scopes=sorted(c.SCOPES),closed=True,evidence=[binding('evidence')],accounting='new_external_upper',conservative_debit_seconds_upper=10.,upper_basis='Synthetic evidence-supported upper for this pure fixture only',measured_elapsed_wall_seconds=4.,measured_cpu_seconds=None,cpu_measurement_status='unavailable')
 manifest=dict(schema='spnv2-final-external-cost-inventory-v1',migration_id=c.MIGRATION,plan_path=c.PLAN,all_external_work_closed=True,pending_cost_items=[],transfer_closure={k:binding(k) for k in ('source_sender_status','source_sender_exit','target_receiver_status')},items=[item])
 info=c.inventory(manifest);assert info['external_debit_seconds_upper']==10. and info['new_items'][0]['measured_cpu_seconds'] is None;count+=1
 renamed=copy.deepcopy(manifest);renamed['created_at']='later';renamed['items'][0]['evidence'][0]['path']=c.PLAN+'/revisions/final_cost_accounting_20261028/relocated.json';assert c.inventory(renamed)['settlement_id']==info['settlement_id'];count+=1
 unknown=copy.deepcopy(manifest);unknown['items'][0]['conservative_debit_seconds_upper']=None;expect_error(lambda:c.inventory(unknown));count+=1
 duplicate=copy.deepcopy(manifest);duplicate['items'].append(copy.deepcopy(item));expect_error(lambda:c.inventory(duplicate));count+=1
 unclosed=copy.deepcopy(manifest);unclosed['pending_cost_items']=['target_receive_unclosed'];expect_error(lambda:c.inventory(unclosed));count+=1
 wrong=copy.deepcopy(manifest);wrong['items'][0]['cpu_measurement_status']='measured';expect_error(lambda:c.inventory(wrong));count+=1
 payload=c.close_payload('real_fixture_attempt',2.,.25,10.,info=info,manifest_binding=binding('manifest'));assert payload['charged_seconds_upper']==14.25 and payload['measured_seconds_before_finalization']==2.25 and payload['measured_current_admitted_wall_seconds']==2. and payload['retrospective_external_debit_seconds_upper']==10. and payload['cpu_seconds'] is None;rows=[dict(sequence=1,kind='attempt_closed',payload=payload,sha256='b'*64)];assert c.find_import(rows,info)==rows[0];assert sum(r['payload']['charged_seconds_upper'] for r in rows if r['kind']=='attempt_closed')==14.25;count+=1
 duplicate_import=rows+[dict(sequence=2,kind='attempt_closed',payload=copy.deepcopy(payload),sha256='c'*64)];expect_error(lambda:c.find_import(duplicate_import,info));count+=1
 recheck=c.close_payload('new_real_metadata_recheck',.1,.1,0.);rechecked=rows+[dict(sequence=2,kind='attempt_closed',payload=recheck,sha256='c'*64)];assert c.find_import(rechecked,info)==rows[0] and sum(r['payload'].get('retrospective_external_debit_seconds_upper',0.) for r in rechecked)==10.;count+=1
 another=copy.deepcopy(info);another['settlement_id']='c'*64;expect_error(lambda:c.find_import(rows,another));count+=1
 paid=copy.deepcopy(manifest);paid['items'][0].update(accounting='already_in_original_ledger',existing_attempt_id='paid_attempt');assert c.inventory(paid,['paid_attempt'])['external_debit_seconds_upper']==0.;expect_error(lambda:c.inventory(paid,[]));count+=1
 snapshot=dict(schema='g50p-diagnostic-budget-snapshot-v1',time_hard_stop_enabled=False,physical_forwards_reserved_upper=40000,physical_forwards_unknown_upper=0,unknown_forward_tokens=[],unresolved_attempt_ids=[]);c.safe_snapshot(snapshot);over=copy.deepcopy(snapshot);over['physical_forwards_reserved_upper']=40001;expect_error(lambda:c.safe_snapshot(over));count+=1
 summary=dict(schema='g50p-risk-diagnostic-summary-v1',completed_windows=48,maximum_windows=48,new_complete_ODR=0);proof=dict(schema='g50p-risk-diagnostic-summary-proof-v1',status='passed',completed_windows=48,all_completed_pairs_independently_accepted=True,model_forwards_added=0,complete_scope=True);provisional=dict(independently_accepted_windows=48,scientific_scope_complete=True);assert c.summary_scope(summary,proof,provisional,48)==48;expect_error(lambda:c.summary_scope(summary,proof,provisional,47));count+=1
 names=['known_inventory_preserves_unavailable_CPU','stable_id_ignores_creation_and_relocation','unknown_fee_rejected','duplicate_item_or_coverage_rejected','pending_external_work_rejected','unknown_CPU_not_faked_measured','recognized_close_contains_single_external_debit','duplicate_same_id_import_rejected','real_recheck_does_not_repeat_external_debit','same_coverage_different_id_rejected','already_paid_requires_closed_original_attempt','physical_forward_cap_preserved','saved_summary_scope_consistent']
 relabel=copy.deepcopy(manifest);relabel['items'][0].update(item_id='renamed_item',coverage_key='renamed_coverage');relabelled=c.inventory(relabel);expect_error(lambda:c.find_import(rows,relabelled));count+=1;names.append('control_relabel_same_evidence_scope_rejected')
 proc=copy.deepcopy(manifest);proc['items'][0].update(kind='process',identity=dict(pid=1,starttime=10,boot_id='real_fixture_boot',cmdline_sha256='d'*64,run_nonce='1'*32),terminal_exit=binding('terminal_exit'),actual_exit_code=0,monotonic_interval=[1.,2.]);pi=c.inventory(proc);pe=c.close_payload('process_import',1.,.1,10.,info=pi,manifest_binding=binding('process_manifest'));pr=[dict(sequence=1,kind='attempt_closed',payload=pe,sha256='f'*64)]
 renamed=copy.deepcopy(proc);renamed['items'][0].update(item_id='renamed_process',coverage_key='renamed_process_coverage',host='same_host_alias');ri=c.inventory(renamed);expect_error(lambda:c.find_import(pr,ri));count+=1;names.append('exact_process_relabel_and_host_alias_rejected')
 renamed['items'][0]['monotonic_interval']=[2.,3.];disjoint=c.inventory(renamed);assert c.find_import(pr,disjoint) is None;count+=1;names.append('explicit_nonoverlapping_same_process_scope_allowed')
 local=copy.deepcopy(manifest);second=copy.deepcopy(local['items'][0]);second.update(item_id='other_label',coverage_key='other_label');local['items'].append(second);expect_error(lambda:c.inventory(local));count+=1;names.append('within_inventory_relabel_duplicate_rejected')
 if e is None:raise AssertionError('Entry module required for exact candidate failure-path tests')
 original_execution=e.EXECUTION
 observation=dict(started_monotonic=10.,started_wall_time=100.,nonce='2'*32,request_path='request.json',request_sha256='e'*64,log_path='actual.log',argv=['entry.py','--execute'])
 try:
  for stage in ('resource_admission','registry_readback','budget_constructor','source_prefix_validation'):
   e.EXECUTION=dict(stage=stage,accounting={});captured=[]
   def broken():raise ValueError(stage+' actual failure fixture')
   def capture(error,obs):captured.append(e.failure_payload(error,obs,e.EXECUTION,12.,102.))
   expect_error(lambda:e.guarded_action(broken,observation,capture));assert len(captured)==1;failure=captured[0];assert failure['accounting_pending'] is True and failure['measured_total_main_wall_seconds_before_failure_receipt']==2. and failure['measured_unadmitted_wall_seconds']==2. and failure['attempt_id'] is None and not failure['ledger_attempt_closed'] and failure['cpu_seconds'] is None;count+=1;names.append('outer_receipt_before_begin_'+stage)
  unclosed=dict(stage='attempt_start_requested',attempt_start_requested=True,accounting=dict(attempt_id='unclosed_actual_allocation',closed=False,started_monotonic=11.));f=e.failure_payload(ValueError('begin return lost'),observation,unclosed,12.,102.);assert f['accounting_pending'] and f['existing_ledger_attempt_must_not_be_reimported'] and not f['ledger_attempt_closed'] and f['measured_unadmitted_wall_seconds']==1.;count+=1;names.append('unclosed_start_is_pending_not_reimported')
  closed=dict(stage='metadata_action_closed',accounting=dict(attempt_id='already_closed',closed=True,charged_seconds_upper=7.,external_debit_seconds_upper=3.,started_monotonic=11.));f=e.failure_payload(ValueError('postclose fixture'),observation,closed,12.,102.);assert not f['accounting_pending'] and f['existing_ledger_attempt_must_not_be_reimported'] and f['ledger_attempt_closed'];count+=1;names.append('ledgered_failure_is_not_charged_again')
  closed['accounting']['charged_seconds_upper']=4.;f=e.failure_payload(ValueError('outside_paid_upper'),observation,closed,12.,102.);assert f['accounting_pending'] and f['uncovered_wall_lower_bound']==1.;count+=1;names.append('uncovered_failure_wall_stays_pending')
  writes=[]
  def store(path,value):writes.append((str(path),copy.deepcopy(value)));return dict(path=str(path),sha256='b'*64)
  def tail_failure(ctx,result):raise ValueError('actual tail failure fixture after candidate write')
  fields=dict(status='complete',final_settled_budget=binding('fresh_budget'),active_seconds_charged_upper=20.,final_cost_publication_proof=binding('candidate_proof'))
  for nonce in ('3'*32,'4'*32):
   ctx=dict(directory=e.P/'launches'/nonce);expect_error(lambda:e.store_candidate_and_finish(ctx,fields,dict(model_forwards=0),store=store,complete=tail_failure))
  assert len(writes)==2 and all(path.endswith('/candidate_final_manifest.json') and path!=str(e.P/'delivery/final_manifest.json') for path,_ in writes) and writes[0][0]!=writes[1][0]
  assert all(v['candidate_only'] and v['promotion_costs_pending'] and v['final_manifest_not_published'] and 'final_settled_budget' not in v for _,v in writes);count+=1;names.append('tail_failure_preserves_retryable_launch_candidate_only')
 finally:e.EXECUTION=original_execution
 assert len(names)==count
 print(__import__('json').dumps(dict(status='pure_contract_tests_passed',cases=count,test_names=names,model_forwards=0,no_runtime_file_or_budget_IO=True,no_settlement_or_publication_executed=True)))
 return 0
