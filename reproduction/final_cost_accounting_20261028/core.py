"""Pure external-cost inventory/once-only accounting rules; no filesystem IO."""
import hashlib,json,math
PLAN='/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025'
MIGRATION='sysu_20261027'
SCOPES={'source_transfers','target_receivers','migration_setup_and_preparation','manager_publication_and_handoff_control','source_code_archival','owned_cleanup_and_fence'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def number(value,label,*,positive=False):
 require(type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0),label+' must be a known finite nonnegative amount, not None/bool/NaN')
 return float(value)
def is_binding(item):return isinstance(item,dict) and set(item)=={'path','sha256'} and isinstance(item['path'],str) and item['path'].startswith('/') and isinstance(item['sha256'],str) and len(item['sha256'])==64 and all(c in '0123456789abcdef' for c in item['sha256'])
def process_key(item):
 identity=item.get('identity')
 if not identity:return None
 require(isinstance(identity,dict),'Process identity must be metadata')
 for key in ('boot_id','cmdline_sha256','run_nonce'):require(isinstance(identity.get(key),str) and identity[key], 'Missing exact process '+key)
 for key in ('pid','starttime'):require(type(identity.get(key)) is int and identity[key]>0,'Missing exact process '+key)
 require(len(identity['cmdline_sha256'])==64 and all(c in '0123456789abcdef' for c in identity['cmdline_sha256']) and len(identity['run_nonce'])>=16,'Malformed exact command hash or nonce')
 return (item['host'],identity['boot_id'],identity['pid'],identity['starttime'],identity['cmdline_sha256'],identity['run_nonce'])

def inventory(manifest,closed_attempt_ids=()):
 require(manifest.get('schema')=='spnv2-final-external-cost-inventory-v1' and manifest.get('migration_id')==MIGRATION and manifest.get('plan_path')==PLAN,'Wrong external-cost inventory identity')
 require(manifest.get('all_external_work_closed') is True and manifest.get('pending_cost_items')==[],'External work/cost evidence is still pending')
 closure=manifest.get('transfer_closure',{})
 require(set(closure)=={'source_sender_status','source_sender_exit','target_receiver_status'} and all(is_binding(x) for x in closure.values()),'Bound actual source/target transfer closure is required')
 items=manifest.get('items');require(isinstance(items,list) and items,'Explicit cost inventory required')
 seen=set();coverage=set();scopes=set();new_items=[];stable=[];processes={};closed=set(closed_attempt_ids)
 for item in items:
  require(isinstance(item,dict),'Malformed cost item');key=item.get('item_id');cover=item.get('coverage_key')
  require(isinstance(key,str) and key and key not in seen and isinstance(cover,str) and cover and cover not in coverage,'Duplicate/missing stable item or coverage key');seen.add(key);coverage.add(cover)
  require(isinstance(item.get('host'),str) and item['host'],'Each item needs its real execution host')
  ss=item.get('covered_scopes');require(isinstance(ss,list) and ss and set(ss)<=SCOPES,'Explicit fee scope classification required');scopes.update(ss)
  require(item.get('kind') in ('process','control_scope') and item.get('closed') is True,'Only genuinely closed work may be imported')
  evidence=item.get('evidence');require(isinstance(evidence,list) and evidence and all(is_binding(x) for x in evidence),'Immutable real evidence bindings required')
  account=item.get('accounting');require(account in ('already_in_original_ledger','new_external_upper'),'Unknown accounting treatment')
  identity=process_key(item)
  if item['kind']=='process':
   require(identity is not None and is_binding(item.get('terminal_exit')) and type(item.get('actual_exit_code')) is int,'Actual process identity and terminal exit required')
  if account=='already_in_original_ledger':
   require(item.get('existing_attempt_id') in closed,'The claimed existing fee attempt is not actually closed in the original ledger');upper=None
  else:
   upper=number(item.get('conservative_debit_seconds_upper'),'External fee upper',positive=True)
   require(isinstance(item.get('upper_basis'),str) and item['upper_basis'],'A new upper needs an explicit evidence/formula basis')
   measured=item.get('measured_elapsed_wall_seconds')
   if measured is not None:number(measured,'Measured elapsed wall')
   cpu=item.get('measured_cpu_seconds')
   if cpu is None:require(item.get('cpu_measurement_status')=='unavailable','Unmeasured CPU must explicitly remain unavailable, not0')
   else:number(cpu,'Measured CPU')
   if identity is not None:
    interval=item.get('monotonic_interval')
    if interval is not None:
     require(isinstance(interval,list) and len(interval)==2,'Malformed per-host monotonic interval');number(interval[0],'Interval start');number(interval[1],'Interval end');require(interval[1]>=interval[0],'Interval reverses time')
    for previous in processes.get(identity,[]):
     require(interval is not None and previous is not None and (interval[1]<=previous[0] or previous[1]<=interval[0]),'Same owned process fee scopes overlap or cannot be separated')
    processes.setdefault(identity,[]).append(interval)
   new_items.append(item)
  stable.append(dict(item_id=key,coverage_key=cover,host=item['host'],kind=item['kind'],covered_scopes=sorted(ss),accounting=account,existing_attempt_id=item.get('existing_attempt_id'),identity=item.get('identity'),monotonic_interval=item.get('monotonic_interval'),conservative_debit_seconds_upper=upper,evidence_sha256=sorted(x['sha256'] for x in evidence),terminal_exit_sha256=(item.get('terminal_exit') or {}).get('sha256'),actual_exit_code=item.get('actual_exit_code')))
 for i,new in enumerate(stable):
  if new['accounting']=='new_external_upper':
   for j,old in enumerate(stable):
    if i!=j:require(not semantic_duplicate(new,old),'The same external work/control evidence is duplicated inside this inventory under different labels')
 require(scopes==SCOPES,'At least one required migration/control fee scope is unaccounted')
 view=dict(schema=manifest['schema'],migration_id=MIGRATION,plan_path=PLAN,items=sorted(stable,key=lambda x:x['item_id']),transfer_closure_sha256={k:v['sha256'] for k,v in sorted(closure.items())})
 return dict(settlement_id=digest(view),stable_view=view,new_items=new_items,new_coverage_keys=sorted(x['coverage_key'] for x in new_items),external_debit_seconds_upper=math.fsum(float(x['conservative_debit_seconds_upper']) for x in new_items),measurement_note='Per-process wall/CPU and additive conservative debit are different; overlapping host wall intervals are not experiment elapsed time. No exact global time claim.')

def _disjoint_intervals(a,b):
 if not(isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==2):return False
 for value in a+b:number(value,'Explicit nonoverlap interval')
 require(a[1]>=a[0] and b[1]>=b[0],'Interval reverses time')
 return a[1]<=b[0] or b[1]<=a[0]

def semantic_duplicate(new,old):
 """Labels/host aliases are not proof that previously paid work is new."""
 nk,ok=process_key(new),process_key(old)
 if nk is not None and ok is not None and nk[1:]==ok[1:]:
  return not _disjoint_intervals(new.get('monotonic_interval'),old.get('monotonic_interval'))
 if new.get('kind')=='control_scope' and old.get('kind')=='control_scope':
  same_evidence=set(new.get('evidence_sha256',[]))==set(old.get('evidence_sha256',[]))
  overlap_scopes=bool(set(new.get('covered_scopes',[]))&set(old.get('covered_scopes',[])))
  return same_evidence and overlap_scopes
 return False

def find_import(rows,info):
 imports=[];previous=[]
 for row in rows:
  if row['kind']!='attempt_closed':continue
  p=row['payload'];sid=p.get('external_cost_settlement_id')
  if not sid:continue
  view=p.get('external_cost_inventory');require(isinstance(view,dict) and isinstance(view.get('items'),list),'Prior external import lacks its exact persisted cost inventory')
  if sid==info['settlement_id']:
   require(view==info['stable_view'],'Same settlement ID has different semantic content')
   imports.append(row)
  else:previous.append((p,view))
 require(len(imports)<=1,'Settlement ID was imported more than once; preserve and investigate')
 new_items=[x for x in info['stable_view']['items'] if x.get('accounting')=='new_external_upper']
 for payload,view in previous:
  require(not(set(payload.get('external_cost_coverage_keys',[]))&set(info['new_coverage_keys'])),'A cost scope was already imported under another settlement ID')
  for new in new_items:
   for old in view['items']:require(not semantic_duplicate(new,old),'External work was already paid: relabelled exact process/overlapping interval or same control evidence+scope')
 return imports[0] if imports else None

def close_payload(attempt_id,elapsed_wall,pre_budget_wall,external_upper,*,info=None,manifest_binding=None,finalization_upper=2.):
 elapsed=number(elapsed_wall,'Current measured attempt wall');pre=number(pre_budget_wall,'Current measured pre-budget wall');ext=number(external_upper,'Explicit retrospective upper');tail=number(finalization_upper,'Publication tail upper',positive=True)
 p=dict(attempt_id=attempt_id,status='external_cost_import_complete' if info is not None else 'zero_forward_cost_metadata_complete',measured_seconds_before_finalization=math.fsum([elapsed,pre]),measured_current_admitted_wall_seconds=elapsed,measured_pre_budget_wall_seconds=pre,measured_seconds_scope='This native metadata main before+after admission; not CPU or cross-host elapsed time',retrospective_external_debit_seconds_upper=ext,finalization_reserved_upper_seconds=tail,charged_seconds_upper=math.fsum([elapsed,pre,ext,tail]),unresolved_forward_tokens=[],exact_total_time_claim=False,cpu_seconds=None,cpu_measurement_status='unavailable',no_new_model_forwards=True)
 if info is not None:
  require(is_binding(manifest_binding),'A real imported manifest binding is required')
  p.update(external_cost_settlement_id=info['settlement_id'],external_cost_manifest=manifest_binding,external_cost_inventory=info['stable_view'],external_cost_coverage_keys=info['new_coverage_keys'])
 return p

def safe_snapshot(snapshot):
 require(snapshot.get('schema')=='g50p-diagnostic-budget-snapshot-v1' and snapshot.get('time_hard_stop_enabled') is False,'Current unlimited-time original budget policy required')
 require(type(snapshot.get('physical_forwards_reserved_upper')) is int and 0<=snapshot['physical_forwards_reserved_upper']<=40000,'Physical-forward cap violated')
 require(snapshot.get('physical_forwards_unknown_upper')==0 and not snapshot.get('unknown_forward_tokens') and not snapshot.get('unresolved_attempt_ids'),'Unresolved real allocations/forwards prevent publication')
 return snapshot

def summary_scope(summary,proof,provisional,accepted_count):
 require(summary.get('schema')=='g50p-risk-diagnostic-summary-v1' and proof.get('schema')=='g50p-risk-diagnostic-summary-proof-v1' and proof.get('status')=='passed','Saved scientific summary/proof required')
 n=summary.get('completed_windows');require(type(n) is int and 0<=n<=48 and n==proof.get('completed_windows')==provisional.get('independently_accepted_windows')==accepted_count,'Scientific scope counts disagree')
 require(proof.get('all_completed_pairs_independently_accepted') is True and proof.get('model_forwards_added')==0 and summary.get('maximum_windows')==48 and summary.get('new_complete_ODR')==0,'Saved independent summary scope differs')
 require(proof.get('complete_scope') is (n==48) and provisional.get('scientific_scope_complete') is (n==48),'Complete/partial scientific claim differs')
 return n
