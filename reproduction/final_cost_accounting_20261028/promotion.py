"""Candidate two-step paid promotion; no actual action without --execute.

fund: original admitted zero-forward attempt pays its real metadata work plus
one fixed60s post-exit installation upper (not measured time), and binds an
immutable installer request/code/nonce. install: original admission/locks and
registration, but NO new budget attempt; consume that already-paid allowance
once, only after real fund/settlement/publisher wrapper0 and exact ledger head.

Request schema: spnv2-paid-final-promotion-request-v1, action=fund|install,
promotion/entry/core/index exact bindings. fund additionally binds
installer_request. The install request contains reservation_id (32hex),
install_nonce (32hex), reservation_path (fixed path from reservation_path()),
candidate_manifest, publisher_nonce and settlement_success_nonce. Root must
precreate that install request; it references the reservation PATH, not its
future hash, avoiding a circular binding. All bindings use the approved namespace.

All outputs are under delivery/paid_final_installation or launches, NOT the
ordinary namespace's read-only revisions overlay. Previous accepted final files
are never replaced. Caught publication failures roll back only this invocation's
new same-inode unaccepted hardlink. A SIGKILL after atomic publication cannot
self-clean: root must verify actual installer wrapper0 AND real elapsed bound
before accepting; otherwise keep pending and quarantine only its proven own
unaccepted link. A retry always needs a new paid reservation; no refunds.

Overrun/unknown duration remains accounting_pending. Any old uncovered failure
needs a bound original-ledger resolution before a new successful promotion; this
module never guesses a deficit or repurposes an old unclosed allocation.
A prior real0 or absent failure receipt may use prior_failure_resolutions[rid]
with schema spnv2-paid-installation-terminal-disposition-v1. It must explicitly
prove NOT accepted, preserve real exit bytes/absence, independently observe exact
termination, bind a supported total wall upper, isolate only the own unaccepted
inode when created, and reference a genuine original-ledger deficit resolution.
A missing measured wall is allowed only with a real finite conservative bound;
an unknown bound cannot become0. No old exit or failure file is changed.
"""
import time
ENTRY_STARTED=time.monotonic();ENTRY_WALL=time.time()
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,math,os,sys
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
COSTROOT=P/'delivery/paid_final_installation'
FINAL=P/'delivery/final_manifest.json'
ALLOWANCE=60.;TAIL=2.
ENTRY_SHA='7b3bc8f1c58e99a96cb112f9b5ece3736aa5e9f07b5ca38fd02292acc325444b'
CORE_SHA='d3a0fbfaad5784b51df4d82acc6b1f12b059732bd18514c17bf284378cb1ce3b'
e=None;c=None;STATE={}

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hexnonce(value):return isinstance(value,str) and len(value)==32 and all(x in '0123456789abcdef' for x in value)
def reservation_path(rid):
 require(hexnonce(rid),'Reservation ID must be a fresh32hex identifier');return COSTROOT/'reservations'/(rid+'.json')
def lease_path(rid):return reservation_path(rid).with_suffix('.attempted.json')
def load_helpers(request=None):
 global e,c
 root=Path(__file__).resolve().parent
 if request is not None:
  for name,expected in (('entry',ENTRY_SHA),('core',CORE_SHA)):
   spec=request.get(name,{});require(Path(spec.get('path','')).resolve()==root/(name+'.py') and spec.get('sha256')==expected,'Only the reviewed published entry/core may be reused')
 require(sha(root/'entry.py')==ENTRY_SHA and sha(root/'core.py')==CORE_SHA,'Published entry/core bytes changed; re-review instead of editing them')
 spec=importlib.util.spec_from_file_location('g50p_paid_promotion_published_entry',root/'entry.py');e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e);c=e.load_core(dict(path=str(root/'core.py'),sha256=CORE_SHA));return e,c

def own_identity():
 proc=Path('/proc')/str(os.getpid());stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=os.getpid(),starttime=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest(),run_nonce=os.environ.get('SPNV2_RUN_NONCE'))
def measurements():
 identity=own_identity();hz=float(os.sysconf('SC_CLK_TCK'));require(hasattr(time,'CLOCK_BOOTTIME'),'Whole-process allowance measurement unavailable')
 elapsed_upper=time.clock_gettime(time.CLOCK_BOOTTIME)-identity['starttime']/hz+1./hz;require(elapsed_upper>=0,'Kernel elapsed clock inconsistent')
 cpu=None
 try:
  import resource
  a,b=resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN);cpu=dict(self_user_seconds=a.ru_utime,self_system_seconds=a.ru_stime,waited_children_user_seconds=b.ru_utime,waited_children_system_seconds=b.ru_stime)
 except (ImportError,OSError):pass
 return dict(measured_entry_wall_seconds=time.monotonic()-ENTRY_STARTED,whole_process_elapsed_upper_seconds=elapsed_upper,elapsed_upper_basis='Linux proc starttime ticks and CLOCK_BOOTTIME, with one tick margin',cpu=cpu,cpu_measurement_status='available_for_self_and_waited_children' if cpu is not None else 'unavailable',identity=identity,exact_total_time_claim=False)
def check_time(measurement,*,before_publication=False):
 value=measurement.get('whole_process_elapsed_upper_seconds');require(type(value) in (int,float) and value>=0,'Unknown installer duration cannot be treated as0')
 require(value<=ALLOWANCE-(TAIL if before_publication else 0.),'Paid installation allowance exceeded; keep pending, never refund or claim complete')

def validate_authority(plan,request_binding,nonce,ledger_head,*,already_attempted=False):
 require(not already_attempted,'Reservation already attempted/consumed; retry requires a NEW paid reservation')
 require(hexnonce(nonce) and nonce==plan.get('install_nonce'),'Installer nonce is not the one named in the paid reservation')
 require(plan.get('installer_request')==request_binding and plan.get('allowance_seconds_upper')==ALLOWANCE,'Installer request or fixed paid allowance differs')
 require(ledger_head==plan.get('fund_close_sha256'),'Another original-ledger writer ran after funding; do not publish a stale final budget')
 return True

def closed_execution(budget,nonce,status):
 result,lb,ob=e.successful_attempt(nonce,status);launch=e.checked(lb);starts=[x for x in budget.rows if x['kind']=='attempt_started' and x['payload']['launch']['run_nonce']==nonce]
 require(len(starts)==1,'Real prior process has no unique ledger allocation');start=starts[0];aid=start['payload']['attempt_id'];closes=[x for x in budget.rows if x['kind']=='attempt_closed' and x['payload']['attempt_id']==aid]
 require(len(closes)==1 and not start['payload']['model_forwards_allowed'] and not closes[0]['payload']['unresolved_forward_tokens'],'Prior metadata process did not actually close without model work')
 require(all(launch[k]==start['payload']['launch'][k] for k in ('pid','starttime','boot_id','cmdline_sha256','run_nonce','run_id')),'Actual prior process identity differs from original ledger')
 return dict(result=result,launch=lb,outcome=ob,start=start,close=closes[0])

def source_ready(ctx,install):
 publisher=closed_execution(ctx['budget'],install['publisher_nonce'],'final_cost_candidate_ready_pending_wrapper_exit');candidate=e.checked(install['candidate_manifest']);require(publisher['result']['candidate_final_manifest']==install['candidate_manifest'],'Real publisher produced another candidate')
 require(candidate.get('schema')=='g50p-final-cost-publication-candidate-v1' and candidate.get('candidate_only') is True and candidate.get('promotion_costs_pending') is True,'A genuine unpromoted cost candidate is required')
 require(publisher['result']['fresh_budget']==candidate['candidate_stage_closed_budget'],'Candidate closed-budget binding differs');c.safe_snapshot(e.checked(candidate['candidate_stage_closed_budget']));proof=e.checked(candidate['candidate_stage_proof']);require(proof['own_publication_close_event']==dict(sequence=publisher['close']['sequence'],sha256=publisher['close']['sha256']),'Candidate proof does not bind the actual publisher close')
 settlement=closed_execution(ctx['budget'],install['settlement_success_nonce'],'external_cost_settlement_verified');receipt=e.checked(candidate['external_cost_settlement']);require(settlement['result']['settlement']==candidate['external_cost_settlement'],'Real settlement produced another receipt')
 manifest=e.checked(receipt['external_manifest']);closed={x['payload']['attempt_id'] for x in ctx['budget'].rows if x['kind']=='attempt_closed'};info=c.inventory(manifest,closed);imported=c.find_import(ctx['budget'].rows,info)
 require(receipt.get('status')=='settled' and receipt.get('all_external_work_closed') is True and receipt.get('pending_cost_items')==[] and imported is not None and imported['sha256']==receipt['import_event_sha256'],'Original external costs are not genuinely settled')
 summary=e.checked(candidate['summary']);sp=e.checked(candidate['proof']);require(sp['summary_sha256']==candidate['summary']['sha256'],'Original scientific summary/proof hash differs')
 require(summary['completed_windows']==sp['completed_windows']==len(ctx['state']['accepted']) and sp.get('status')=='passed' and sp.get('all_completed_pairs_independently_accepted') is True,'Accepted scientific scope differs; never rerun summary')
 return dict(candidate=candidate,publisher=publisher,settlement=settlement)

def validate_terminal_disposition(doc,plan,actual_exit,lease_identity,terminal,*,canonical_absent):
 """Root disposition may preserve genuine0 or genuinely unavailable exit."""
 require(doc.get('schema')=='spnv2-paid-installation-terminal-disposition-v1' and doc.get('status')=='not_accepted_resolved','Explicit root terminal disposition required')
 require(doc.get('reservation_id')==plan['reservation_id'] and doc.get('install_nonce')==plan['install_nonce'],'Disposition belongs to another reserved installer')
 require(doc.get('not_accepted') is True and doc.get('old_final_never_accepted') is True and doc.get('original_exit_evidence_preserved') is True and canonical_absent,'Never replace/relax an accepted or still-present final')
 require(doc.get('actual_exit_code')==actual_exit and terminal.get('actual_exit_code')==actual_exit,'Keep the real old exit exactly; do not turn0 into1')
 require(terminal.get('reservation_id')==plan['reservation_id'] and terminal.get('install_nonce')==plan['install_nonce'] and terminal.get('exact_identity_terminated') is True,'Exact old installer termination must be independently observed')
 observed=terminal.get('identity',{})
 require(all(observed.get(k)==lease_identity.get(k) for k in ('pid','starttime','boot_id','cmdline_sha256','run_nonce')),'Disposition does not prove termination of the exact old owned process')
 require(terminal.get('terminal_basis') in ('supervisor_reaped','not_present','different_starttime','host_boot_changed'),'SSH failure or unknown PID alone is not terminal evidence')
 if actual_exit is None:require(terminal.get('exit_evidence_unavailable') is True,'Missing old exit is unknown, never fabricated')
 upper=terminal.get('all_installation_disposition_wall_seconds_upper')
 require(type(upper) in (int,float) and math.isfinite(upper) and upper>0 and isinstance(terminal.get('upper_basis'),str) and terminal['upper_basis'],'Measured/unknown wall needs a real supported finite total upper, never a default0')
 require(type(doc.get('final_link_was_created')) is bool,'Explicit link disposition is required')
 if actual_exit==0:require(doc['final_link_was_created'] is True,'A successful approved installer needs its own unaccepted final-link quarantine evidence')
 return upper

def verify_root_disposition(ctx,row,plan,doc,actual_exit):
 lease=json.loads(lease_path(plan['reservation_id']).read_text());terminal=e.checked(doc['terminal_observation'])
 upper=validate_terminal_disposition(doc,plan,actual_exit,lease['identity'],terminal,canonical_absent=not os.path.lexists(FINAL))
 if actual_exit is not None:
  path=Path(plan['install_log_path']+'.exitcode');require(doc.get('original_exit_file_sha256')==e.sha(path),'Original real exit bytes changed')
 else:require(doc.get('original_exit_file_sha256') is None,'Do not claim a hash for absent exit evidence')
 if doc['final_link_was_created']:
  candidate=P/'launches'/plan['install_nonce']/'final_manifest_candidate.json';quarantine=candidate.with_name('unaccepted_final_manifest_quarantined.json')
  require(doc.get('own_candidate',{}).get('path')==str(candidate) and doc.get('quarantined_own_link',{}).get('path')==str(quarantine),'Only this old installer own link may be isolated')
  e.checked(doc['own_candidate']);e.checked(doc['quarantined_own_link']);require(not candidate.is_symlink() and not quarantine.is_symlink() and os.path.samefile(candidate,quarantine),'Quarantine is not the exact own unaccepted candidate inode')
 funded,_,_=e.successful_attempt(plan['fund_nonce'],'post_exit_installation_funded');require(funded['reservation_id']==plan['reservation_id'],'Earlier allowance lacks real fund wrapper0')
 ref=doc.get('original_ledger_resolution',{});matches=[x for x in ctx['budget'].rows if x['kind']=='attempt_closed' and x['sequence']==ref.get('sequence') and x['sha256']==ref.get('sha256')];require(len(matches)==1,'Deficit/coverage resolution is not a recognized original-ledger close')
 paid=matches[0]
 if upper<=ALLOWANCE and doc.get('deficit_resolved_as')=='covered_by_original_prepaid_allowance':require(paid['sha256']==row['sha256'],'The claimed old paid allowance is another fee event');return
 require(paid['payload'].get('external_cost_settlement_id'),'Uncovered/unknown cost cannot be waived without actual ledger resolution')
 covered=[x for x in paid['payload'].get('external_cost_inventory',{}).get('items',[]) if x.get('accounting')=='new_external_upper' and x.get('coverage_key')=='paid_installation_overrun:'+plan['reservation_id'] and doc['terminal_observation']['sha256'] in x.get('evidence_sha256',[])]
 require(len(covered)==1 and type(covered[0].get('conservative_debit_seconds_upper')) in (int,float) and covered[0]['conservative_debit_seconds_upper']>=max(0.,upper-ALLOWANCE),'Actual prior deficit is not covered by its bound original-ledger import')

def prior_failure_check(ctx,request):
 """An uncertain/overrun prior installer cannot silently disappear from costs."""
 resolutions=request.get('prior_failure_resolutions',{})
 for row in ctx['budget'].rows:
  if row['kind']!='attempt_closed' or 'post_exit_installation_reservation' not in row['payload']:continue
  plan=row['payload']['post_exit_installation_reservation'];rid=plan['reservation_id']
  if not lease_path(rid).exists():continue
  failure=P/'launches'/plan['install_nonce']/'promotion_failure_receipt.json';code_path=Path(plan['install_log_path']+'.exitcode');actual_exit=int(code_path.read_text().strip()) if code_path.is_file() else None;ref=resolutions.get(rid);proof=e.checked(ref) if ref is not None else None
  if isinstance(proof,dict) and proof.get('schema')=='spnv2-paid-installation-terminal-disposition-v1':verify_root_disposition(ctx,row,plan,proof,actual_exit);continue
  if not failure.exists():raise ValueError('Prior installer has no failure receipt; root must supply a strict not-accepted terminal disposition, never fake an old exit')
  doc=json.loads(failure.read_text());require(actual_exit is not None and actual_exit!=0,'Preserve a real0/missing exit; use strict root disposition rather than changing it to1')
  if doc.get('allowance_covered_with_final_tail') is True:continue
  require(ref is not None,'Prior installation overrun/unknown time must be settled, not0, before new funding')
  require(proof.get('schema')=='spnv2-paid-installation-failure-resolution-v1' and proof.get('status')=='settled' and proof.get('reservation_id')==rid and proof.get('failure_receipt')==e.bind(failure),'Wrong prior failure resolution')
  if proof.get('covered_by_existing_prepaid_allowance') is True:
   observed=e.checked(proof['actual_terminal_observation']);upper=observed.get('all_installation_and_receipt_wall_seconds_upper');require(observed.get('reservation_id')==rid and observed.get('install_nonce')==plan['install_nonce'] and observed.get('actual_exit_code')==int(code_path.read_text().strip()) and type(upper) in (int,float) and 0<=upper<=ALLOWANCE and proof.get('fund_close_sha256')==row['sha256'],'No real terminal bound showing the earlier failed attempt was fully covered by its already-paid allowance')
   continue
  event=proof.get('original_ledger_close',{});matches=[x for x in ctx['budget'].rows if x['kind']=='attempt_closed' and x['sequence']==event.get('sequence') and x['sha256']==event.get('sha256')]
  require(len(matches)==1 and matches[0]['payload'].get('external_cost_settlement_id'),'Prior deficit was not resolved by a recognized original-ledger settlement')

def fund_charge(elapsed,pre,process_upper,plan):
 require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (elapsed,pre,process_upper)),'Known actual fund timing required');margin=max(0.,process_upper-(elapsed+pre))
 return dict(measured_seconds_before_finalization=elapsed+pre,measured_current_admitted_wall_seconds=elapsed,measured_pre_budget_wall_seconds=pre,pre_module_clock_margin_seconds_upper=margin,post_exit_installation_conservative_upper_seconds=ALLOWANCE,finalization_reserved_upper_seconds=TAIL,charged_seconds_upper=elapsed+pre+margin+ALLOWANCE+TAIL,retrospective_external_debit_seconds_upper=0.,unresolved_forward_tokens=[],exact_total_time_claim=False,post_exit_installation_reservation_id=plan['reservation_id'],post_exit_installation_reservation=plan,post_exit_installation_plan_sha256=digest(plan))
def fund_close(ctx,plan):
 b,a=ctx['budget'],ctx['attempt'];require(not a.closed and not a.pending and a.deadline_monotonic is None,'Real zero-forward unlimited-time fund attempt required');m=measurements();elapsed=b.clock.monotonic()-a.record['started_monotonic'];pre=m['measured_entry_wall_seconds']-elapsed;require(pre>=0,'Fund timing origin inconsistent');payload=fund_charge(elapsed,pre,m['whole_process_elapsed_upper_seconds'],plan);payload.update(attempt_id=a.attempt_id,status='post_exit_installation_funded',fund_cpu_observation=m['cpu'])
 row=b._append('attempt_closed',payload);a.closed=True;b._attempt=None;ctx.update(close_event=row,tail_started=time.monotonic());ctx['accounting'].update(closed=True,charged_seconds_upper=payload['charged_seconds_upper'],external_debit_seconds_upper=0.,post_exit_installation_prepaid_upper=ALLOWANCE)
 snapshot=b.snapshot();c.safe_snapshot(snapshot);require(snapshot['physical_forwards_reserved_upper']==ctx['before']['physical_forwards_reserved_upper'] and snapshot['physical_forwards_confirmed']==ctx['before']['physical_forwards_confirmed'],'Fund changed physical forwards');require(e.sha(P/'budget/journal.jsonl',ctx['journal_before']['bytes'])==ctx['journal_before']['sha256'],'Old ledger bytes changed');return row,snapshot

def write_fund_result(ctx,result):
 from budget import atomic_json
 atomic_json(ctx['directory']/'result.json',result)
 def tail_check():
  if time.monotonic()-ctx['tail_started']>TAIL:
   m=measurements();a=ctx['attempt'];ctx['budget']._append('time_finalization_observed',dict(attempt_id=a.attempt_id,measured_total_seconds=m['measured_entry_wall_seconds'],charged_seconds_upper=m['whole_process_elapsed_upper_seconds']+ALLOWANCE+TAIL,time_hard_stop_enabled=False,exact_total_time_claim=False,post_exit_installation_prepaid_upper_seconds=ALLOWANCE));raise ValueError('Fund publication tail exceeded bound; reservation cannot be used because actual fund exit is not0')
 tail_check();atomic_json(ctx['directory']/'outcome.json',dict(status='post_exit_installation_funded',exit_code=0,result=str(ctx['directory']/'result.json'),result_sha256=e.sha(ctx['directory']/'result.json'),model_forwards=0,actual_wrapper_exit_still_required=True));tail_check();print(json.dumps(result))

def fund(request,request_binding,log_path):
 install=e.checked(request['installer_request']);require(install.get('schema')=='spnv2-paid-final-promotion-request-v1' and install.get('action')=='install','Bound static installer request required');rid=install['reservation_id'];STATE['reservation_id']=rid;path=reservation_path(rid);require(install.get('reservation_path')==str(path) and hexnonce(install.get('install_nonce')) and install['install_nonce']!=os.environ.get('SPNV2_RUN_NONCE'),'Wrong reservation path or reused installer nonce')
 require(all(install.get(k)==request.get(k) for k in ('promotion','entry','core','index')),'Fund and installer code/index bindings differ')
 with e.admitted(request,'g50p_paid_final_fund_'+rid,log_path,ENTRY_STARTED) as ctx:
  require(not FINAL.exists() and not path.exists() and not lease_path(rid).exists() and not(P/'launches'/install['install_nonce']).exists(),'Do not replace an old final/reservation/installer execution')
  require(not any(x['kind']=='attempt_closed' and x['payload'].get('post_exit_installation_reservation_id')==rid for x in ctx['budget'].rows),'Reservation already paid; do not debit it twice')
  prior_failure_check(ctx,request);ready=source_ready(ctx,install);e.begin(ctx)
  plan=dict(schema='spnv2-paid-final-installation-plan-v1',reservation_id=rid,install_nonce=install['install_nonce'],install_log_path=install['install_log_path'],fund_nonce=ctx['nonce'],fund_request=request_binding,installer_request=request['installer_request'],installer_code=request['promotion'],entry=request['entry'],core=request['core'],index=request['index'],candidate_manifest=install['candidate_manifest'],publisher_nonce=install['publisher_nonce'],settlement_success_nonce=install['settlement_success_nonce'],allowance_seconds_upper=ALLOWANCE,allowance_is_conservative_not_measured=True,no_refund=True,no_new_budget_attempt_for_install=True,source_prefix=ctx['prefix'])
  require(Path(plan['install_log_path']).is_absolute() and Path(plan['install_log_path']).resolve()==Path(plan['install_log_path']) and Path(plan['install_log_path']).resolve().is_relative_to(P),'Installer log must be its actual canonical named project wrapper log')
  event,snapshot=fund_close(ctx,plan);bb=e.immutable(ctx['directory']/'budget_closed.json',snapshot);reservation=dict(schema='spnv2-paid-final-installation-reservation-v1',plan=plan,plan_sha256=digest(plan),fund_close=dict(sequence=event['sequence'],sha256=event['sha256'],attempt_id=event['payload']['attempt_id']),fund_closed_budget=bb,source_candidate=install['candidate_manifest'],actual_fund_wrapper0_required=True)
  rb=e.immutable(path,reservation);write_fund_result(ctx,dict(status='post_exit_installation_funded',reservation=rb,reservation_id=rid,allowance_seconds_upper=ALLOWANCE,allowance_is_not_measured_time=True,model_forwards=0,promotion_code=request['promotion']))

def exclusive_json(path,value):
 data=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())

def atomic_new_link(source,target,*,link=None,sync=None,same=None,unlink=None,post_check=None):
 """Never replace a prior final. Roll back only our own unaccepted inode."""
 link=os.link if link is None else link;same=os.path.samefile if same is None else same;unlink=os.unlink if unlink is None else unlink
 if sync is None:
  def sync(path):
   fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
 created=False
 try:
  link(source,target);created=True;sync(target.parent)
  if post_check is not None:post_check()
 except BaseException:
  if created and same(source,target):unlink(target);sync(target.parent)
  raise

def install(request,request_binding,log_path):
 rid=request['reservation_id'];path=reservation_path(rid);require(request.get('reservation_path')==str(path),'Wrong fixed reservation path');reservation=json.loads(e.allowed(path).read_text());plan=reservation['plan'];require(digest(plan)==reservation['plan_sha256'],'Reservation plan changed')
 require(plan['installer_code']==request['promotion'] and plan['installer_request']==request_binding and plan['install_nonce']==os.environ.get('SPNV2_RUN_NONCE') and plan['install_log_path']==str(log_path),'Only the exact reserved code/request/nonce/log may install')
 funded,_,_=e.successful_attempt(plan['fund_nonce'],'post_exit_installation_funded');require(funded['reservation']==e.bind(path),'Actual successful fund did not produce this exact reservation')
 STATE.update(mode='install',reservation=reservation,reservation_binding=e.bind(path),lease_owned=False,fund_verified=False)
 check_time(measurements(),before_publication=True)
 lease=dict(schema='spnv2-paid-final-installation-attempt-v1',reservation_id=rid,install_nonce=plan['install_nonce'],reservation=e.bind(path),request=request_binding,identity=own_identity(),started_wall_time=ENTRY_WALL,allowance_seconds_upper=ALLOWANCE,formal_consumption_not_yet_admitted=True)
 require(not lease_path(rid).exists(),'Reservation already attempted: a retry needs a new paid reservation');exclusive_json(lease_path(rid),lease);STATE['lease_owned']=True
 with e.admitted(request,'g50p_paid_final_install_'+rid,log_path,ENTRY_STARTED) as ctx:
  require(not FINAL.exists(),'Existing final is never overwritten; verify or isolate only a proven unaccepted old link separately')
  row=[x for x in ctx['budget'].rows if x['kind']=='attempt_closed' and x['payload'].get('post_exit_installation_reservation_id')==rid];require(len(row)==1 and row[0]['sha256']==reservation['fund_close']['sha256'] and row[0]['sequence']==reservation['fund_close']['sequence'],'Reservation has no unique recognized original-ledger payment')
  require(row[0]['payload']['post_exit_installation_reservation']==plan and row[0]['payload']['post_exit_installation_plan_sha256']==digest(plan),'Paid plan differs from actual fund close event')
  authority=dict(plan,fund_close_sha256=reservation['fund_close']['sha256']);validate_authority(authority,request_binding,ctx['nonce'],ctx['budget'].rows[-1]['sha256']);closed_execution(ctx['budget'],plan['fund_nonce'],'post_exit_installation_funded');STATE['fund_verified']=True
  require(not any(x['kind']=='paid_final_installation_consumed' and x['payload'].get('reservation_id')==rid for x in ctx['budget'].rows),'Paid allowance already consumed')
  ready=source_ready(ctx,request);check_time(measurements(),before_publication=True)
  consumed=ctx['budget']._append('paid_final_installation_consumed',dict(reservation_id=rid,install_nonce=ctx['nonce'],paid_fund_close=reservation['fund_close'],installer_launch=e.bind(ctx['directory']/'launch.json'),installer_code=request['promotion'],installer_request=request_binding,lease=e.bind(lease_path(rid)),allowance_seconds_upper=ALLOWANCE,additional_debit_seconds=0.,model_forwards=0,status='consumed_for_single_install_attempt_actual_exit_still_required'))
  snapshot=ctx['budget'].snapshot();c.safe_snapshot(snapshot);fund_snapshot=e.checked(reservation['fund_closed_budget']);require(snapshot['active_seconds_charged_upper']==fund_snapshot['active_seconds_charged_upper'] and snapshot['physical_forwards_reserved_upper']==fund_snapshot['physical_forwards_reserved_upper'] and snapshot['physical_forwards_confirmed']==fund_snapshot['physical_forwards_confirmed'],'Another cost/model writer changed the paid snapshot')
  require(e.sha(P/'budget/journal.jsonl',ctx['journal_before']['bytes'])==ctx['journal_before']['sha256'],'Prior original-ledger bytes were rewritten')
  bb=e.immutable(ctx['directory']/'paid_installation_budget_snapshot.json',snapshot);observed=measurements();check_time(observed,before_publication=True)
  ir=e.immutable(ctx['directory']/'installation_measurement.json',dict(schema='spnv2-paid-installation-measurement-v1',reservation=e.bind(path),installer_code=request['promotion'],observation_before_final_publication=observed,allowance_upper_seconds=ALLOWANCE,raw_measurements_not_added_as_new_fee=True,publication_and_exit_tail_upper_seconds=TAIL,actual_wrapper0_and_whole_elapsed_bound_required=True))
  candidate=ready['candidate'];final=dict(schema='g50p-diagnostic-delivery-manifest-v1',status=candidate['scientific_delivery_status'],summary=candidate['summary'],proof=candidate['proof'],provisional=candidate['provisional'],final_settled_budget=bb,external_cost_settlement=candidate['external_cost_settlement'],verified_cost_candidate=request['candidate_manifest'],scientific_scope_complete=candidate['scientific_scope_complete'],independently_accepted_windows=candidate['independently_accepted_windows'],complete_ODR_runs=0,active_seconds_charged_upper=snapshot['active_seconds_charged_upper'],physical_forwards_reserved_upper=snapshot['physical_forwards_reserved_upper'],physical_forwards_confirmed=snapshot['physical_forwards_confirmed'],physical_forwards_unknown_upper=snapshot['physical_forwards_unknown_upper'],time_hard_stop_enabled=False,budget_policy=snapshot['budget_policy'],initial_cost_receipts=snapshot['initial_cost_receipts'],source_prefix=ctx['prefix'],installation_funding_reservation=e.bind(path),installation_paid_upper_seconds=ALLOWANCE,installation_measurement=ir,installation_launch=e.bind(ctx['directory']/'launch.json'),installation_outcome_path=str(ctx['directory']/'outcome.json'),installation_actual_exit_path=str(log_path)+'.exitcode',reservation_consumption_event=dict(sequence=consumed['sequence'],sha256=consumed['sha256']),valid_only_with_actual_installer_wrapper_exit0_and_elapsed_within_allowance=True,exact_total_time_claim=False,no_install_budget_attempt_opened=True,no_refund=True)
  proposed=e.immutable(ctx['directory']/'final_manifest_candidate.json',final);result=dict(status='paid_final_installation_complete_pending_actual_wrapper_exit',final_manifest=dict(path=str(FINAL),sha256=proposed['sha256']),installation_candidate=proposed,reservation_id=rid,model_forwards=0,actual_wrapper0_and_elapsed_bound_still_required=True)
  from budget import atomic_json
  atomic_json(ctx['directory']/'result.json',result);atomic_json(ctx['directory']/'outcome.json',dict(status=result['status'],exit_code=0,result=str(ctx['directory']/'result.json'),result_sha256=e.sha(ctx['directory']/'result.json'),model_forwards=0,actual_wrapper0_and_elapsed_bound_still_required=True))
  check_time(measurements(),before_publication=True)
  atomic_new_link(ctx['directory']/'final_manifest_candidate.json',FINAL,post_check=lambda:check_time(measurements()))
  # No further checked IO: root must verify the real exit and total bound.

def failure(error,observation):
 ctx=e.EXECUTION if e is not None else None;code=error.code if isinstance(error,SystemExit) and type(error.code) is int else 130 if isinstance(error,KeyboardInterrupt) else 1
 account=copy.deepcopy((ctx or {}).get('accounting') or {});paid_payload=((ctx or {}).get('close_event') or {}).get('payload',{});reserved=paid_payload.get('post_exit_installation_conservative_upper_seconds');fund_reserve_actually_charged=reserved==ALLOWANCE
 try:m=measurements()
 except Exception as timing_error:m=dict(measured_entry_wall_seconds=time.monotonic()-ENTRY_STARTED,whole_process_elapsed_upper_seconds=None,cpu=None,cpu_measurement_status='unavailable',timing_error=repr(timing_error))
 elapsed=m.get('whole_process_elapsed_upper_seconds');covered=STATE.get('fund_verified') is True and type(elapsed) in (int,float) and elapsed+TAIL<=ALLOWANCE
 payload=dict(schema='spnv2-paid-promotion-failure-v1',mode=STATE.get('mode',observation.get('action')),reservation_id=(STATE.get('reservation',{}).get('plan') or {}).get('reservation_id',STATE.get('reservation_id')),original_ledger_accounting=account,fund_installation_upper_actually_charged=fund_reserve_actually_charged,fund_prepaid_upper_is_invalid_without_actual_fund_wrapper0=True,request=observation.get('request'),nonce=os.environ.get('SPNV2_RUN_NONCE'),log_path=observation.get('log_path'),started_wall_time=ENTRY_WALL,measurements_before_failure_receipt=m,allowance_upper_seconds=ALLOWANCE,allowance_covered_with_final_tail=covered,accounting_pending=not covered,unfunded_overrun_lower_seconds=None if elapsed is None else max(0.,elapsed-ALLOWANCE),fund_verified_in_original_ledger=STATE.get('fund_verified',False),lease_owned=STATE.get('lease_owned',False),prepaid_upper_never_refunded=True,retry_requires_new_paid_reservation=True,error_type=type(error).__name__,error=str(error),actual_wrapper_exit_still_required=True,model_forwards=0)
 try:
  if ctx is not None and ctx.get('directory_created_here'):directory=ctx['directory']
  elif P.is_dir() and STATE.get('lease_owned') and hexnonce(os.environ.get('SPNV2_RUN_NONCE')) and not(P/'launches'/os.environ['SPNV2_RUN_NONCE']).exists():directory=P/'launches'/os.environ['SPNV2_RUN_NONCE'];directory.mkdir(exist_ok=False)
  elif P.is_dir():directory=P/'launches'/('promotion_unadmitted_'+str(os.getpid())+'_'+str(time.time_ns()));directory.mkdir(exist_ok=False)
  else:raise ValueError('No approved namespace for a runtime failure receipt')
  if e is not None:
   e._atomic_failure(directory/'promotion_failure_receipt.json',payload);e._atomic_failure(directory/'outcome.json',dict(status='failed',exit_code=code,promotion_failure_receipt=e.bind(directory/'promotion_failure_receipt.json'),accounting_pending=payload['accounting_pending'],model_forwards=0,no_final_success_claim=True))
  else:exclusive_json(directory/'promotion_failure_receipt.json',payload)
 except BaseException as write_error:payload['failure_receipt_error']=repr(write_error);print(json.dumps(payload,ensure_ascii=True,default=repr),file=sys.stderr)

def self_test():
 names=[]
 plan=dict(reservation_id='1'*32,install_nonce='2'*32,installer_request=dict(path='request',sha256='a'*64),allowance_seconds_upper=60.)
 p=fund_charge(1.,.5,1.6,plan);assert p['post_exit_installation_conservative_upper_seconds']==60. and p['measured_seconds_before_finalization']==1.5 and abs(p['charged_seconds_upper']-63.6)<1e-10;names.append('recognized_fund_charge_separates_fixed60_from_measured_time')
 def rejected(fn):
  try:fn()
  except (ValueError,FileExistsError):return
  raise AssertionError('Expected rejection')
 authority=dict(plan,fund_close_sha256='head');validate_authority(authority,plan['installer_request'],'2'*32,'head');rejected(lambda:validate_authority(authority,plan['installer_request'],'3'*32,'head'));rejected(lambda:validate_authority(authority,plan['installer_request'],'2'*32,'other'));names.append('exact_nonce_request_and_no_other_ledger_writer')
 rejected(lambda:validate_authority(authority,plan['installer_request'],'2'*32,'head',already_attempted=True));names.append('one_time_reservation_cannot_retry_or_refund')
 check_time(dict(whole_process_elapsed_upper_seconds=57.));rejected(lambda:check_time(dict(whole_process_elapsed_upper_seconds=61.)));rejected(lambda:check_time(dict(whole_process_elapsed_upper_seconds=None)));names.append('unknown_or_over60_duration_never_completes')
 from types import SimpleNamespace
 target=SimpleNamespace(parent='directory');events=[]
 def exists_link(a,b):raise FileExistsError('old accepted final')
 rejected(lambda:atomic_new_link('candidate',target,link=exists_link,sync=lambda p:None,same=lambda a,b:True,unlink=lambda p:events.append('deleted_old')));assert not events
 def broken():raise ValueError('caught post-link IO/timing failure')
 rejected(lambda:atomic_new_link('candidate',target,link=lambda a,b:events.append('new_link'),sync=lambda p:None,same=lambda a,b:True,unlink=lambda p:events.append('removed_own_unaccepted_link'),post_check=broken));assert events==['new_link','removed_own_unaccepted_link'];names.append('exclusive_final_and_only_own_caught_failure_rollback')
 lease=dict(pid=1,starttime=2,boot_id='fixture_boot',cmdline_sha256='a'*64,run_nonce='2'*32);terminal=dict(reservation_id=plan['reservation_id'],install_nonce=plan['install_nonce'],identity=lease,actual_exit_code=0,exact_identity_terminated=True,terminal_basis='supervisor_reaped',all_installation_disposition_wall_seconds_upper=62.,upper_basis='Synthetic supported upper, not a real fee')
 disposition=dict(schema='spnv2-paid-installation-terminal-disposition-v1',status='not_accepted_resolved',reservation_id=plan['reservation_id'],install_nonce=plan['install_nonce'],not_accepted=True,old_final_never_accepted=True,original_exit_evidence_preserved=True,actual_exit_code=0,final_link_was_created=True)
 assert validate_terminal_disposition(disposition,plan,0,lease,terminal,canonical_absent=True)==62.
 changed=dict(disposition,actual_exit_code=1);rejected(lambda:validate_terminal_disposition(changed,plan,0,lease,terminal,canonical_absent=True));rejected(lambda:validate_terminal_disposition(dict(disposition,not_accepted=False),plan,0,lease,terminal,canonical_absent=True));rejected(lambda:validate_terminal_disposition(disposition,plan,0,lease,terminal,canonical_absent=False))
 missing=dict(disposition,actual_exit_code=None,final_link_was_created=False);observed=dict(terminal,actual_exit_code=None,terminal_basis='not_present',exit_evidence_unavailable=True);assert validate_terminal_disposition(missing,plan,None,lease,observed,canonical_absent=True)==62.
 rejected(lambda:validate_terminal_disposition(missing,plan,None,lease,dict(observed,all_installation_disposition_wall_seconds_upper=None),canonical_absent=True));rejected(lambda:validate_terminal_disposition(missing,plan,None,lease,dict(observed,all_installation_disposition_wall_seconds_upper=0.),canonical_absent=True));names.append('root_disposition_preserves_real0_and_missing_exit_without_fake_failure')
 print(json.dumps(dict(status='pure_promotion_tests_passed',cases=len(names),test_names=names,model_forwards=0,no_runtime_or_ledger_IO=True,no_fund_or_install_executed=True)));return 0

def main():
 if sys.argv[1:]==['--self-test']:return self_test()
 observation={};STATE.clear()
 try:
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=('fund','install'),required=True);parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True);parser.add_argument('--log-path',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args();require(args.execute,'Explicit --execute required');observation.update(action=args.action,request=dict(path=args.request,sha256=args.request_sha256),log_path=args.log_path)
  path=Path(args.request).resolve(strict=True);require(path.is_relative_to(P) or path.is_relative_to(P.parent.parent/'migrations/sysu_20261027'),'Request must belong to this own plan/migration');require(path.stat().st_size<=8<<20 and sha(path)==args.request_sha256,'Promotion request binding changed');request=json.loads(path.read_text());own=request.get('promotion',{});require(Path(own.get('path','')).resolve()==Path(__file__).resolve() and own.get('sha256')==sha(__file__),'Explicit promotion code binding required');require(request.get('schema')=='spnv2-paid-final-promotion-request-v1' and request.get('action')==args.action,'Wrong promotion action/schema');load_helpers(request);e.EXECUTION=None;STATE['mode']=args.action
  if args.action=='fund':fund(request,observation['request'],args.log_path)
  else:install(request,observation['request'],args.log_path)
  return 0
 except BaseException as error:failure(error,observation);raise
if __name__=='__main__':raise SystemExit(main())
