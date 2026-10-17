"""Full user-plan scope checks. Validation is not experimental completion."""
import collections
import itertools
from pathlib import Path

CONFIG_EPOCHS = {'phi3_bn_ehs':20,'phi6_gn_ehs':20,'table2_h':10,'table2_e':20,
 'table2_eh':20,'table3_base':20,'table3_erase':20,'table3_flare':20,
 'table4_phi0_bn':20,'table4_phi3_gn':20}
DOMAINS={'synthetic':11994,'lightbox':6740,'sunlamp':2791}
REPORTING_KINDS={'table6_gpu_forward_resources','final_base_analysis',
 'final_extended_analysis','final_scientific_plots','full_completion_audit'}
BASE_KINDS={'source_training_20_epochs','source_training_10_epochs',
 'engineering_gpu_resume_pilot','full_source_evaluation','odr_fixed','odr_Bscan',
 'odr_Nscan','odr_alias_verification'}

def require(value, message):
 if not value:raise ValueError(message)

def option(command,name):
 require(isinstance(command,list) and command.count(name)==1 and command.index(name)+1<len(command),'Missing/ambiguous command option '+name)
 return command[command.index(name)+1]

def expected_odr_tuples():
 fixed={(3,line,d,s,1024,4) for line,d,s in itertools.product(['A','B1','B2'],['lightbox','sunlamp'],range(2021,2026))}
 fixed|={(6,'A',d,s,4096,1) for d,s in itertools.product(['lightbox','sunlamp'],range(2021,2026))}
 b={(3,line,d,s,1024,batch) for line,d,s,batch in itertools.product(['A','B2'],['lightbox','sunlamp'],range(2021,2026),[1,2,4,8,16,32])}
 n={(phi,'A',d,s,count,4) for phi,d,s,count in itertools.product([3,6],['lightbox','sunlamp'],range(2021,2026),[2**i for i in range(2,14)])}
 return {'fixed':fixed,'Bscan':b,'Nscan':n}

def validate_scope(manifest,odr_manifest,configs,*,require_reporting=False):
 tasks=manifest['tasks'];ids=[t['id'] for t in tasks]
 require(len(ids)==len(set(ids)),'Duplicate task IDs')
 require(set(configs)==set(CONFIG_EPOCHS),'Ten exact independent source configurations required')
 require(not(set(t.get('kind') for t in tasks)-BASE_KINDS-REPORTING_KINDS),'Unreviewed extra task kind')
 seen=set();ancestors={}
 for task in tasks:
  require(all(x in seen for x in task.get('requires',[])),'Task dependency missing or out of order: '+task['id'])
  ancestors[task['id']]=set(task.get('requires',[]))
  for parent in task.get('requires',[]):ancestors[task['id']].update(ancestors[parent])
  seen.add(task['id'])
  for command in [task.get('command',[]),task.get('verify_command',[])]:
   require(not any(any(marker in token.lower() for marker in ['batch16','batch_16','true_batch16']) for token in command),'Removed real-batch16 task reintroduced')
  gate=task.get('gpu',False)
  if gate is not False:
   require(gate=={'index':0,'min_free_mb':14500,'max_used_mb':1200},'Unreviewed GPU resource specification')
 trains=[t for t in tasks if t.get('kind','').startswith('source_training_')]
 by_config={Path(option(t['command'],'--cfg')).stem:t for t in trains}
 require(len(trains)==len(by_config)==10 and set(by_config)==set(CONFIG_EPOCHS),'Wrong source-training set or duplicated configuration')
 expected_windows=expected_samples=0
 for name,epochs in CONFIG_EPOCHS.items():
  cfg=configs[name];train=cfg['TRAIN'];physical=1 if name=='phi6_gn_ehs' else 4;task=by_config[name]
  require(train['END_EPOCH']==epochs and train['BEGIN_EPOCH']==0 and cfg['SEED']==2021,'Epoch/seed/init boundary mismatch: '+name)
  require(task['kind']==f'source_training_{epochs}_epochs','Training task epoch label mismatch')
  require(train['IMAGES_PER_GPU']==physical and train['ACCUM_STEPS']==4,'Physical batch/accumulation mismatch: '+name)
  require(cfg['DATASET']['INPUT_SIZE']==[768,512] and cfg['FP16'] is True,'Image size/AMP mismatch: '+name)
  require(train['OPTIMIZER']=='AdamW' and train['LR']==.001 and train['WD']==.0001,'Optimizer protocol mismatch: '+name)
  require(train['LR_STEP']==[15,18] and train['LR_FACTOR']==.1,'Learning-rate schedule mismatch: '+name)
  require(train['REPRODUCIBLE'] is True and cfg['AUTO_RESUME'] is False and not train['RESUME_FILE'],'Independent initialization/resume protocol mismatch')
  require('verify_train' in task['verify_command'] and option(task['verify_command'],'--cfg')==option(task['command'],'--cfg'),'Missing strict source verification')
  windows=11991 if physical==1 else 2997
  expected_windows+=windows*epochs;expected_samples+=windows*physical*4*epochs
 evals=[t for t in tasks if t.get('kind')=='full_source_evaluation'];eval_keys=[];eval_by_key={}
 for t in evals:
  stem=Path(option(t['command'],'--cfg')).stem
  matches=[(name,d) for name,d in itertools.product(CONFIG_EPOCHS,DOMAINS) if stem==name+'_eval_'+d]
  require(len(matches)==1,'Unknown source evaluation config: '+stem);name,domain=matches[0];eval_keys.append((name,domain))
  require(int(option(t['command'],'--expected-samples'))==DOMAINS[domain],'Evaluation sample count mismatch')
  require('verify_eval' in t['verify_command'],'Missing independent source evaluation verifier')
  require(option(t['verify_command'],'--cfg')==option(t['command'],'--cfg') and int(option(t['verify_command'],'--expected-samples'))==DOMAINS[domain],'Paired evaluation/verification config or sample count differs')
  require(by_config[name]['id'] in ancestors[t['id']],'Evaluation can precede its accepted training')
  eval_by_key[name,domain]=t['id']
  if require_reporting:
   require(any(Path(x).name=='capture_source.py' for x in t['command']) and any(Path(x).name=='capture_source.py' for x in t['verify_command']),'All thirty source evaluations require same-forward extension entry points')
 require(len(evals)==len(set(eval_keys))==30 and set(eval_keys)==set(itertools.product(CONFIG_EPOCHS,DOMAINS)),'Thirty full source domain evaluations required')
 pilots=[t for t in tasks if t.get('kind')=='engineering_gpu_resume_pilot']
 require(len(pilots)==4 and {t['id'] for t in pilots}=={'pilot_phi6_EHS','pilot_table2_h','pilot_table2_e','pilot_table2_eh'},'Four new-protocol pilot gates required')
 pilot_configs={'pilot_phi6_EHS':'phi6_gn_ehs','pilot_table2_h':'table2_h','pilot_table2_e':'table2_e','pilot_table2_eh':'table2_eh'}
 for pilot in pilots:
  name=pilot_configs[pilot['id']]
  require(Path(option(pilot['command'],'--cfg')).stem==name and option(pilot['command'],'--attempts')=='32' and option(pilot['command'],'--checkpoint')=='16','Pilot config/continuous-resume boundary mismatch')
  require(pilot['id'] in ancestors[by_config[name]['id']],'Formal training lacks its accepted pilot')
  require(option(pilot['verify_command'],'--report')==option(pilot['command'],'--report'),'Pilot CPU verifier points to different report')
 order=list(CONFIG_EPOCHS)
 for previous,current in zip(order,order[1:]):
  require(eval_by_key[previous,'sunlamp'] in ancestors[by_config[current]['id']],'Formal source order/evaluation barrier broken: '+current)
 odr=odr_manifest['tasks'];odr_ids={t['id'] for t in odr};ordered={t['id']:i for i,t in enumerate(odr)}
 require(len(odr)==len(odr_ids)==400 and odr_ids<=set(ids),'All400 ODR requests must appear exactly once')
 require({t['id'] for t in tasks if t.get('kind','').startswith('odr_')}==odr_ids,'Extra unreviewed ODR executions')
 expected=expected_odr_tuples()
 for group,tuples in expected.items():
  members=[t for t in odr if t['group']==group]
  actual=[tuple(t[k] for k in ['phi','line','domain','seed','N','B']) for t in members]
  require(len(actual)==len(set(actual))==len(tuples) and set(actual)==tuples,'ODR '+group+' protocol grid mismatch')
 aliases=[t for t in odr if t.get('reuse_of')];lookup={t['id']:t for t in odr};queue={t['id']:t for t in tasks}
 require(len(aliases)==30 and len(odr)-len(aliases)==370,'Expected370 executions and30 verified aliases')
 for t in odr:
  source_name='phi3_bn_ehs' if t['phi']==3 else 'phi6_gn_ehs';source_cfg=configs[source_name]
  final=str(Path(source_cfg['OUTPUT_DIR'])/source_cfg['MODEL']['BACKBONE']['NAME']/source_cfg['EXP_NAME']/'model_final.pth.tar')
  require(t['source_final']==final and t['source_report']==option(by_config[source_name]['verify_command'],'--report'),'ODR points to an old/unrelated source final')
  task=queue[t['id']];require(option(task['command'],'--task')==t['id'] and option(task['verify_command'],'--task')==t['id'],'Parent/ODR identity mismatch')
  if t.get('reuse_of'):
   require(t['reuse_of'] in lookup and ordered[t['reuse_of']]<ordered[t['id']],'Alias canonical must precede it')
   canonical=lookup[t['reuse_of']]
   require(not canonical.get('reuse_of') and all(t[k]==canonical[k] for k in ['phi','line','domain','seed','N','B','source_final','source_report','code_sha256_manifest']),'Cross-protocol/source alias reuse')
   require(task.get('gpu',False) is False,'Alias must not allocate GPU')
  else:require(isinstance(task.get('gpu'),dict),'Actual ODR requires serial GPU gate')
  if require_reporting:
   require(any(Path(x).name=='run_odr.py' for x in task['command']) and any(Path(x).name=='verify_odr.py' for x in task['verify_command']),'ODR missing same-forward extension verification')
 reporting=collections.Counter(t.get('kind') for t in tasks if t.get('kind') in REPORTING_KINDS)
 fixed_ids={t['id'] for t in odr if t['group']=='fixed'};scan_ids=odr_ids-fixed_ids;all_evals={t['id'] for t in evals}
 main_evals={eval_by_key[name,d] for name,d in itertools.product(['phi3_bn_ehs','phi6_gn_ehs'],DOMAINS)}
 require(all(main_evals<=ancestors[x] for x in fixed_ids),'Fixed ODR can precede either main model/domain acceptance')
 require(fixed_ids<=ancestors['pilot_table2_h'],'Fixed40-to-Table2 barrier broken')
 require(all(fixed_ids|all_evals<=ancestors[x] for x in scan_ids),'Scan can precede fixed40 or formal source/evaluation completion')
 if require_reporting:require(reporting==collections.Counter({k:1 for k in REPORTING_KINDS}),'Every required final reporting/resource stage must occur exactly once')
 if require_reporting:
  stages={t['kind']:t['id'] for t in tasks if t.get('kind') in REPORTING_KINDS};bench=stages['table6_gpu_forward_resources']
  require(all_evals|fixed_ids<=ancestors[bench] and all(bench in ancestors[x] for x in scan_ids),'Table4-to-benchmark-to-scans barrier broken')
  base,extended,plots,final=[stages[k] for k in ['final_base_analysis','final_extended_analysis','final_scientific_plots','full_completion_audit']]
  require(odr_ids|all_evals|{bench}<=ancestors[base],'Final base analysis lacks complete upstream evidence barrier')
  require(base in ancestors[extended] and {base,extended,bench}<=ancestors[plots] and {base,extended,plots}<=ancestors[final],'Final reporting sequence broken')
 return dict(status='full_scope_validated_not_experiment_completion',formal_configs=10,formal_epochs=sum(CONFIG_EPOCHS.values()),
  source_evaluations=30,pilots=4,fixed_ODR_requests=40,B_scan_requests=120,N_scan_requests=240,
  unique_ODR_experiments=370,verified_alias_requests=30,expected_training_update_attempts=expected_windows,
  expected_training_sample_appearances=expected_samples,reporting_stage_counts=dict(reporting))
