"""Archive project runtime code for Git review; never operate experiments or Git refs."""
import argparse,hashlib,json,pathlib,os,tempfile
REPO=pathlib.Path(__file__).resolve().parents[1]
BASE=REPO.parent;RUNS=BASE/'spnv2_runs';PLAN=RUNS/'plans/rebaseline_20261017'
ARCHIVE=REPO/'reproduction/runtime/rebaseline_20261017'
GROUPS=('pipeline','tasks','tests','configs','pilots','queue_proposals','resources','odr')
EXCLUDED={'logs','reports','outputs','artifacts','validation','history','backups','downloads','source','source_A','source_B1','source_B2','v3','__pycache__'}
EXTENSIONS={'.py','.yaml','.yml','.sh','.md','.patch'}

def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
 try:
  with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def sync(include_analysis=False,include_bench=False):
 mapping={}
 groups=GROUPS+(('analysis',) if include_analysis else ())
 for group in groups:
  root=PLAN/group
  if not root.exists():continue
  for source in sorted(root.rglob('*')):
   relative=source.relative_to(root)
   if group=='resources' and 'bench' in relative.parts and not include_bench:continue
   if group=='analysis' and relative.parts and relative.parts[0] in ('runs','tests'):continue
   if source.is_file() and not set(relative.parts)&EXCLUDED and source.suffix in EXTENSIONS:
    mapping[ARCHIVE/group/relative]=source
 for name in ('EXECUTION_PLAN.md','protocol_matrix.json','source_manifest.json','queue.json','queue_v2.json'):
  if (PLAN/name).is_file():mapping[ARCHIVE/name]=PLAN/name
 for directory in ('v3b',):
  for name in ('proposal.json.gz','proposal_index.json','asset_catalog.json','report_contracts.json','archive_verification.json'):
   source=PLAN/'queue_proposals'/directory/name
   if source.is_file():mapping[ARCHIVE/'queue_proposals'/directory/name]=source
 if include_bench:
  for name in ('report_schema.json','pending_task.json','preparation_manifest.json'):
   source=PLAN/'resources/bench'/name
   if source.is_file():mapping[ARCHIVE/'resources/bench'/name]=source
 if include_analysis:
  source=PLAN/'analysis/implementation_manifest.json'
  if source.is_file():mapping[ARCHIVE/'analysis/implementation_manifest.json']=source
 for name in ('A_sha256.json','B1_sha256.json','B2_sha256.json','implementation_manifest.json'):
  mapping[ARCHIVE/'odr'/name]=PLAN/'odr'/name
 for name in ('run_in_screen_guarded.sh','training.screenrc','compact_progress.py'):
  mapping[REPO/'reproduction/runtime/shared'/name]=RUNS/'scripts'/name
 mapping[REPO/'reproduction/runtime/shared/progress.sh']=RUNS/'progress.sh'
 records=[]
 for target,source in sorted(mapping.items()):
  resolved=source.resolve();assert BASE in resolved.parents and resolved.is_file()
  content=source.read_bytes()
  if source.suffix=='.gz':
   import gzip
   text=gzip.decompress(content).decode('utf-8')
  else:text=content.decode('utf-8')
  if '-----BEGIN OPENSSH PRIVATE KEY-----' in text or '-----BEGIN RSA PRIVATE KEY-----' in text:
   raise ValueError('Private key material is not a code archive input: '+str(source))
  if not target.exists() or target.read_bytes()!=content:atomic(target,content)
  assert source.read_bytes()==content,'Runtime source changed while archiving: '+str(source)
  records.append(dict(runtime_path=str(source),repository_path=str(target.relative_to(REPO)),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
 generated={}
 for name in ('odr/manifest.json','queue_proposals/pending_source_tasks.json','queue_proposals/full_source_v2_proposal.json'):
  file=PLAN/name
  if file.exists():generated[name]=dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),reason='Generated task catalogue; versioned generator/configs and active queue reproduce this evidence')
 index=dict(schema='runtime-code-archive-v1',files=records,generated_catalogues=generated,
            notes=['Code copies for Git review. Actual runs keep their original server paths.','Dataset, trained weights, raw logs, live state and paper PDFs are not copied.','Source A/B1/B2 frozen copies are represented by main code plus exact patches/hash inventories.'])
 atomic(REPO/'reproduction/archive_manifest.json',(json.dumps(index,ensure_ascii=False,indent=2)+'\n').encode())
 return index

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--include-analysis',action='store_true');p.add_argument('--include-bench',action='store_true');a=p.parse_args()
 index=sync(a.include_analysis,a.include_bench);print(json.dumps(dict(files=len(index['files']),total_bytes=sum(x['bytes'] for x in index['files']))))
