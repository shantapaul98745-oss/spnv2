"""Child wrapper: capture only actual post-adaptation evaluation in original audited ODR."""
import argparse
from entry_contract import *
from odr_contract import *

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args(argv);need(os.environ.get('STY'),'Guarded screen required');verify_protocol()
 with original_odr_namespace() as common:
  task=common.job(a.task);need(not task.get('reuse_of'),'Alias must never launch the GPU child wrapper');common.verify_implementation();common.verify_code(task['line']);source=common.accepted_source(task['phi']);paths=odr_paths(task['id']);plan=read(paths['plan'])
  need(plan['task']==task and plan['actual_command']==planned_task(task)['command'],'Parent wrapper plan missing/changed');need(not paths['capture'].exists() and not paths['extension'].exists(),'Previous capture/extension output preserved')
  root=O/('source_'+task['line'])
  with namespace([root/'tools',root/'core']):
   inference=importlib.import_module('engine.inference');need(Path(inference.__file__).resolve()==(root/'core/engine/inference.py').resolve(),'Wrong ODR line inference module')
   def factory(v,heads):
    need(str(v['cfg'].MODEL.PRETRAIN_FILE)==source['final'] and list(v['cfg'].TEST.HEAD)==heads,'Actual ODR cfg/source mismatch')
    return context(kind='odr',identifier=task['id'],cfg_path=task['config'],weights=dict(source_final=task['source_final'],adapted_before_evaluation=Path(task['output'])/'adapted_before_eval.pth.tar'),base_entry=O/'audited_entry.py',source_proof=task['source_report'],heads=heads,extra=dict(line=task['line'],line_manifest=dict(path=str(O/(task['line']+'_sha256.json')),sha256=sha(O/(task['line']+'_sha256.json'))),original_task_sha256=digest(task),original_ODR_implementation_sha256=sha(O/'implementation_manifest.json')))
   with EvaluationCapture(inference,paths=paths,context_factory=factory,csv_factory=lambda v:Path(v['cfg'].DATASET.ROOT)/v['cfg'].DATASET.DATANAME/v['cfg'].TRAIN.VAL_CSV,heads_factory=lambda v:list(v['cfg'].TEST.HEAD),result_factory=lambda v:Path(v['log_dir'])/task['domain']) as captured:
    with arguments([str(O/'audited_entry.py'),'--task',task['id']]):runpy.run_path(str(O/'audited_entry.py'),run_name='__main__')
   value=captured.build(task['config'])
  need(read(Path(task['output'])/'adaptation_audit.json')['task']==task,'Original adaptation audit task changed by wrapper')
  stable_write(paths['root']/'execution_record.json',dict(schema='spnv2-odr-evaluation-entrypoint-execution-v1',task_id=task['id'],original_task_sha256=digest(task),base_command=task['command'],actual_command=planned_task(task)['command'],context_sha256=digest(captured.captured['context']),report=str(paths['report']),report_sha256=sha(paths['report']),wrapper_protocol_sha256=sha(E/'protocol_manifest.json')))
  event('odr_extension_produced',task=task['id'],output=str(paths['report']),status=value['status']);return value
if __name__=='__main__':
 try:main()
 except BaseException as error:event('error',error=repr(error),message='ODR observer child failed; preserve adaptation and block downstream');raise
