"""Serial parent queue wrapper; override only original runner module's local job reference."""
import argparse
from entry_contract import *
from odr_contract import *

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv);verify_protocol()
 with original_odr_namespace() as common:
  original_job=common.job;task=original_job(a.task);parent=local_module(O/'run_task.py','_entry_base_odr_parent');need(parent.job is original_job,'Original runner module did not import the original job function')
  parent.job=lambda task_id:planned_task(original_job(task_id))
  try:
   if a.execute:
    need(os.environ.get('STY'),'Guarded screen required');fixed_extension_barrier(task,common);ensure_plan(task)
   with arguments([str(O/'run_task.py'),'--task',a.task]+(['--execute'] if a.execute else [])):parent.main()
   need(common.job is original_job,'Frozen common.job reference was mutated')
   if a.execute:return verify_extension(task,common)
  finally:parent.job=original_job;need(common.job is original_job,'Frozen common.job leaked a mutation')
if __name__=='__main__':
 try:main()
 except BaseException as error:event('error',error=repr(error),message='ODR entrypoint/extension acceptance failed; block parent queue');raise
