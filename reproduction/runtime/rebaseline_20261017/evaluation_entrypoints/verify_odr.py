"""Independent CPU CLI: original full ODR verification then deterministic extension verification."""
import argparse
from entry_contract import *
from odr_contract import *

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args(argv);need(os.environ.get('STY'),'Guarded CPU screen required');verify_protocol()
 with CPU_environment():
  original_verify(a.task)
  with original_odr_namespace() as common:return verify_extension(common.job(a.task),common)
if __name__=='__main__':
 try:proof=main();print(json.dumps(proof,sort_keys=True,allow_nan=False))
 except BaseException as error:event('error',error=repr(error),message='Original/extended independent CPU verification failed; block downstream');raise
