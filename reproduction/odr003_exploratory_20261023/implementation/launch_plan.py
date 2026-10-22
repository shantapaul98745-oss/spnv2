"""Run the existing queue engine in the independent ODR003 plan.

The old plan's lock is held as well, so the old manager cannot restart alongside
this manager. Historical task states remain at their original paths unchanged.
"""
import fcntl,os,sys
from pathlib import Path
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
OLD=N.parent/'rebaseline_20261017'
if __name__=='__main__':
 assert os.environ.get('STY')
 with (OLD/'pipeline.lock').open('a') as historical_lock:
  fcntl.flock(historical_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  sys.path.insert(0,str(N/'pipeline'))
  import runner
  runner.PLAN=N
  runner.main()
