"""One-shot CUDA entrypoint initialization smoke; no model/tensor/forward."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import traceback

PLAN=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
OUTPUT=PLAN/'preflight/test_cuda_entry_init_20261023_a.json'


def main():
    assert os.environ.get('STY'), 'Unique guarded screen required'
    assert not OUTPUT.exists(), 'Never overwrite previous smoke evidence'
    raw=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    processes=[]
    for line in raw.splitlines():
        pid,name,memory=[x.strip() for x in line.split(',',2)]
        assert pid=='7890' and name=='/usr/libexec/gnome-remote-desktop-daemon', 'Unexpected GPU process before smoke'
        processes.append({'pid':int(pid),'process_name':name,'memory_mib':int(memory)})
    import torch
    assert torch.cuda.is_available(), 'Actual CUDA device unavailable'
    before=torch.cuda.is_initialized()
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    initialized_after_set_device=torch.cuda.is_initialized()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    allocated=torch.cuda.max_memory_allocated(device)
    reserved=torch.cuda.max_memory_reserved(device)
    assert allocated==0 and reserved==0, 'Unexpected tensor allocation in entrypoint-only smoke'
    result=dict(status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                screen=os.environ['STY'],pid=os.getpid(),device=str(device),device_name=torch.cuda.get_device_name(device),
                torch_version=torch.__version__,cuda_version=torch.version.cuda,
                initialized_before=before,initialized_after_set_device=initialized_after_set_device,
                reset_peak_memory_stats_passed=True,synchronize_passed=True,
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                model_instantiated=False,forward_passes=0,tensor_allocations=0,GT_read=False,
                prelaunch_compute_processes=processes,
                sequence=['is_available','device_cuda0','set_device','reset_peak_memory_stats','synchronize'],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with OUTPUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        failure=OUTPUT.with_name(OUTPUT.stem+'_failure.json')
        if not failure.exists():
            with failure.open('x') as f:json.dump({'status':'failed','error':repr(error),'traceback':traceback.format_exc()},f,indent=2)
        raise
