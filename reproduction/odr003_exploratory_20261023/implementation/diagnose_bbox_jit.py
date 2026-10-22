"""Single fixed-input GPU decoder diagnostic; no labels or adaptation."""
import os,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
D=N/'tests/bbox_jit_20261023_a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('STY') and not D.exists()
 D.mkdir()
 r=json.loads((N/'preflight/official_source_receipt.json').read_text());assert r['status']=='passed'
 assert sha(r['source_path'])==r['source_sha256']
 core=json.loads(Path(r['source_core_manifest_path']).read_text())['root'];sys.path.insert(0,core)
 import torch
 from config import cfg,update_config
 from nets import build_spnv2
 from dataset.transforms import build_transforms
 from data_boundary import ImageManifestDataset
 from utils.postprocess import raw_output_to_bbox
 from utils.utils import set_seeds_cudnn
 torch.set_num_threads(2)
 update_config(cfg,SimpleNamespace(cfg=r['config_path'],opts=['MODEL.PRETRAIN_FILE','']))
 set_seeds_cudnn(cfg,seed=2021)
 m=build_spnv2(cfg);m.load_state_dict(torch.load(r['source_path'],map_location='cpu',weights_only=True),strict=True)
 m=m.cuda().eval();m.requires_grad_(False)
 ds=ImageManifestDataset(N/'data/lightbox/stream_2021.json',transforms=build_transforms(cfg,is_train=False,load_labels=False))
 sample=ds[0];x=sample['image'].unsqueeze(0).cuda()
 head=m.heads[m.head_names.index('efficientpose')]
 with torch.no_grad():
  features=m.backbone(x)
  raw=head.box_net(features).float()
  raw_saved=raw.clone();anchors_saved=head.anchors.clone()
  outputs=[];graphs=[]
  for i in range(6):
   out=raw_output_to_bbox(raw,head.input_size,head.anchors);torch.cuda.synchronize()
   outputs.append(out.cpu().clone());graphs.append(str(torch.jit.last_executed_optimized_graph()))
  assert torch.equal(raw,raw_saved) and torch.equal(head.anchors,anchors_saved)
 compare=[]
 for i,o in enumerate(outputs):
  delta=(o-outputs[0]).abs();compare.append({'call':i+1,'different_coordinates':int((delta!=0).sum()),'max_abs_difference':float(delta.max()),'same_as_last':torch.equal(o,outputs[-1])})
 torch.save({'raw':raw.cpu(),'anchors':head.anchors.cpu(),'input_size':head.input_size.cpu(),'outputs':outputs},D/'fixed_inputs_outputs.pt')
 for i,g in enumerate(graphs):(D/('graph_%d.txt'%i)).write_text(g)
 result=dict(status='diagnostic_complete',source_sha256=r['source_sha256'],sample_id=sample['sample_id'],input_immutable=True,
             comparisons=compare,hot_outputs_identical=all(torch.equal(o,outputs[-1]) for o in outputs[1:]),
             cold_hot_graphs_differ=graphs[0]!=graphs[-1],script_sha256=sha(__file__),
             tensor_artifact_sha256=sha(D/'fixed_inputs_outputs.pt'),no_labels=True,no_optimizer=True)
 (D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
