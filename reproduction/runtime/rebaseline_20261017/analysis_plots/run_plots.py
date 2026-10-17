"""Create accepted real scientific plots or explicit pending point manifests."""
import argparse,datetime
from common import *
from render import curve,distribution,parameters,benchmark

def build(snapshot,run_id):
 env_guard();bundle=gate_snapshot(snapshot);out=D/'runs'/run_id;out.mkdir(exist_ok=False);plots=[];curves=expected_curves(bundle);validate_seed_interface(bundle,curves);paired=fixed_pairs(bundle);param=parameter_evidence();bench=bench_evidence()
 for i,item in enumerate(curves):
  if item['status']=='ready':
   p=out/('scan_'+str(i)+'.png');curve(item,p);plots.append(dict(kind=item['kind'],png=p.name,png_sha256=sha(p),data=item))
 for item in paired:
  p=out/(item['task_id']+'_distribution.png');distribution(item,p);plots.append(dict(kind='paired_distribution',png=p.name,png_sha256=sha(p),data=item))
 p=out/'Table6_parameters_only.png';parameters(param,p);plots.append(dict(kind='Table6_parameters',png=p.name,png_sha256=sha(p),data=param))
 if bench['status']=='accepted':
  p=out/'Table6_GPU_forward_resources.png';benchmark(bench,p);plots.append(dict(kind='Table6_GPU_forward',png=p.name,png_sha256=sha(p),data=bench))
 write(out/'benchmark_inputs.json',bench)
 write(out/'scan_points.json',curves);write(out/'fixed_seed2021_distributions.json',paired);write(out/'parameter_inputs.json',param);write(out/'plots.json',plots)
 reference=read(D/'reference/paper_reference.json');gaps=[];by={(r['source_config'],r['domain'],r['metric']):r for r in bundle['matrix'] if r['stage']=='source'}
 for ref in reference['metric_rows']:
  row=by.get((ref.get('config'),ref.get('domain'),ref.get('metric')));value=float(row['accepted_mean']) if row and row['status']=='已验收' and finite(row['accepted_mean']) else None;compatible=bool(value is not None and (not ref.get('requires_all_samples') or row['valid_denominator']==row['n_samples']))
  gaps.append(dict(reference=ref,current_accepted_mean=value,source_sha256=row['source_sha256'] if row else None,current_valid_denominator=int(row['valid_denominator']) if row and row['valid_denominator'] else None,status='accepted_numeric_comparison' if compatible else 'pending_source' if value is None else 'limited_denominator_mismatch',difference_actual_minus_paper=value-ref['value'] if compatible else None,causal_equivalence_not_claimed=True))
 write(out/'paper_gaps.json',gaps)
 pending=dict(status='pending' if any(c['status']!='ready' for c in curves) or len(paired)<8 else 'ready',expected_metric_series=56,expected_scan_points=504,ready_series=sum(c['status']=='ready' for c in curves),pending_scan_points=sum(p['status']!='ready' for c in curves for p in c['points']),expected_fixed_seed2021_pairs=8,accepted_fixed_pairs=len(paired),Table6='parameters_only',GPU_resource_bench=bench['status'],extended_evaluation=dict(status='pending_observer_evidence',raw_numeric=None,all_samples_comparable=False,requires='Same-forward raw-numeric H+E observer, finite N/N coverage for all-samples; no recovery from NaN MAT'),five_seeds_are='ODR observation streams, not source trainings',real_scan_curve_fabricated=False,required_parent_tasks=[dict(id='analysis_refresh_after_source_ODR_acceptance',command='analysis/run_analysis.py --run-id UNIQUE --cases then analysis/verify_snapshot.py --run-id UNIQUE',depends_on='accepted real source/ODR reports and 3MAT'),dict(id='scientific_plots_refresh',command='analysis_plots/run_plots.py --snapshot VERIFIED_ANALYSIS_RUN --run-id UNIQUE then analysis_plots/verify_plots.py --run-id UNIQUE',depends_on='independently accepted analysis snapshot'),dict(id='extended_raw_numeric_observer_evidence',status='pending',requires='root independent observer and verifier; separate N/N coverage before any all-samples paper comparison'),dict(id='resource_bench_plot_refresh',status=bench['status'],requires='resources/bench/reports/table6_forward_four_models_20261017.json + independent *_verified.json + exitcode0 + four raw SHA-bound arrays; no extrapolation')])
 write(out/'pending.json',pending);write(out/'input_manifest.json',dict(snapshot=bundle['snapshot'],inputs=bundle['inputs']|param['inputs']|bench['inputs']|{str(D/'reference/paper_reference.json'):sha(D/'reference/paper_reference.json')},created_at=datetime.datetime.now().astimezone().isoformat(),no_GPU_execution=True))
 write(out/'artifact_manifest.json',dict(files_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()},implementation_sha256={p.name:sha(p) for p in D.glob('*.py')},scope='accepted real evidence only; mock fixtures disjoint'))
 return pending
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--run-id',required=True);a=p.parse_args();print(json.dumps(build(a.snapshot,a.run_id),ensure_ascii=False))
