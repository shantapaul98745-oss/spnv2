"""Pure saved-record diagnostic contracts; no model, image, label or GPU use."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_existing as module

from analyze_existing import (SOURCE_SHA,summary_audit,ranking_metrics,decision_runs,
                              trace_audit,static_diagnostic,distribution,_initial_error_bins)


def summary_fixture():
    out=dict(source_sha256=SOURCE_SHA,resources_by_run={})
    for domain,seeds in [('lightbox',(154,152,152)),('sunlamp',(114,108,113))]:
        for seed,q in zip((2021,2022,2023),seeds):
            vetoes=23 if domain=='sunlamp' and seed==2021 else 94 if domain=='sunlamp' and seed==2022 else 0
            counts=dict(candidates=q,commits=q-vetoes,rollbacks=vetoes)
            windows=[dict(number=i,decision='rollback' if i>q-vetoes else 'commit',checked_images=2)
                     for i in range(1,q+1)]
            out['resources_by_run'][f'P:{domain}:{seed}']=dict(counts=counts,window_diagnostics=dict(
                window_decisions=windows,pool_diagnostics={},rule_comparable_image_counts={'new_unavailable':2*q},
                rule_triggered_image_counts={'new_unavailable':vetoes},rule_triggered_window_counts={'new_unavailable':vetoes}))
    return out


def trace_fixture():
    observations=[dict(kind='observation',sample_id=f's{i:04d}',observation_index=i,accepted=i<12,
        arrival=dict(entropy=.1,confidence=.8,reprojection=.01,rotation_disagreement=2.,translation_disagreement=.1)) for i in range(1024)]
    lookup={e['sample_id']:e for e in observations};slots={'uniform':[],'fragile':[]};windows=[]
    for number in (1,2,3):
        current=[f's{i:04d}' for i in range((number-1)*4,number*4)]
        chosen={}
        for origin,ids in [('current_window',current),('uniform',slots['uniform']),('fragile',slots['fragile'])]:
            for slot,sid in enumerate(ids):
                if sid not in chosen:chosen[sid]=dict(sample_id=sid,protection_sources=[])
                chosen[sid]['protection_sources'].append(dict(pool=origin,slot=slot))
        records=list(chosen.values());pairs=[]
        for sid in chosen:
            trigger=sid=='s0000' and number>=2
            sub={name:dict(comparable=True,triggered=False,value=1.,threshold=15.,
                compared_value=1.,compared_threshold=15.,operator='post > pre + threshold')
                for name in ('h_rotation_change','he_disagreement_increase','reprojection_increase')}
            pairs.append(dict(sample_id=sid,rollback=trigger,reasons=['previously_legal_physical_rejection'] if trigger else [],
                rules=dict(new_unavailable=dict(comparable=True,triggered=trigger),
                           joint_geometry_worsening=dict(comparable=True,triggered=False,subconditions=sub))))
        events=[]
        if number==1:
            for pool in slots:
                for slot,sid in enumerate(current):
                    events.append(dict(pool=pool,selected=True,slot=slot,sample_id=sid,replaced_sample_id=None))
                    slots[pool].append(sid)
        elif number==2:
            events.append(dict(pool='fragile',selected=True,slot=0,sample_id='s0004',replaced_sample_id='s0000'))
            slots['fragile'][0]='s0004'
        windows.append(dict(kind='window',number=number,sample_ids=current,decision='rollback' if number>=2 else 'commit',
            protection_records=records,protection=dict(rows=pairs,checked_image_count=len(records),rollback=number>=2),
            pool_transition=dict(pool_events=events)))
    counts=dict(observed=1024,accepted=12,candidates=3,commits=1,rollbacks=2)
    return dict(scientific_trace=observations+windows,counters=counts,
                pools={pool:dict(slots=value) for pool,value in slots.items()}),dict(counts=counts)


class SavedDiagnosticContracts(unittest.TestCase):
    def test_saved_summary_793_117_and_drift_is_not_silently_accepted(self):
        source=summary_fixture();result=summary_audit(source)
        self.assertEqual((result['candidates'],result['rollbacks'],result['commits']),(793,117,676))
        source['resources_by_run']['P:sunlamp:2022']['counts']['rollbacks']=93
        with self.assertRaises(ValueError):summary_audit(source)

    def test_auroc_average_ties_and_ap_are_not_selected_threshold_metrics(self):
        result=ranking_metrics([.1,.4,.35,.8],[False,False,True,True])
        self.assertEqual(result['AUROC'],.75);self.assertAlmostEqual(result['AUPRC'],5/6)
        tied=ranking_metrics([1.,1.,1.,1.],[True,False,True,False])
        self.assertEqual(tied['AUROC'],.5);self.assertEqual(tied['AUPRC'],.5)
        reversed_direction=ranking_metrics([4.,3.,2.,1.],[False,False,True,True])
        self.assertEqual(reversed_direction['AUROC'],0.)
        self.assertFalse(reversed_direction['direction_flipped_after_results'])

    def test_missing_and_one_class_remain_explicitly_unavailable(self):
        result=ranking_metrics([None,1.,2.],[True,False,None])
        self.assertEqual((result['population'],result['available'],result['missing_score'],result['missing_label']),(3,1,1,1))
        self.assertIsNone(result['AUROC']);self.assertIsNone(result['AUPRC'])
        self.assertEqual(distribution([None,None])['mean'],None)

    def test_nonfinite_saved_metadata_is_not_a_missing_zero(self):
        with self.assertRaises(ValueError):ranking_metrics([float('nan'),1.],[False,True])
        with self.assertRaises(ValueError):distribution([float('inf')])

    def test_source_dedup_residence_recurrence_and_replacements_from_saved_records(self):
        doc,resource=trace_fixture();result=trace_audit(doc,resource)
        self.assertEqual(result['sources']['deduplicated_mentions_removed'],7)
        self.assertEqual(result['sources']['source_combinations']['fragile+uniform'],7)
        self.assertEqual(result['recurring_trigger_images'][0]['sample_id'],'s0000')
        self.assertEqual(result['recurring_trigger_images'][0]['triggering_windows'],[2,3])
        self.assertEqual(result['recurring_trigger_images'][0]['longest_consecutive_trigger_run'],2)
        old=[x for x in result['residence_intervals'] if x['sample_id']=='s0000' and x['pool']=='fragile'][0]
        self.assertEqual((old['start_after_window'],old['end_after_window'],old['right_censored']),(1,2,False))
        self.assertEqual(result['final_reservoir_slots']['fragile'][0],'s0004')
        self.assertEqual(result['gate_accepted_vs_rejected']['accepted']['entropy']['total'],12)
        self.assertEqual(result['gate_accepted_vs_rejected']['rejected']['entropy']['total'],1012)

    def test_inconsistent_pool_sources_block_instead_of_filling_history(self):
        doc,resource=trace_fixture()
        doc['scientific_trace'][-1]['protection_records'][-1]['protection_sources'][0]['slot']=99
        with self.assertRaises((ValueError,IndexError)):trace_audit(doc,resource)

    def test_initial_error_stratification_and_predeclared_score_direction(self):
        initial=[dict(sample_id=f'd/{i}',valid=True,error=i/512) for i in range(512)]
        final=[dict(sample_id=f'd/{i}',valid=True,positive_degradation=1. if i>=256 else 0.) for i in range(512)]
        streams={str(seed):dict(reference_ids=[r['sample_id'] for r in initial],
               methods=dict(O=dict(per_sample=initial),**{m:dict(per_sample=final) for m in ('A','G50','P','RW','RP')})) for seed in (2021,2022,2023)}
        summary=dict(domains={domain:dict(streams=streams) for domain in ('lightbox','sunlamp')})
        saved={domain:[dict(sample_id=f'd/{i}',error=i/512,entropy=i/512,confidence=1-i/512,
                           reprojection=i/512,rotation_disagreement=i/512,translation_disagreement=i/512) for i in range(512)]
               for domain in ('lightbox','sunlamp')}
        config=dict(signals=[dict(name=f,field=f,risk_direction=d) for f,d in module.STATIC_DIRECTIONS.items()],
                    targets=[dict(name=m+'_final_degradation',method=m,kind='greater_than',field='positive_degradation',value=0.)
                             for m in ('A','G50','P','RW','RP')],
                    initial_error_quantiles=[0,.25,.5,.75,1])
        result=static_diagnostic(summary,saved,config)
        rows=result['domains']['lightbox']['results']
        self.assertTrue(all(row['metrics']['AUROC']==1. for row in rows if row['stratum']=='all'))
        self.assertTrue(all(row['metrics']['AUROC'] is None for row in rows if row['stratum']!='all'))
        self.assertFalse(result['score_threshold_search']);self.assertFalse(result['causal_update_harm_claim'])

    def test_no_config_does_not_invent_outcome_or_initial_error_bins(self):
        result=static_diagnostic({}, {}, None)
        self.assertEqual(result['status'],'not_evaluated')

    def test_equal_initial_error_ties_use_predeclared_ID_order_not_after_error(self):
        rows=[dict(sample_id=f'd/{i:04d}',valid=True,error=1.) for i in range(511)]
        rows.append(dict(sample_id='d/missing',valid=False,error=None))
        layers,bins=_initial_error_bins(list(reversed(rows)),[0,.25,.5,.75,1],[r['sample_id'] for r in rows[:-1]])
        self.assertEqual([layer['count'] for layer in layers],[128,128,128,127])
        self.assertEqual(layers[0]['sample_ids'],[f'd/{i:04d}' for i in range(128)])
        self.assertIsNone(bins['d/missing'])

    def test_server_wrapper_denies_unadmitted_work_before_any_read_or_forward(self):
        with patch.object(module,'load_json',side_effect=AssertionError('unadmitted metadata read')):
            with self.assertRaisesRegex(ValueError,'admission'):
                module.execute('not-opened.json',admission={},launch_record={},conflict_check=lambda:None,
                               budget=SimpleNamespace(check_time=lambda:None))

    def test_server_wrapper_rejects_different_frozen_parent_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frozen=root/'frozen.json';extract=root/'extract.json';task=root/'task.json'
            frozen.write_text(json.dumps(dict(runs={'lightbox:2021':dict(run_key='P:lightbox:2021',
                original_final_checkpoint=dict(state_sha256='a'*64))})),encoding='utf8')
            frozen_binding=dict(path=str(frozen),sha256=module.sha(frozen))
            extract.write_text(json.dumps(dict(schema='g50p-risk-diagnostic-A-extraction-proof-v1',status='passed',
                model_forwards=0,frozen_inputs=frozen_binding,parent_final_state_sha256={'P:lightbox:2021':'b'*64})),encoding='utf8')
            task.write_text(json.dumps(dict(run_id='A_local_contract',frozen_inputs=frozen_binding,
                extraction_proof_path=str(extract))),encoding='utf8')
            with patch.object(module,'analyze',side_effect=AssertionError('mismatched parent analyzed')):
                with self.assertRaisesRegex(ValueError,'parent checkpoint SHA'):
                    module.execute(task,admission=dict(allow_start=True,boundary_verified=True),
                        launch_record=dict(run_id='A_local_contract'),conflict_check=lambda:None,
                        budget=SimpleNamespace(check_time=lambda:None))


if __name__=='__main__':unittest.main()
