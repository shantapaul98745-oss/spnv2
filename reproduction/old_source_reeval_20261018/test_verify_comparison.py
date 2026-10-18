"""CPU-only acceptance-gate tests; optionally check existing real report bindings.

Run in an independent CPU screen:
  python -B -s test_verify_comparison.py --real-plan /.../rebaseline_20261017

The production verifier's actual pure functions are loaded from its AST to avoid
importing queue/report-writing helpers.  No GPU modules or mutation paths load.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verifier', default=str(Path(__file__).with_name('verify_comparison.py')))
parser.add_argument('--real-plan')
ARGS, UNITTEST_ARGS = parser.parse_known_args()
VERIFIER = Path(ARGS.verifier)
tree = ast.parse(VERIFIER.read_text(encoding='utf-8'), filename=str(VERIFIER))
selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in {'require', 'full_sample_metrics'}]
if len(selected) != 2:
    raise RuntimeError('Expected the actual require/full_sample_metrics verifier functions')
namespace = {'math': math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(VERIFIER), 'exec'), namespace)
full_sample_metrics = namespace['full_sample_metrics']


def fixture(n=6740):
    return {'sample_count': n, 'counts': {'total': n},
            'extended_primary': {'applicable': True, 'metrics': {
                key: {'mean': value, 'count': n, 'total': n, 'missing': 0,
                      'coverage': 1.0, 'denominator_scope': 'all_samples', 'unit': unit}
                for key, value, unit in [('eT', 0.28, 'm'), ('eR', 9.9, 'deg'), ('pose', 0.22, 'score')]
            }}}


class FullSampleGateTests(unittest.TestCase):
    def test_accepts_complete_finite_n_over_n(self):
        value = fixture()
        before = copy.deepcopy(value)
        result = full_sample_metrics(value, 6740)
        self.assertEqual(set(result), {'eT', 'eR', 'pose'})
        self.assertEqual(result['pose']['count'], 6740)
        self.assertEqual(value, before)

    def test_rejects_missing_predictions_for_each_metric(self):
        for key in ('eT', 'eR', 'pose'):
            with self.subTest(metric=key):
                value = fixture()
                value['extended_primary']['metrics'][key]['missing'] = 1
                with self.assertRaises(ValueError):
                    full_sample_metrics(value, 6740)

    def test_rejects_accepted_subset_even_if_finite(self):
        for key in ('eT', 'eR', 'pose'):
            with self.subTest(metric=key):
                value = fixture()
                value['extended_primary']['metrics'][key]['count'] = 6730
                with self.assertRaises(ValueError):
                    full_sample_metrics(value, 6740)

    def test_rejects_wrong_metric_denominator_and_scope(self):
        for field, replacement in [('total', 6730), ('denominator_scope', 'original_accepted_subset')]:
            with self.subTest(field=field):
                value = fixture()
                value['extended_primary']['metrics']['pose'][field] = replacement
                with self.assertRaises(ValueError):
                    full_sample_metrics(value, 6740)

    def test_rejects_wrong_outer_sample_counts(self):
        for mode in ('sample_count', 'counts.total'):
            with self.subTest(mode=mode):
                value = fixture()
                if mode == 'sample_count':
                    value['sample_count'] = 6730
                else:
                    value['counts']['total'] = 6730
                with self.assertRaises(ValueError):
                    full_sample_metrics(value, 6740)

    def test_rejects_nan_inf_and_missing_mean(self):
        for key in ('eT', 'eR', 'pose'):
            for mean in (float('nan'), float('inf'), -float('inf'), None):
                with self.subTest(metric=key, mean=mean):
                    value = fixture()
                    value['extended_primary']['metrics'][key]['mean'] = mean
                    with self.assertRaises(ValueError):
                        full_sample_metrics(value, 6740)

    def test_rejects_inapplicable_hybrid(self):
        value = fixture()
        value['extended_primary']['applicable'] = False
        with self.assertRaises(ValueError):
            full_sample_metrics(value, 6740)

    @unittest.skipUnless(ARGS.real_plan, 'Pass --real-plan to check saved production reports read-only')
    def test_real_accepted_reports_have_compatible_fields_and_bindings(self):
        plan = Path(ARGS.real_plan)
        for domain, n in [('synthetic', 11994), ('lightbox', 6740), ('sunlamp', 2791)]:
            with self.subTest(domain=domain):
                base = json.loads((plan / ('reports/eval_phi3_' + domain + '_verified.json')).read_text())
                self.assertEqual(base['status'], 'passed')
                self.assertIn('summary', base)
                link = base['extension']
                for key in ('report', 'verification', 'entrypoint_verification'):
                    self.assertEqual(sha(link[key]['path']), link[key]['sha256'])
                report = json.loads(Path(link['report']['path']).read_text())
                proof = json.loads(Path(link['verification']['path']).read_text())
                entry = json.loads(Path(link['entrypoint_verification']['path']).read_text())
                self.assertEqual(proof['status'], 'passed')
                self.assertEqual(entry['status'], 'passed')
                self.assertEqual(report['domain'], domain)
                self.assertEqual(report['context']['weights']['source_final']['sha256'], base['source_sha256'])
                config = report['context']['configuration']
                self.assertEqual(config['sha256'], sha(config['path']))
                self.assertEqual(report['counts']['total'], n)
                self.assertLessEqual(report['counts']['original_accepted'], n)
                self.assertIsInstance(report['sample_id_sha256'], str)
                self.assertEqual(set(full_sample_metrics(report, n)), {'eT', 'eR', 'pose'})


if __name__ == '__main__':
    print('VERIFIER_SHA256=' + sha(VERIFIER), flush=True)
    unittest.main(argv=[__file__, *UNITTEST_ARGS], verbosity=2)
