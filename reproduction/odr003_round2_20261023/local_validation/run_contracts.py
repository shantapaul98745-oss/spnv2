"""Local CPU-only contracts; never imports or launches the real SPNv2 model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT/'implementation'
sys.path.insert(0, str(IMPL))
sys.path.insert(1, str(ROOT.parent/'odr003_exploratory_20261023/implementation'))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append(dict(id=test.id(), outcome='passed'))
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append(dict(id=test.id(), outcome='failed'))
    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append(dict(id=test.id(), outcome='error'))
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes.append(dict(id=test.id(), outcome='skipped', reason=reason))


if __name__ == '__main__':
    if platform.system() != 'Windows':
        raise RuntimeError('This local contract driver is deliberately restricted to the local Windows host')
    import torch
    if torch.cuda.is_available() or '+cpu' not in torch.__version__:
        raise RuntimeError('Use the isolated local CPU-only Torch runtime')
    torch.set_num_threads(2)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(IMPL), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=RecordedResult).run(suite)
    log = ROOT/'local_validation/contract_output.txt'
    log.write_text(output.getvalue(), encoding='utf8')
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(IMPL.glob('*.py'))}
    passed = result.wasSuccessful() and not result.skipped
    def coverage(*modules):
        return passed and all(any(x['outcome']=='passed' and x['id'].startswith(module+'.')
                                  for x in result.outcomes) for module in modules)
    record = dict(schema='odr003-round2-local-contract-report-v1', status='passed' if passed else 'failed',
                  recorded_at=datetime.now(timezone.utc).isoformat(),
                  platform=dict(host='local_windows', python=sys.version, torch_version=torch.__version__,
                                real_spnv2_forward=False, shared_host_compute=False),
                  command=[sys.executable, '-B', str(Path(__file__).resolve())],
                  validation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  implementation_sha256=code, tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), skipped=len(result.skipped), test_results=result.outcomes,
                  stdout_path=str(log.resolve()), stdout_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                  stdout_archive_path='/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_round2_20261023/local_validation/contract_output.txt',
                  contract_coverage=dict(commit_rollback_state=coverage('test_protected_learning','test_protected_engine'),
                     pool_replay=coverage('test_guard_pools','test_verify_round2'),
                     interruption_recovery=coverage('test_durable_state','test_protected_engine','test_reference_driver'),
                     protection_rules=coverage('test_guard_pools'), RW_endpoints=coverage('test_protected_engine','test_round2_protocol')),
                  limits='Synthetic/mock local contracts only; actual GPU engineering and independent saved-state proof still required')
    (ROOT/'local_validation/contract_report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=record['status'], tests_run=result.testsRun, failures=record['failures'], errors=record['errors'])))
    if not passed: print(output.getvalue()[-25000:])
    raise SystemExit(0 if passed else 1)
