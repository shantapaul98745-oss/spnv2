"""Local saved-record diagnostics only; no SPNv2/image/GPU execution."""
import hashlib
import io
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import saved_state_diagnostic as diagnostic

diagnostic.torch, diagnostic.np = torch, np


class SavedDiagnosticTests(unittest.TestCase):
    def test_exact_and_changed_tensor_scale(self):
        left = torch.tensor([1., 2.])
        self.assertIsNone(diagnostic.numeric(left, left.clone()))
        result = diagnostic.numeric(left, torch.tensor([1., 2.5]))
        self.assertEqual(result['mismatched_elements'], 1)
        self.assertEqual(result['max_absolute'], .5)

    def test_recursive_first_difference_keeps_location(self):
        result = diagnostic.first_difference({'arrival':[1., 2.]}, {'arrival':[1., 3.]})
        self.assertEqual(result['path'], 'root.arrival.1')
        self.assertEqual(result['absolute_difference'], 1.)

    def test_saved_cpu_tensor_never_needs_cuda(self):
        tensor = torch.tensor([1., 2.])
        result = diagnostic.CPUUnpickler(io.BytesIO(pickle.dumps(tensor))).load()
        self.assertTrue(torch.equal(result, tensor))
        self.assertEqual(result.device.type, 'cpu')
        self.assertFalse(torch.cuda.is_initialized())

    def test_report_binding_tamper_cannot_be_diagnosed_as_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            path.write_text('{}', encoding='utf8')
            binding = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(diagnostic.checked(binding), path.resolve())
            path.write_text('{"changed": true}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                diagnostic.checked(binding)


if __name__ == '__main__':
    unittest.main()
