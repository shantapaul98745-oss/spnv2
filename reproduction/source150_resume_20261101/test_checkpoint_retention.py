"""Three bounded file-only contracts; no torch import or scientific inference."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from checkpoint_retention import FIXED_LIMITS, NativeCheckpointRetention, SCHEMA, sha256


def writer(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".native-tmp")
    with tmp.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.total = Path(self.temp.name).resolve()
        self.run = self.total / "model" / "run"
        self.run.mkdir(parents=True)
        self.policy = dict(schema=SCHEMA, retain_latest=2, run_output=str(self.run),
                           total_output_root=str(self.total), **FIXED_LIMITS)
        self.store = NativeCheckpointRetention(self.policy, disk_usage=lambda _: SimpleNamespace(free=100 * 1024**3))

    def test_three_saves_leave_exact_latest_two_native_payloads(self):
        for payload in (b"native-zero", b"native-one", b"native-two"):
            self.store.save(payload, self.store.current, writer)
        self.assertEqual(self.store.current.read_bytes(), b"native-two")
        self.assertEqual(self.store.previous.read_bytes(), b"native-one")
        meta = json.loads(self.store.metadata.read_text())
        self.assertEqual(meta["current"]["sha256"], sha256(self.store.current))
        self.assertEqual(meta["previous"]["sha256"], sha256(self.store.previous))
        self.assertFalse(list(self.run.glob(".checkpoint.pending-*")))

    def test_native_writer_failure_keeps_both_published_recovery_points(self):
        self.store.save(b"one", self.store.current, writer)
        self.store.save(b"two", self.store.current, writer)
        before = [p.read_bytes() for p in (self.store.current, self.store.previous, self.store.metadata)]

        def fails(value, path):
            Path(path).write_bytes(b"partial")
            raise OSError("injected interrupted serialization")

        with self.assertRaises(OSError):
            self.store.save(b"three", self.store.current, fails)
        self.assertEqual(before, [p.read_bytes() for p in (self.store.current, self.store.previous, self.store.metadata)])
        self.assertEqual(len(list(self.run.glob(".checkpoint.pending-*"))), 1)

    def test_non_checkpoint_pins_untouched_and_low_disk_refused_before_writer(self):
        for name in ("initial_state.pth.tar", "model_final.pth.tar", "manifest.json"):
            path = self.run / name
            self.store.save(name.encode(), path, writer)
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        self.store.disk_usage = lambda _: SimpleNamespace(free=20 * 1024**3)
        called = []
        with self.assertRaises(RuntimeError):
            self.store.save(b"never", self.store.current, lambda *args: called.append(args))
        self.assertFalse(called)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir() if p.name in before})


if __name__ == "__main__":
    unittest.main()
