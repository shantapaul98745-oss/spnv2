"""Tiny synthetic transport contracts; no model imports or checkpoint reads."""
from pathlib import Path
import hashlib
import io
import json
import os
import struct
import subprocess
import tempfile
import unittest

from receive_assets_v2 import Receiver
from transport_v2 import (FRAME_BYTES, artifact_family,
                          choose_basis, completion_ack, frame, signature, verified_ack)

ZSTD = os.environ.get("SPNV2_TEST_ZSTD", "/home/ps/miniconda3/envs/SPNv2/bin/zstd")
BASIS = "spnv2_runs/plans/g50p_risk_diagnostic_20261025/pairs/g50p_diag_v1_sunlamp_s2021_w0016/matched_scientific_state.bin"
OUTPUT = "spnv2_runs/plans/odr003_round2_20261023/runs/g50p2_det_20261024_p_sunlamp_s2021_N1024_B4/durable/checkpoints/generation-00000000000000000192/state.bin"
OTHER = "spnv2_runs/plans/g50p_risk_diagnostic_20261025/pairs/g50p_diag_v1_lightbox_s2022_w0001/matched_scientific_state.bin"
CANDIDATE = "spnv2_runs/plans/g50p_risk_diagnostic_20261025/pairs/g50p_diag_v1_sunlamp_s2021_w0016/candidate_learner.bin"


class TransportContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spnv2-transport-v2-contract-")
        self.root = Path(self.temp.name)
        self.migration = self.root / "migration"; self.migration.mkdir()
        data = (b"only synthetic unstructured fixture data\x00" + bytes(range(256))) * 128
        self.data = {BASIS: data, OUTPUT: data[:-11] + b"NEW-END-011", OTHER: data, CANDIDATE: data}
        self.allowed = {}
        for index, (rel, raw) in enumerate(self.data.items()):
            source = self.root / ("synthetic-source-" + str(index)); source.write_bytes(raw)
            self.allowed[rel] = dict(relative_path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                     source_path=str(source), source_stat=signature(source))
        manifest = dict(files=list(self.allowed.values()), raw_total_bytes=sum(row["bytes"] for row in self.allowed.values()))
        (self.migration / "asset_transfer_manifest.json").write_text(json.dumps(manifest))

    def tearDown(self):
        self.temp.cleanup()

    def compressed(self, rel, basis=None):
        command = [ZSTD, "-q"]
        if basis:
            command += ["--patch-from=" + self.allowed[basis]["source_path"], "--long=31"]
        return subprocess.run(command + ["-c", self.allowed[rel]["source_path"]], check=True, stdout=subprocess.PIPE).stdout

    def header(self, rel, basis=None):
        row = self.allowed[rel]
        value = {key: row[key] for key in ("relative_path", "bytes", "sha256")}
        if basis:
            value.update(compression="zstd_patch", basis_relative_path=basis, basis_sha256=self.allowed[basis]["sha256"])
        return value

    def session(self, header, payload, *, memory=16 << 30, fail=False):
        data = io.BytesIO(); data.write((json.dumps(header) + "\n").encode())
        if payload is not None:
            for start in range(0, len(payload), FRAME_BYTES):
                frame(data, payload[start:start + FRAME_BYTES])
            frame(data, b"")
        data.seek(0); replies = io.BytesIO()
        receiver = Receiver(self.migration, self.root / "project-tmp", ZSTD, data, replies, memory=lambda: memory)
        if fail:
            with self.assertRaises((ValueError, EOFError, BrokenPipeError)):
                receiver.serve()
        else:
            receiver.serve()
        return [json.loads(line) for line in replies.getvalue().splitlines()], receiver

    def test_legacy_normal_and_receipt_replay_create_exact_target_stat_ack(self):
        replies, receiver = self.session(self.header(BASIS), self.compressed(BASIS))
        self.assertEqual([reply["status"] for reply in replies], ["ready", "verified"])
        self.assertTrue(verified_ack(self.allowed[BASIS], replies[-1]))
        self.assertEqual((receiver.root / BASIS).read_bytes(), self.data[BASIS])
        replay, _ = self.session(self.header(BASIS), None)
        self.assertEqual([reply["status"] for reply in replay], ["already_verified"])
        self.assertTrue(verified_ack(self.allowed[BASIS], replay[0]))
        self.assertEqual(replay[0]["file_stat"], signature(receiver.root / BASIS))

    def test_patch_science_cross_artifact_roundtrip_is_exact_raw_bytes(self):
        self.session(self.header(BASIS), self.compressed(BASIS))
        replies, receiver = self.session(self.header(OUTPUT, BASIS), self.compressed(OUTPUT, BASIS))
        self.assertEqual(replies[0]["compression"], "zstd_patch")
        self.assertEqual(replies[0]["basis_relative_path"], BASIS)
        self.assertTrue(verified_ack(self.allowed[OUTPUT], replies[-1]))
        self.assertEqual((receiver.root / OUTPUT).read_bytes(), self.data[OUTPUT])

    def test_memory_and_missing_or_stale_receipt_negotiate_normal_before_frames(self):
        replies, _ = self.session(self.header(OUTPUT, BASIS), self.compressed(OUTPUT))
        self.assertEqual(replies[0]["compression"], "zstd")
        self.assertIn("no_current_verified_receipt", replies[0]["fallback_reason"])
        # Use a separate normal output as the verified basis for a low-memory request.
        self.session(self.header(BASIS), self.compressed(BASIS))
        replies, _ = self.session(self.header(OTHER, BASIS), self.compressed(OTHER), memory=(8 << 30) - 1)
        self.assertEqual(replies[0]["compression"], "zstd")
        self.assertEqual(replies[0]["fallback_reason"], "target_mem_available_below_8GiB")
        target_basis = self.migration / "source_snapshot" / BASIS
        stat = target_basis.stat()
        os.utime(target_basis, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        (self.migration / "source_snapshot" / OUTPUT).unlink()
        replies, _ = self.session(self.header(OUTPUT, BASIS), self.compressed(OUTPUT))
        self.assertEqual(replies[0]["compression"], "zstd")
        self.assertIn("no_current_verified_receipt", replies[0]["fallback_reason"])

    def test_unapproved_wrong_digest_and_cross_family_basis_reject_before_ready(self):
        variants = [self.header(OUTPUT, BASIS), self.header(OUTPUT, BASIS), self.header(OUTPUT, CANDIDATE)]
        variants[0]["basis_relative_path"] = "../outside"
        variants[1]["basis_sha256"] = "0" * 64
        for header in variants:
            replies, receiver = self.session(header, None, fail=True)
            self.assertEqual([reply["status"] for reply in replies], ["failed"])
            self.assertFalse((receiver.root / OUTPUT).exists())

    def test_bad_raw_sha_never_publishes_or_receipts(self):
        wrong = self.root / "corrupt-synthetic"; wrong.write_bytes(b"X" * len(self.data[OUTPUT]))
        payload = subprocess.run([ZSTD, "-q", "-c", str(wrong)], check=True, stdout=subprocess.PIPE).stdout
        replies, receiver = self.session(self.header(OUTPUT), payload, fail=True)
        self.assertEqual(replies[-1]["status"], "failed")
        self.assertFalse((receiver.root / OUTPUT).exists())
        self.assertFalse(receiver.receipt_path(OUTPUT).exists())

    def test_basis_selection_prefers_same_group_then_latest_exact_verified_ack(self):
        verified = {BASIS: 1, OTHER: 99, CANDIDATE: 100}
        self.assertEqual(choose_basis(self.allowed[OUTPUT], verified, self.allowed)["relative_path"], BASIS)
        self.assertEqual(choose_basis(self.allowed[OUTPUT], {OTHER: 99}, self.allowed)["relative_path"], OTHER)
        self.assertEqual(artifact_family(BASIS), artifact_family(OUTPUT))
        self.assertNotEqual(artifact_family(BASIS)[0], artifact_family(CANDIDATE)[0])
        self.assertFalse(verified_ack(self.allowed[BASIS], dict(status="already_verified")))
        # Source metadata must still match at basis selection, irrespective of target ACK.
        Path(self.allowed[BASIS]["source_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Immutable source"):
            choose_basis(self.allowed[OUTPUT], verified, self.allowed)

    def test_completion_skip_requires_exact_sha_and_any_supplied_path_size(self):
        row = self.allowed[BASIS]
        ack = dict(status="already_verified", sha256=row["sha256"])
        self.assertTrue(completion_ack(row, ack))
        self.assertFalse(verified_ack(row, ack))  # Cannot become a dictionary basis.
        for wrong in (dict(status="already_verified"), dict(ack, sha256="0" * 64),
                      dict(ack, relative_path=OUTPUT), dict(ack, bytes=row["bytes"] + 1)):
            self.assertFalse(completion_ack(row, wrong))

    def test_oversized_frame_refuses_before_allocating_or_publishing(self):
        data = io.BytesIO((json.dumps(self.header(OUTPUT)) + "\n").encode() + struct.pack("!I", FRAME_BYTES + 1))
        output = io.BytesIO()
        receiver = Receiver(self.migration, self.root / "project-tmp", ZSTD, data, output)
        with self.assertRaisesRegex(ValueError, "Invalid transfer frame size"):
            receiver.serve()
        self.assertFalse((receiver.root / OUTPUT).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
