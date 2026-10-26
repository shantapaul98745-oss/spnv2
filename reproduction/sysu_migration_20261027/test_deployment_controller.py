"""Synthetic filesystem/receipt checks; never invoke Controller.execute."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.sysu_migration_20261027 import deployment_controller as c
from reproduction.sysu_migration_20261027 import namespace_runner as ns


class ControllerMetadataTests(unittest.TestCase):
    def test_merge_preserves_asset_and_existing_lock_inodes_and_recovers_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);src=root/'staging/plans';dst=root/'canonical/plans'
            (src/'owned').mkdir(parents=True);(dst/'owned').mkdir(parents=True)
            source_lock=src/'owned/pipeline.lock';target_lock=dst/'owned/pipeline.lock'
            source_lock.write_bytes(b'');target_lock.write_bytes(b'')
            asset=src/'owned/report.json';asset.write_bytes(b'{"preserved":true}\n')
            asset_inode=asset.stat().st_ino;target_inode=target_lock.stat().st_ino;source_lock_inode=source_lock.stat().st_ino
            mover=c.TreeMover(root,root/'records',root/'archive',[target_lock])
            mover.move(src,dst)
            self.assertEqual((dst/'owned/report.json').stat().st_ino,asset_inode)
            self.assertEqual(target_lock.stat().st_ino,target_inode)
            archived=root/'archive'/source_lock.relative_to(root)
            self.assertEqual(archived.stat().st_ino,source_lock_inode)
            mover.move(src,dst)  # durable completion markers make a rerun read-only
            self.assertFalse(src.exists())

    def test_differing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);src=root/'stage';dst=root/'native'
            src.mkdir();dst.mkdir()
            (src/'evidence.json').write_text('source evidence')
            (dst/'evidence.json').write_text('different native evidence')
            mover=c.TreeMover(root,root/'records',root/'archive')
            with self.assertRaisesRegex(ns.MigrationConfigError,'existing destination'):
                mover.move(src,dst)
            self.assertEqual((src/'evidence.json').read_text(),'source evidence')
            self.assertEqual((dst/'evidence.json').read_text(),'different native evidence')

    def test_transfer_completion_requires_each_immutable_receipt_and_unchanged_stat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);m=root/'migration';s=m/'source_snapshot';d=root/'diagnostic'
            m.mkdir();rel='spnv2_runs/plans/owned/report.json';path=s/rel
            path.parent.mkdir(parents=True);path.write_bytes(b'fixture metadata\n')
            row=dict(relative_path=rel,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            receipts=m/'asset_receipts';receipts.mkdir()
            c.immutable(receipts/(c.sha_bytes(rel.encode())+'.json'),dict(status='verified',**row,file_stat=c.signature(path)))
            status=dict(phase='receiving',active=rel,error=None)
            c.atomic(m/'asset_receive_status.json',status)
            controller=c.Controller.__new__(c.Controller);controller.inputs={'asset_manifest':{'files':[row]}}
            with patch.multiple(c,U=root,M=m,S=s,D=d):
                self.assertFalse(controller.verified_assets())
                c.atomic(m/'asset_receive_status.json',dict(phase='complete',active=None,error=None,
                    verified_files=1,expected_files=1,verified_raw_bytes=row['bytes'],expected_raw_bytes=row['bytes'],model_forwards=0))
                self.assertTrue(controller.verified_assets())
                path.write_bytes(b'changed after receiver verification\n')
                with self.assertRaisesRegex(ns.MigrationConfigError,'receipt/stat mismatch'):
                    controller.verified_assets()


if __name__=='__main__':
    unittest.main()
