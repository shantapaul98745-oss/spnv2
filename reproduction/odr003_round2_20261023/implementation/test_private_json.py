"""Pure writer contract; no server, process, resource query, or model calls."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import guarded_entry


class Handle(io.StringIO):
    def fileno(self): return 41
    def close(self): pass


class PrivateJSONTests(unittest.TestCase):
    def world(self):
        handle = Handle()
        fake_os = SimpleNamespace(getpid=lambda: 123, O_WRONLY=1, O_CREAT=64, O_EXCL=128,
                   O_RDONLY=0, O_DIRECTORY=65536, open=Mock(side_effect=[41, 42]),
                   fdopen=Mock(return_value=handle), fsync=Mock(), replace=Mock(), close=Mock())
        return fake_os, handle

    def test_private_inode_created_before_writing_and_published_with_fsync(self):
        fake_os, handle = self.world()
        ordered = Mock()
        for name in ('open', 'fdopen', 'fsync', 'replace', 'close'):
            ordered.attach_mock(getattr(fake_os, name), name)
        with patch.object(guarded_entry, 'os', fake_os), patch.object(Path, 'mkdir'):
            guarded_entry.atomic_json('/own/registry.json', {'records': []})
        fake_os.open.assert_any_call(Path('/own/registry.json.123.tmp'), 1 | 64 | 128, 0o600)
        self.assertEqual(handle.getvalue(), '{"records": []}\n')
        calls = ordered.mock_calls
        self.assertLess(calls.index(unittest.mock.call.fsync(41)),
                        calls.index(unittest.mock.call.replace(Path('/own/registry.json.123.tmp'), Path('/own/registry.json'))))
        fake_os.fsync.assert_any_call(42)
        fake_os.close.assert_called_once_with(42)

    def test_existing_temporary_inode_is_not_reused_or_published(self):
        fake_os, _ = self.world()
        fake_os.open.side_effect = FileExistsError('preserve previous attempt')
        with patch.object(guarded_entry, 'os', fake_os), patch.object(Path, 'mkdir'):
            with self.assertRaises(FileExistsError):
                guarded_entry.atomic_json('/own/registry.json', {'records': []})
        fake_os.fdopen.assert_not_called()
        fake_os.replace.assert_not_called()

    def test_invalid_json_never_replaces_trusted_record(self):
        fake_os, _ = self.world()
        with patch.object(guarded_entry, 'os', fake_os), patch.object(Path, 'mkdir'):
            with self.assertRaises(ValueError):
                guarded_entry.atomic_json('/own/registry.json', {'bad': float('nan')})
        fake_os.replace.assert_not_called()


if __name__ == '__main__': unittest.main()
