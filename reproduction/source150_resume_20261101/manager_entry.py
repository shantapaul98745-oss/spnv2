"""Run the unchanged source queue with a separately archived identity entry."""
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


def main():
    revision = Path(__file__).resolve().parent
    binding = json.loads((revision / 'deployment/binding.json').read_text())
    original = Path(binding['runner']['path'])
    if hashlib.sha256(original.read_bytes()).hexdigest() != binding['runner']['sha256']:
        raise RuntimeError('Original queue implementation changed')
    sys.path.insert(0, str(original.parent))
    import runner
    from progress import atomic_json
    from task_entry import process_identity
    nonce = uuid.uuid4().hex
    os.environ['SPNV2_SOURCE150_MANAGER_NONCE'] = nonce
    identity = process_identity(os.getpid())
    identity.update(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    nonce=nonce, command=sys.argv,
                    cmdline_sha256=hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest())
    record = revision / 'manager_identity.json'
    atomic_json(record, identity)
    if json.loads(record.read_text()) != identity:
        raise RuntimeError('Manager identity registration readback failed')
    # The unchanged runner locates its execution entry beside __file__.
    # Only that metadata entry changes; source tools/core remain hash-identical.
    runner.__file__ = __file__
    runner.main()


if __name__ == '__main__':
    main()
