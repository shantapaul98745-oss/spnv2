"""Run the frozen trainer with a separately bound, native storage-only adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from checkpoint_retention import NativeCheckpointRetention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-entry", required=True)
    parser.add_argument("--policy", required=True)
    args, original_args = parser.parse_known_args()
    if original_args[:1] == ["--"]:
        original_args = original_args[1:]
    base = Path(args.base_entry).resolve(strict=True)
    policy_path = Path(args.policy).resolve(strict=True)
    policy = json.loads(policy_path.read_text())
    expected = policy.get("base_entry_sha256")
    actual = hashlib.sha256(base.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise RuntimeError("Frozen training entry SHA does not match the storage policy")
    retention = NativeCheckpointRetention(policy)
    retention.preflight()
    sys.path.insert(0, str(base.parents[1] / "core"))
    sys.path.insert(0, str(base.parent))
    from utils import training_state
    native_writer = training_state.atomic_torch_save

    def guarded_save(value, path):
        return retention.save(value, path, native_writer)

    training_state.atomic_torch_save = guarded_save
    sys.argv = [str(base)] + original_args
    runpy.run_path(str(base), run_name="__main__")


if __name__ == "__main__":
    main()
