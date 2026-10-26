"""Candidate manifest-whitelisted receiver; default normal v1 framing remains.

Only main acquires the shared v1 lock or serves transport. A patch dictionary
must be an already-verified full immutable manifest asset at this target, with
its own current target stat matching its trusted receipt. No delta is stored.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
import time

from transport_v2 import (FRAME_BYTES, TARGET_PATCH_MIN_MEMORY, artifact_family,
                          digest, exact, load_manifest, mem_available, signature)

DEFAULT_M = "/home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027"
DEFAULT_ZSTD = "/home/ps/miniconda3/envs/SPNv2/bin/zstd"
DEFAULT_TMP = "/tmp/spnv2_sysu_20261026/asset-transfer"


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


class Receiver:
    def __init__(self, migration, temporary, zstd, input_stream, output_stream, *, memory=mem_available):
        self.migration, self.temporary, self.zstd = Path(migration), Path(temporary), str(zstd)
        self.root, self.receipts = self.migration / "source_snapshot", self.migration / "asset_receipts"
        self.manifest, self.allowed = load_manifest(self.migration / "asset_transfer_manifest.json")
        self.input, self.output, self.memory = input_stream, output_stream, memory
        self.receipts.mkdir(exist_ok=True)
        self.temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.temporary.is_symlink() or self.temporary.stat().st_uid != os.getuid():
            raise ValueError("Unexpected temporary owner/type")

    def reply(self, value):
        self.output.write((json.dumps(value) + "\n").encode()); self.output.flush()

    def path(self, rel):
        path = self.root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Asset path escapes staging root or uses a symlink")
        return path

    def receipt_path(self, rel):
        return self.receipts / (hashlib.sha256(rel.encode()).hexdigest() + ".json")

    def receipt(self, rel):
        path = self.receipt_path(rel)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("Unexpected verified receipt type/size")
        return json.loads(path.read_text())

    def verified(self, row):
        path, receipt = self.path(row["relative_path"]), self.receipt(row["relative_path"])
        return bool(path.is_file() and receipt and receipt.get("status") == "verified"
                    and receipt.get("relative_path") == row["relative_path"] and receipt.get("sha256") == row["sha256"]
                    and receipt.get("bytes") == row["bytes"] and receipt.get("file_stat") == signature(path)
                    and path.stat().st_size == row["bytes"])

    def progress(self, active=None, compressed=0, error=None, compression=None, basis=None):
        rows = [row for row in self.allowed.values() if self.verified(row)]
        atomic(self.migration / "asset_receive_status.json", dict(schema="sysu-asset-transfer-v1", transport_version=2,
               updated_at=time.time(), phase="complete" if len(rows) == len(self.allowed) else ("failed" if error else "receiving"),
               verified_files=len(rows), expected_files=len(self.allowed), verified_raw_bytes=sum(row["bytes"] for row in rows),
               expected_raw_bytes=self.manifest["raw_total_bytes"], active=active, active_compressed_bytes=compressed,
               active_compression=compression, active_basis=basis, error=error, model_forwards=0))

    def record(self, row, *, reused, compressed=0, compression="zstd", basis=None):
        path = self.path(row["relative_path"])
        value = dict(status="verified", relative_path=row["relative_path"], sha256=row["sha256"], bytes=row["bytes"],
                     file_stat=signature(path), verified_at=time.time(), reused=reused, compressed_bytes=compressed,
                     compression=compression, basis_relative_path=basis)
        atomic(self.receipt_path(row["relative_path"]), value)
        return value

    def negotiate(self, header, row):
        mode = header.get("compression", "zstd")
        if mode == "zstd":
            if "basis_relative_path" in header or "basis_sha256" in header:
                raise ValueError("Normal header cannot specify a patch basis")
            return dict(status="ready", compression="zstd"), None
        if mode != "zstd_patch":
            raise ValueError("Unsupported compression mode")
        rel = header.get("basis_relative_path")
        basis = self.allowed.get(rel) if isinstance(rel, str) else None
        if (basis is None or rel == row["relative_path"] or header.get("basis_sha256") != basis["sha256"]
                or artifact_family(rel) is None or artifact_family(row["relative_path"]) is None
                or artifact_family(rel)[0] != artifact_family(row["relative_path"])[0]):
            raise ValueError("Patch basis is outside the approved manifest/family/digest")
        if self.memory() < TARGET_PATCH_MIN_MEMORY:
            return dict(status="ready", compression="zstd", fallback_reason="target_mem_available_below_8GiB"), None
        if not self.verified(basis):
            return dict(status="ready", compression="zstd", fallback_reason="target_basis_has_no_current_verified_receipt"), None
        return dict(status="ready", compression="zstd_patch", basis_relative_path=rel, basis_sha256=basis["sha256"]), basis

    def decode(self, row, mode, basis):
        path = self.path(row["relative_path"])
        temp = self.temporary / (hashlib.sha256(row["relative_path"].encode()).hexdigest() + ".part")
        if temp.exists():
            if temp.is_symlink() or not temp.is_file() or temp.stat().st_uid != os.getuid():
                raise ValueError("Unexpected temporary owner/type")
            temp.unlink()
        if path.parent.stat().st_dev != self.temporary.stat().st_dev:
            raise ValueError("Temporary and destination filesystems differ; atomic commit unavailable")
        command = ["nice", "-n", "19", "ionice", "-c", "3", self.zstd, "-q", "-d"]
        basis_path, basis_stat = None, None
        if mode == "zstd_patch":
            basis_path = self.path(basis["relative_path"]); basis_stat = signature(basis_path)
            command += ["--patch-from=" + str(basis_path), "--long=31"]
        command += ["-c"]
        count, hashed, compressed = 0, hashlib.sha256(), 0
        state = dict(error=None, stderr=bytearray())
        with temp.open("xb") as output:
            decoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def consume():
                nonlocal count
                try:
                    for block in iter(lambda: decoder.stdout.read(FRAME_BYTES), b""):
                        count += len(block)
                        if count > row["bytes"]:
                            raise ValueError("Decoded asset exceeds approved size")
                        hashed.update(block); output.write(block)
                except BaseException as error:
                    state["error"] = repr(error)
                    if decoder.poll() is None:
                        decoder.kill()

            def consume_error():
                for block in iter(lambda: decoder.stderr.read(8192), b""):
                    if len(state["stderr"]) < 65536:
                        state["stderr"].extend(block[:65536 - len(state["stderr"])])

            reader, errors = threading.Thread(target=consume), threading.Thread(target=consume_error)
            reader.start(); errors.start(); last = time.monotonic()
            try:
                while True:
                    size = struct.unpack("!I", exact(self.input, 4))[0]
                    if not size:
                        break
                    if size > FRAME_BYTES:
                        raise ValueError("Invalid transfer frame size")
                    compressed += size
                    if compressed > row["bytes"] * 2 + FRAME_BYTES:
                        raise ValueError("Compressed asset exceeds transfer bound")
                    decoder.stdin.write(exact(self.input, size))
                    if time.monotonic() - last >= 20:
                        self.progress(row["relative_path"], compressed, compression=mode, basis=basis["relative_path"] if basis else None)
                        last = time.monotonic()
                decoder.stdin.close(); code = decoder.wait(); reader.join(); errors.join()
                if code or state["error"]:
                    raise ValueError("Decompression failed: " + str(state["error"]) + state["stderr"].decode(errors="replace"))
            finally:
                if decoder.poll() is None:
                    decoder.kill()
                decoder.wait(); reader.join(); errors.join()
                for handle in (decoder.stdin, decoder.stdout, decoder.stderr):
                    if not handle.closed:
                        handle.close()
            output.flush(); os.fsync(output.fileno())
        if count != row["bytes"] or hashed.hexdigest() != row["sha256"]:
            raise ValueError("Transferred asset digest/size differs")
        if basis and (not self.verified(basis) or signature(basis_path) != basis_stat):
            raise ValueError("Verified patch basis changed during decoding")
        os.replace(temp, path); os.chmod(path, 0o600)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self.record(row, reused=False, compressed=compressed, compression=mode,
                           basis=basis["relative_path"] if basis else None)

    def serve(self):
        self.progress()
        while True:
            line = self.input.readline(16384)
            if not line:
                break
            row = None
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Oversized or incomplete asset header")
                header = json.loads(line)
                row = self.allowed.get(header.get("relative_path"))
                if (row is None or type(header.get("bytes")) is not int or header["bytes"] != row["bytes"]
                        or header.get("sha256") != row["sha256"]):
                    raise ValueError("Asset outside approved manifest")
                path = self.path(row["relative_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                # Validate patch request even if this output already exists.
                ready, basis = self.negotiate(header, row)
                if self.verified(row) or (path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]):
                    receipt = self.record(row, reused=True)
                    self.progress(); self.reply(dict(receipt, status="already_verified")); continue
                # All negotiation, basis receipt and memory checks precede frame1.
                self.reply(ready); self.progress(row["relative_path"], compression=ready["compression"], basis=basis["relative_path"] if basis else None)
                receipt = self.decode(row, ready["compression"], basis)
                self.progress(); self.reply(receipt)
            except BaseException as error:
                self.progress(row["relative_path"] if row else None, error=repr(error))
                self.reply(dict(status="failed", error=repr(error)))
                raise
        self.progress()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-dir", default=DEFAULT_M)
    parser.add_argument("--zstd", default=DEFAULT_ZSTD)
    parser.add_argument("--temporary-root", default=DEFAULT_TMP)
    args = parser.parse_args()
    # Forced-key configuration remains a separate root-controlled action.
    migration = Path(args.migration_dir)
    lock = (migration / "asset_receive.lock").open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        Receiver(migration, args.temporary_root, args.zstd, sys.stdin.buffer, sys.stdout.buffer).serve()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
