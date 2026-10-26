"""Candidate v2 sender. Import is inert; only main starts transport.

Normal requests retain v1 wire compatibility. Patch bases must already have an
exact v2 verified ACK in this connection/replay pass. Private key contents are
never read by Python, copied, removed, or modified by this program.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time

from transport_v2 import (FRAME_BYTES, SOURCE_PATCH_MIN_MEMORY, choose_basis,
                          completion_ack, compression_command, frame, immutable_source,
                          load_manifest, mem_available, verified_ack)

DEFAULT_M = "/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027"
DEFAULT_ZSTD = "/home/ps/anaconda3/envs/SPNv2/bin/zstd"


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-dir", default=DEFAULT_M)
    parser.add_argument("--zstd", default=DEFAULT_ZSTD)
    parser.add_argument("--enable-patch", action="store_true", help="Use only previously verified, manifest-bound bases")
    args = parser.parse_args()
    migration = Path(args.migration_dir)
    manifest, allowed = load_manifest(migration / "asset_transfer_manifest.json")
    nonce = os.environ["SPNV2_TRANSFER_NONCE"]
    lock = (migration / "asset_send.lock").open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Shares v1 ownership exclusion.
    os.nice(19)
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    proc = Path("/proc") / str(os.getpid())
    raw_stat = (proc / "stat").read_text()
    identity = dict(pid=os.getpid(), starttime=int(raw_stat[raw_stat.rfind(")") + 2:].split()[19]),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    cmdline_sha256=hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest(), nonce=nonce)
    start = time.monotonic()
    state = dict(schema="spnv2-asset-send-v2", identity=identity, started_at=time.time(), phase="starting",
                 verified_files=0, verified_raw_bytes=0, compressed_bytes_sent=0, active=None,
                 expected_files=len(allowed), expected_raw_bytes=manifest["raw_total_bytes"],
                 patch_enabled=args.enable_patch, patch_files=0, normal_files=0,
                 model_forwards=0, cost_settlement_pending=True)

    def save(**values):
        state.update(values, updated_at=time.time(), elapsed_seconds=time.monotonic() - start)
        atomic(migration / "asset_send_v2_status.json", state)

    key, known_hosts = migration / "transfer_keys/migration_ed25519", migration / "transfer_keys/known_hosts"
    ssh_args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "UserKnownHostsFile=" + str(known_hosts),
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-i", str(key),
                "-p", "12611", "ps@1.tcp.vip.cpolar.top"]
    connection = None
    verified, sequence = {}, 0
    log = (migration / "asset_ssh_v2.log").open("ab")

    def ack():
        raw = connection.stdout.readline(16384)
        if not raw or not raw.endswith(b"\n"):
            raise EOFError("Receiver ended or sent an oversized acknowledgement; inspect asset_ssh_v2.log")
        value = json.loads(raw)
        if value.get("status") == "failed":
            raise ValueError("Receiver rejected asset: " + value.get("error", ""))
        return value

    try:
        save(phase="transferring")
        for row in manifest["files"]:
            save(active=row["relative_path"], active_raw_bytes=row["bytes"], active_compressed_bytes=0)
            for attempt in range(1, 4):
                try:
                    if connection is None:
                        connection = subprocess.Popen(ssh_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
                    path = immutable_source(row)
                    basis = None
                    available = mem_available() if args.enable_patch else 0
                    if args.enable_patch and available >= SOURCE_PATCH_MIN_MEMORY:
                        basis = choose_basis(row, verified, allowed)
                    header = dict(relative_path=row["relative_path"], bytes=row["bytes"], sha256=row["sha256"])
                    if basis:
                        header.update(compression="zstd_patch", basis_relative_path=basis["relative_path"], basis_sha256=basis["sha256"])
                    connection.stdin.write((json.dumps(header) + "\n").encode()); connection.stdin.flush()
                    reply = ack()
                    if reply["status"] == "already_verified":
                        if not completion_ack(row, reply):
                            raise ValueError("Skipped asset has no exact raw-SHA acknowledgement")
                        immutable_source(row)
                        break
                    if reply["status"] != "ready":
                        raise ValueError("Unexpected receiver state")
                    compression = reply.get("compression", "zstd")
                    if compression not in ("zstd", "zstd_patch") or (compression == "zstd_patch" and basis is None):
                        raise ValueError("Receiver negotiated an unrequested compression mode")
                    if compression == "zstd_patch":
                        if reply.get("basis_relative_path") != basis["relative_path"] or reply.get("basis_sha256") != basis["sha256"]:
                            raise ValueError("Receiver patch basis differs")
                        basis_path = immutable_source(basis)
                    else:
                        basis_path = None
                    save(active_compression=compression, active_basis=basis["relative_path"] if basis_path else None,
                         source_mem_available=available, fallback_reason=reply.get("fallback_reason"))
                    encoder = subprocess.Popen(compression_command(args.zstd, path, basis_path), stdout=subprocess.PIPE, stderr=log)
                    compressed, last = 0, time.monotonic()
                    try:
                        while True:
                            block = encoder.stdout.read(FRAME_BYTES)
                            if not block:
                                break
                            frame(connection.stdin, block); connection.stdin.flush()
                            compressed += len(block); state["compressed_bytes_sent"] += len(block)
                            if time.monotonic() - last >= 20:
                                save(active_compressed_bytes=compressed); last = time.monotonic()
                        if encoder.wait() != 0:
                            raise ValueError("Compression process failed")
                        immutable_source(row)
                        if basis is not None and basis_path is not None:
                            immutable_source(basis)
                        frame(connection.stdin, b""); connection.stdin.flush()
                        reply = ack()
                        if reply.get("status") != "verified" or not completion_ack(row, reply):
                            raise ValueError("No verified receiver receipt")
                        immutable_source(row)
                        if basis is not None and basis_path is not None:
                            immutable_source(basis)
                        state["patch_files" if compression == "zstd_patch" else "normal_files"] += 1
                    finally:
                        if encoder.poll() is None:
                            encoder.terminate(); encoder.wait()
                    break
                except (BrokenPipeError, EOFError, ConnectionError, OSError) as error:
                    if connection is not None:
                        if connection.poll() is None:
                            connection.terminate()
                        connection.wait(); connection = None
                    if attempt == 3:
                        raise
                    save(phase="waiting_transport", transport_error=repr(error), retry_number=attempt, next_retry_seconds=300)
                    time.sleep(300); save(phase="transferring")
            if verified_ack(row, reply):
                sequence += 1; verified[row["relative_path"]] = sequence
            state["verified_files"] += 1; state["verified_raw_bytes"] += row["bytes"]
            save(phase="transferring", active=None)
            print("VERIFIED", state["verified_files"], state["verified_raw_bytes"], row["relative_path"], flush=True)
        if connection is not None:
            connection.stdin.close()
            if connection.wait() != 0:
                raise ValueError("Receiver session exit differs from zero")
        save(phase="complete", active=None)
        print("ALL_ASSETS_VERIFIED", state["verified_files"], state["verified_raw_bytes"], flush=True)
    except BaseException as error:
        save(phase="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if connection is not None and connection.poll() is None:
            connection.terminate(); connection.wait()
        log.close(); lock.close()


if __name__ == "__main__":
    main()
