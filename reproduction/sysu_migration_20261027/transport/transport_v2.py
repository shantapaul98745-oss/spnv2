"""Shared stdlib-only contracts for lossless, manifest-bound transport v2."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import struct

FRAME_BYTES = 1048576
SOURCE_PATCH_MIN_MEMORY = 12 << 30
TARGET_PATCH_MIN_MEMORY = 8 << 30
SCIENCE = re.compile(r"spnv2_runs/plans/odr003_round2_20261023/runs/g50p2_det_20261024_p_(lightbox|sunlamp)_s(2021|2022|2023)_N1024_B4/durable/checkpoints/generation-\d{20}/state\.bin$")
PAIR = re.compile(r"spnv2_runs/plans/g50p_risk_diagnostic_20261025/pairs/g50p_diag_v1_(lightbox|sunlamp)_s(2021|2022|2023)_w\d{4}(?:_[A-Za-z0-9_]+)?/(candidate_learner|matched_scientific_state)\.bin$")
HEX = re.compile(r"[0-9a-f]{64}$")


def artifact_family(relative_path):
    """N state + matched scientific state share science; learner is separate."""
    match = SCIENCE.fullmatch(relative_path)
    if match:
        return "science", match[1], int(match[2])
    match = PAIR.fullmatch(relative_path)
    if match:
        return ("candidate" if match[3] == "candidate_learner" else "science", match[1], int(match[2]))
    return None


def relative(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid manifest relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("Manifest path is not canonical and relative")
    return value


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    rows = value["files"]
    allowed = {}
    for row in rows:
        rel = relative(row["relative_path"])
        if rel in allowed or type(row["bytes"]) is not int or row["bytes"] < 0 or not HEX.fullmatch(row["sha256"]):
            raise ValueError("Invalid or duplicate manifest asset")
        allowed[rel] = row
    if sum(row["bytes"] for row in rows) != value["raw_total_bytes"]:
        raise ValueError("Manifest raw total differs")
    return value, allowed


def signature(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def immutable_source(row):
    path = Path(row["source_path"])
    if path.is_symlink() or not path.is_file() or signature(path) != row["source_stat"] or row["source_stat"]["size"] != row["bytes"]:
        raise ValueError("Immutable source asset changed: " + row["relative_path"])
    return path


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable is unavailable; cannot admit patch memory")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(FRAME_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def exact(stream, count):
    blocks = []
    while count:
        block = stream.read(count)
        if not block:
            raise EOFError("Transfer interrupted")
        blocks.append(block)
        count -= len(block)
    return b"".join(blocks)


def frame(stream, block):
    if len(block) > FRAME_BYTES:
        raise ValueError("Invalid transfer frame size")
    stream.write(struct.pack("!I", len(block)))
    stream.write(block)


def compression_command(zstd, path, basis=None):
    command = ["nice", "-n", "19", "ionice", "-c", "3", str(zstd), "-q", "-T1", "-3"]
    if basis is None:
        command += ["--long=27"]
    else:
        command += ["--patch-from=" + str(basis), "--long=31"]
    return command + ["-c", str(path)]


def verified_ack(row, ack):
    """Only v2's fully bound ACK can establish an eligible dictionary basis."""
    stat = ack.get("file_stat")
    return (ack.get("status") in ("verified", "already_verified")
            and ack.get("relative_path") == row["relative_path"]
            and ack.get("sha256") == row["sha256"] and ack.get("bytes") == row["bytes"]
            and isinstance(stat, dict) and set(stat) == {"device", "inode", "size", "mtime_ns"}
            and all(type(v) is int and v >= 0 for v in stat.values()) and stat["size"] == row["bytes"])


def completion_ack(row, ack):
    """Every counted completion binds the requested raw SHA, including skips."""
    return (ack.get("status") in ("verified", "already_verified") and ack.get("sha256") == row["sha256"]
            and ("relative_path" not in ack or ack["relative_path"] == row["relative_path"])
            and ("bytes" not in ack or ack["bytes"] == row["bytes"]))


def choose_basis(row, verified, allowed):
    family = artifact_family(row["relative_path"])
    if family is None:
        return None
    choices = []
    for rel, sequence in verified.items():
        other = allowed[rel]
        group = artifact_family(rel)
        if rel != row["relative_path"] and group and group[0] == family[0]:
            immutable_source(other)
            choices.append((group[1:] == family[1:], sequence, rel))
    return allowed[max(choices)[2]] if choices else None
