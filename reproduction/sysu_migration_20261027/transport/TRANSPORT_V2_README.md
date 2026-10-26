# Lossless asset transport v2 candidates

No candidate is launched by importing these files. Active v1 programs, private
keys, forced-key commands, manifests and schedules are untouched. Root must
review and independently prove the required real decode, then switch at an
owned verified file boundary. Copy `transport_v2.py` beside both candidate
scripts; no third-party Python package or model import is used.

Sender flags: `--migration-dir` (default original source migration directory),
`--zstd` (default frozen source environment 1.5.7), and optional `--enable-patch`.
Without the last flag normal v1 headers/framing remain the default. The existing
`SPNV2_TRANSFER_NONCE` is required. Fixed SSH endpoint/key-path/host-key options
are preserved; Python does not read/copy/remove/modify private key contents.
The source main and compressor use nice19 and ionice idle. Sender owns the same
`asset_send.lock` as v1, and writes only its new `asset_send_v2_status.json`
(`spnv2-asset-send-v2`) and `asset_ssh_v2.log` plus ordinary stdout.
V2 counts every completed/skip ACK only with the exact requested raw SHA. An
old receiver's `already_verified` ACK without SHA is refused explicitly; it
cannot falsely complete a file or establish a basis. V1 senders can still use
the v2 receiver's normal mode and ignore its additional ACK metadata.

Receiver flags: `--migration-dir` (default target migration directory), `--zstd`
(default frozen target environment 1.5.7), `--temporary-root` (default existing
project `/tmp/.../asset-transfer`). It owns the same `asset_receive.lock` as v1.
It keeps compatible `asset_receipts/*.json` and `asset_receive_status.json`
(`sysu-asset-transfer-v1`, additional `transport_version:2`/mode fields).
Temporary and output filesystems must support the same atomic `os.replace`.

Patch header extends the original `relative_path,bytes,sha256` with
`compression:"zstd_patch",basis_relative_path,basis_sha256`. Every frame remains
a big-endian 4-byte length followed by at most 1MiB; zero length terminates.
The target still requires exact original raw byte count/SHA before publication.
The full decoded original file is stored; there is no persistent delta reference.
Manifest-approved full basis assets stay immutable and retained throughout the
transfer; transport never removes them.

Only a previous v2 `verified`/`already_verified` ACK containing the exact
manifest path/SHA/size and actual target file stat can make a basis eligible.
Existing complete files are replayed as headers and acknowledged, rebuilding
this map. Basis selection uses newest ACK, prefers the same domain+seed, and
otherwise falls back within the same family. Source N checkpoints and matched
scientific states are family `science`; candidate learners are `candidate`.
No other manifest file is eligible. Source immutable stat is checked before
encoding and after it; target basis receipt/stat is checked before decoding
and again before commit.

Source needs MemAvailable >=12GiB before requesting a patch. Target needs >=8GiB
and an exact current verified basis receipt. Missing/stale target basis or low
target memory replies `ready,compression:"zstd",fallback_reason` **before any
payload is consumed**; sender honors this before starting its encoder. Invalid
manifest/family/digest bases reject. Any error after payload starts aborts that
request/connection; normal bytes are never substituted within a patch stream.
Both patch directions use the environment zstd with `--patch-from` and
`--long=31` (2GiB decoder window). Default normal uses `--long=27`.

Tiny synthetic tests, run from this candidate directory only:

```sh
SPNV2_TEST_ZSTD=/home/ps/miniconda3/envs/SPNv2/bin/zstd PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_transport_v2 -v
```

No test reads a real checkpoint, imports Torch, runs inference or touches GT.
