"""Versioned, fail-closed SYSU namespace construction; default is metadata only.

This is a candidate launcher, not a deployed controller.  The read-only source
mirror provides the historical absolute namespace.  Only the owned diagnostic
plan, four shared coordination lock files, and project temp/cache are writable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

SCHEMA = "spnv2-sysu-namespace-config-v1"
MIGRATION = "sysu_20261027"
LOGICAL_ROOT = "/home/DATA/Downloads/UsrSourceCode/LuXinghai"
TARGET_ROOT = "/home/ps/Downloads/UsrSourceCode/LuXinghai"
DIAGNOSTIC = "spnv2_runs/plans/g50p_risk_diagnostic_20261025"
LOCK_PLANS = ("rebaseline_20261017", "odr003_exploratory_20261023",
              "odr003_round2_20261023", "g50p_risk_diagnostic_20261025")
READ_ONLY = ("implementation", "tasks", "revisions", "panels", "inputs", "A",
             "cost_A", "validation", "preflight", "index.json", "frozen_inputs.json",
             "protocol.json", "budget/identity.json")
CODE_NAMES = {"namespace_runner", "bootstrap", "dispatch_config", "storage_adapter", "storage_policy"}
INITIALIZATION_NAME = "efficientnet_b3_rwightman-cf984f9c.pth"
INITIALIZATION_SHA256 = "b3899882250c22946d0229d266049fcd133c169233530b36b9ffa7983988362f"
HEX = re.compile(r"[0-9a-f]{64}$")
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class MigrationConfigError(ValueError):
    pass


def require(value, message):
    if not value:
        raise MigrationConfigError(message)


def absolute(value):
    require(isinstance(value, str) and "\x00" not in value, "Invalid path")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts,
            "Paths must be canonical absolute POSIX paths")
    return path


def metadata(path, max_bytes=8 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
            "Expected bounded regular metadata: " + str(path))
    def reject(value):
        raise MigrationConfigError("Nonfinite metadata: " + value)
    return json.loads(path.read_text(encoding="utf8"), parse_constant=reject)


def sha(path, max_bytes=128 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes,
            "Expected bounded regular binding: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_image_mounts(config):
    """Optional project-owned exact-byte cache; shared dataset is never edited."""
    cache = config.get("frozen_image_cache")
    if cache is None:
        return []
    require(isinstance(cache, dict) and set(cache) == {"source", "target", "manifest", "receipt"},
            "Frozen image cache requires explicit source/target/manifest/receipt")
    p = config["paths"]
    source = p["target_root"] + "/spnv2_runs/frozen/required_images_20261027/sunlamp/images_768x512_RGB"
    domain = p["logical_dataset"] + "/sunlamp"
    require(str(absolute(cache["source"])) == source and cache["target"] == domain + "/images_768x512_RGB",
            "Frozen image cache must preserve the exact approved sunlamp derived-image path")
    return [(False, p["dataset"] + "/sunlamp/images", domain + "/images"),
            (False, p["dataset"] + "/sunlamp/test.json", domain + "/test.json"),
            (False, source, cache["target"])]


def initialization_cache_mounts(config):
    item = config.get("initialization_cache")
    if item is None:
        return []
    require(isinstance(item, dict) and set(item) == {"source", "target", "sha256", "receipt"},
            "Initialization cache requires source/target/frozen SHA/receipt")
    source = config["paths"]["source_snapshot"] + "/spnv2_runs/frozen/sysu_external_cache/" + INITIALIZATION_NAME
    target = "/home/ps/.cache/torch/hub/checkpoints/" + INITIALIZATION_NAME
    require(item["source"] == source and item["target"] == target and item["sha256"] == INITIALIZATION_SHA256,
            "Initialization cache must be the unchanged frozen artifact at its original torch path")
    return [(False, source, target)]


def mounts(config, scientific=False, manager=False):
    """Return ordered (writable, physical source, logical target) bind tuples."""
    p = config["paths"]
    root, plan = p["logical_root"], p["logical_root"] + "/" + DIAGNOSTIC
    source_plan = p["source_snapshot"] + "/" + DIAGNOSTIC
    result = [(False, p["source_snapshot"], root),
              (False, p["repository"], root + "/spnv2"),
              (False, p["environment"], p["environment"]),
              (False, p["environment"], p["logical_environment"]),
              (False, p["dataset"], p["logical_dataset"]),
              (False, p["dataset"], p["dataset"]),
              *frozen_image_mounts(config),
              *initialization_cache_mounts(config),
              (False, p["control_root"], p["logical_control_root"])]
    if manager:
        require(scientific, "A metadata manager uses an explicitly admitted deployment namespace")
        result.append((True, p["control_root"] + "/config_versions", p["logical_control_root"] + "/config_versions"))
    if scientific:
        result.append((True, source_plan, plan))
        result.extend((False, source_plan + "/" + name, plan + "/" + name)
                      for name in READ_ONLY)
        # All historical task files remain RO; only the new technical revision
        # is writable. Accepted pair output directories are protected mounts.
        revision = config["new_task_revision"]
        result.append((True, source_plan + "/tasks/" + revision,
                       plan + "/tasks/" + revision))
        result.extend((False, source_plan + "/pairs/" + name, plan + "/pairs/" + name)
                      for name in config["accepted_run_ids"])
        for name in LOCK_PLANS:
            result.append((True, p["target_root"] + "/spnv2_runs/plans/" + name + "/pipeline.lock",
                           root + "/spnv2_runs/plans/" + name + "/pipeline.lock"))
    result.extend((True, p[name], p[name]) for name in ("tmp", "cache"))
    return result


def physical(config, logical, *, inside_namespace=False, scientific=False):
    logical = str(absolute(logical))
    if inside_namespace:
        return Path(logical)
    # Overlay order matters when a nested bind has the same target specificity.
    choices = [(len(target), position, source, target)
               for position, (_, source, target) in enumerate(mounts(config, scientific))
               if logical == target or logical.startswith(target + "/")]
    require(choices, "Binding outside the explicit namespace")
    _, _, source, target = max(choices)
    return Path(source + logical[len(target):])


def binding(config, value, *, inside_namespace=False, scientific=False):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and HEX.fullmatch(value["sha256"] or ""), "Invalid immutable binding")
    path = physical(config, value["path"], inside_namespace=inside_namespace, scientific=scientific)
    require(sha(path) == value["sha256"], "Immutable binding changed: " + value["path"])
    return path


def check_budget_journal(config, handoff, *, inside_namespace=False):
    """Keep the final source journal as an immutable prefix as target costs grow."""
    item = config["budget_journal"]
    require(set(item) == {"path", "sha256"} and HEX.fullmatch(item["sha256"] or ""),
            "Invalid final budget journal binding")
    path = physical(config, item["path"], inside_namespace=inside_namespace, scientific=True)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128 << 20,
            "Expected bounded budget journal metadata")
    raw = path.read_bytes()
    length = handoff.get("budget_journal_bytes")
    require(type(length) is int and 0 < length <= len(raw)
            and raw[:length].endswith(b"\n")
            and hashlib.sha256(raw[:length]).hexdigest() == item["sha256"],
            "Final source journal prefix was removed, changed, or refunded")
    previous, attempts, seen_attempts, reserved, consumed = "0" * 64, set(), set(), 0, 0
    for sequence, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line.endswith(b"\n"), "Torn current budget journal")
        row = json.loads(line)
        digest = row.pop("sha256")
        canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        require(row["sequence"] == sequence and row["previous_sha256"] == previous
                and hashlib.sha256(canonical.encode()).hexdigest() == digest,
                "Current budget journal checksum/sequence differs")
        previous = digest
        payload, kind = row["payload"], row["kind"]
        if kind == "attempt_started":
            require(payload["attempt_id"] not in seen_attempts, "Duplicate budget attempt")
            seen_attempts.add(payload["attempt_id"])
            attempts.add(payload["attempt_id"])
        elif kind == "attempt_closed":
            require(payload["attempt_id"] in attempts, "Unknown or duplicate budget close")
            attempts.remove(payload["attempt_id"])
        elif kind == "forward_reserved":
            require(type(payload["count"]) is int and payload["count"] > 0, "Invalid physical reservation")
            reserved += payload["count"]
        elif kind == "initial_cost_imported":
            require(type(payload["physical_forwards_upper"]) is int and payload["physical_forwards_upper"] >= 0,
                    "Invalid historical validation cost")
            reserved += payload["physical_forwards_upper"]
        consumed += len(line)
        if consumed == length and "snapshot" in handoff:
            require(not attempts and reserved == handoff["snapshot"]["physical_forwards_reserved_upper"],
                    "Final source budget snapshot does not match the immutable journal prefix")
    require(not attempts and reserved <= 40000, "Current budget has an unresolved attempt or exceeds the physical cap")
    return path


def check_asset_layout(config, *, inside_namespace=False):
    """Canonical U layout is admitted only by a bound no-content-change receipt."""
    p = config["paths"]
    if p["source_snapshot"] != p["target_root"]:
        return
    require(p["target_root"] == TARGET_ROOT, "Canonical assets must remain under the exact approved target U")
    require(isinstance(config.get("asset_layout_receipt"), dict)
            and isinstance(config.get("asset_transfer_manifest"), dict),
            "Canonical layout requires explicit receipt and transfer-manifest bindings")
    receipt = metadata(binding(config, config["asset_layout_receipt"], inside_namespace=inside_namespace))
    manifest_binding = config["asset_transfer_manifest"]
    manifest = metadata(binding(config, manifest_binding, inside_namespace=inside_namespace))
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) > 0,
            "Canonical asset layout requires the actual bound transfer manifest")
    if receipt.get("schema") == "spnv2-streaming-canonical-asset-layout-v1":
        require(receipt.get("status") == "verified_metadata_with_incremental_immutable_assets"
                and receipt.get("target_root") == TARGET_ROOT
                and receipt.get("source_staging") == TARGET_ROOT + "/spnv2_runs/migrations/" + MIGRATION + "/source_snapshot"
                and receipt.get("transfer_manifest_sha256") == manifest_binding["sha256"]
                and receipt.get("no_file_content_changes") is True
                and receipt.get("staging_paths_preserved") is True
                and receipt.get("source_dispatch_fenced") is True
                and receipt.get("dependencies") == config.get("streaming_migration", {}).get("dependencies"),
                "Streaming canonical receipt is incomplete or for another immutable dependency plan")
        return
    require(receipt.get("schema") == "spnv2-canonical-asset-layout-v1"
            and receipt.get("status") == "verified_renamed_immutable_assets"
            and receipt.get("target_root") == TARGET_ROOT
            and receipt.get("source_staging") == TARGET_ROOT + "/spnv2_runs/migrations/" + MIGRATION + "/source_snapshot"
            and receipt.get("transfer_manifest_sha256") == manifest_binding["sha256"]
            and type(receipt.get("verified_file_count")) is int and receipt["verified_file_count"] == len(files)
            and receipt.get("no_file_content_changes") is True
            and receipt.get("source_dispatch_fenced") is True,
            "Canonical asset rename receipt is missing, incomplete, or for another layout")


def check_cuda_namespace_fix(config, *, inside_namespace=False):
    item=config.get('cuda_namespace_fix')
    require(item is not None,'Explicit CUDA proc namespace revision required')
    revision=metadata(binding(config,item,inside_namespace=inside_namespace,scientific=True))
    index=metadata(binding(config,config['diagnostic_index'],inside_namespace=inside_namespace,scientific=True))
    require(index.get('cuda_namespace_fix')==item and revision.get('schema')=='spnv2-cuda-proc-namespace-revision-v1'
            and revision.get('technical_repair_only') is True,'Wrong CUDA namespace revision')
    require(revision['before_index']['sha256']=='8de2d8c94004f876fcf9b2d8344b9fa69f19db057d93b7ed6bfc0df16073bf04','CUDA namespace baseline changed')
    before=metadata(binding(config,revision['before_index'],inside_namespace=inside_namespace,scientific=True))
    require({k:v for k,v in index.items() if k not in ('migration_extensions','cuda_namespace_fix')}==
            {k:v for k,v in before.items() if k!='migration_extensions'},'CUDA proc repair changed task/science/resource/budget bindings')
    require(before['migration_extensions']==revision['before_extensions'] and index['migration_extensions']==revision['after_extensions'],'CUDA extension revision differs')
    require(revision['mount_delta']==['--bind','/proc','/proc'] and revision['user_pid_uid_and_other_mounts_unchanged'] is True,'CUDA mount delta is not exact')
    probe=metadata(binding(config,revision['own_comm_validation'],inside_namespace=inside_namespace,scientific=True))
    cases={r['case']:r for r in probe['results']}
    require(probe['status']=='passed' and probe['all_other_mounts_unchanged'] is True and probe['no_torch_or_cuda_or_model_probe'] is True
            and probe['model_forwards']==0 and cases['original_readonly_proc']['result']['errno']==30
            and cases['private_native_proc_rw_bind']['result']['write_ok'] is True
            and all(row['actual_exit_code']==0 and row['result']['comm_unchanged'] is True
                    and row['result']['command_identity_unchanged'] is True and row['result']['foreign_process_written'] is False for row in cases.values()),
            'Missing exact own-comm metadata validation')


def validate_config(config, *, scientific=False, inside_namespace=False):
    require(config.get("schema") == SCHEMA and config.get("migration_id") == MIGRATION,
            "Wrong migration config schema/identity")
    p = config["paths"]
    required = {"target_root", "source_snapshot", "repository", "environment",
                "logical_environment", "dataset", "logical_dataset", "tmp", "cache",
                "logical_root", "control_root", "logical_control_root"}
    require(set(p) == required, "Missing or unknown namespace path")
    for path in p.values():
        absolute(path)
    require(p["logical_root"] == LOGICAL_ROOT, "Historical root must remain exact")
    require(p["repository"] == p["target_root"] + "/spnv2", "Repository target differs")
    migration = p["target_root"] + "/spnv2_runs/migrations/" + MIGRATION
    require(p["source_snapshot"] in (migration + "/source_snapshot", p["target_root"])
            and p["control_root"] == migration + "/runtime_control"
            and p["logical_control_root"] == LOGICAL_ROOT + "/spnv2_runs/migrations/" + MIGRATION + "/runtime_control",
            "Source/control roots differ from this isolated migration")
    if p["source_snapshot"] == p["target_root"]:
        require(p["target_root"] == TARGET_ROOT, "Canonical assets must remain under the exact approved target U")
    require(p["logical_environment"] == "/home/ps/anaconda3/envs/SPNv2"
            and p["logical_dataset"] == "/home/DATA/Downloads/DataSet/speedplus",
            "Historical environment/data aliases differ")
    require(p["environment"] == "/home/ps/miniconda3/envs/SPNv2"
            and p["dataset"] == "/home/ps/Downloads/DataSet/speedplus",
            "Unapproved target environment/data")
    require(p["tmp"].startswith("/tmp/spnv2_sysu_")
            and p["cache"].startswith("/home/ps/.cache/spnv2_sysu_"),
            "Only project-specific temp/cache may be writable")
    screen = config.get("screen_socket_dir")
    require(screen == p["tmp"] + "/screen", "Explicit project screen_socket_dir must be TMP/screen")
    screen_path = Path(screen)
    require(screen_path.is_dir() and screen_path.resolve() == screen_path.absolute(),
            "Project screen socket directory must pre-exist without symlink aliases")
    screen_stat = screen_path.stat()
    require(screen_stat.st_uid == os.getuid() == 1000 and stat.S_IMODE(screen_stat.st_mode) == 0o700,
            "Project screen socket directory must be owned by UID1000 with mode0700")
    require(set(config["code_bindings"]) == CODE_NAMES, "Every extension file must be explicitly bound")
    package = LOGICAL_ROOT + "/spnv2/reproduction/"
    expected_names = {"namespace_runner": package + "shared_gpu_20261027/cuda_namespace_fix_20261027/namespace_runner.py",
                      "bootstrap": package + "shared_gpu_20261027/cuda_namespace_fix_20261027/bootstrap.py",
                      "dispatch_config": package + "shared_gpu_20261027/cuda_namespace_fix_20261027/dispatch_config.py",
                      "storage_adapter": package + "sysu_migration_20261027/storage_adapter_v1.py",
                      "storage_policy": package + "storage_management_20261027/storage_policy.py"}
    for name, item in config["code_bindings"].items():
        require(item["path"] == expected_names[name], "Unexpected extension path")
        binding(config, item, inside_namespace=inside_namespace)
    frozen_image_mounts(config)  # Validate optional paths even for a probe argv.
    initialization_cache_mounts(config)
    if not scientific:
        return config
    check_asset_layout(config, inside_namespace=inside_namespace)
    require(config.get("source_dispatch_fenced") is True,
            "Scientific execution requires an explicit completed source fence")
    require(config.get("target_scientific_dispatch_authorized") is True,
            "Target scientific dispatch has not been explicitly authorized")
    fence = metadata(binding(config, config["source_fence_receipt"], inside_namespace=inside_namespace))
    require(fence.get("schema") == "spnv2-source-dispatch-fenced-v1"
            and fence.get("status") == "source_dispatch_fenced"
            and fence.get("nonce") == config["source_fence_nonce"]
            and isinstance(fence.get("nonce"), str) and len(fence["nonce"]) == 32
            and fence.get("no_scientific_workers_interrupted") is True
            and fence.get("guardian", {}).get("status") == "holding_source_dispatch_fence"
            and set(fence["guardian"].get("holds", [])) == {"diagnostic_pipeline", "diagnostic_manager"},
            "Missing immutable proof of source dispatch fencing")
    handoff = metadata(binding(config, config["budget_handoff_receipt"], inside_namespace=inside_namespace))
    require(handoff.get("schema") == "spnv2-sysu-budget-handoff-v1"
            and handoff.get("migration_id") == MIGRATION and handoff.get("status") == "final"
            and handoff.get("source_fence_receipt") == config["source_fence_receipt"]
            and handoff.get("budget_identity") == config["budget_identity"]
            and handoff.get("budget_journal") == config["budget_journal"]
            and handoff.get("diagnostic_index") == config["diagnostic_index"]
            and handoff.get("budget_journal_bytes") == fence.get("source_budget_journal_bytes")
            and config["budget_journal"]["sha256"] == fence.get("source_budget_journal_sha256"),
            "Final budget handoff is missing or does not bind this source fence/ledger")
    snapshot = handoff["snapshot"]
    require(snapshot.get("schema") == "g50p-diagnostic-budget-snapshot-v1"
            and snapshot == fence.get("source_budget")
            and snapshot.get("unresolved_attempt_ids") == []
            and snapshot.get("time_hard_stop_enabled") is False
            and snapshot.get("max_active_seconds") is None
            and type(snapshot.get("physical_forwards_reserved_upper")) is int
            and 0 <= snapshot["physical_forwards_reserved_upper"] <= 40000,
            "Budget handoff is unresolved, out of bounds, or re-enables the old time stop")
    plan = LOGICAL_ROOT + "/" + DIAGNOSTIC
    require(config["budget_identity"]["path"] == plan + "/budget/identity.json"
            and config["budget_journal"]["path"] == plan + "/budget/journal.jsonl",
            "Budget handoff must retain the original stable budget paths")
    require(config["diagnostic_index"]["path"] == plan + "/index.json", "Diagnostic index path differs")
    for name in ("diagnostic_index", "budget_identity", "effective_protocol", "budget_policy"):
        binding(config, config[name], inside_namespace=inside_namespace, scientific=True)
    check_budget_journal(config, handoff, inside_namespace=inside_namespace)
    check_cuda_namespace_fix(config,inside_namespace=inside_namespace)
    require(config.get('resource_revision') is not None and config.get('resource_policy') is not None,'Explicit shared resource revision required')
    current_index=metadata(binding(config,config['diagnostic_index'],inside_namespace=inside_namespace,scientific=True))
    require(current_index.get('resource_revision')==config['resource_revision'] and current_index.get('resource_policy')==config['resource_policy'],'Resource config/index binding differs')
    resource=metadata(binding(config,config['resource_revision'],inside_namespace=inside_namespace,scientific=True))
    code=resource['validator']
    require(code['path']==LOGICAL_ROOT+'/spnv2/reproduction/shared_gpu_20261027/resource_amendment.py','Resource bridge path differs')
    import importlib.util
    module_path=binding(config,code,inside_namespace=inside_namespace,scientific=True)
    module_spec=importlib.util.spec_from_file_location('spnv2_ns_resource_bridge',module_path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    resolve=None if inside_namespace else lambda value: physical(config,value,scientific=True)
    module.load(current_index,resolve)
    require(snapshot.get("budget_policy") == config["budget_policy"], "Budget policy binding differs")
    require(config.get("initialization_cache") is not None, "Explicit frozen initialization cache is required")
    initialization = config["initialization_cache"]
    initial_receipt = metadata(binding(config, initialization["receipt"], inside_namespace=inside_namespace))
    initial_path = Path(initialization["target"]) if inside_namespace else Path(initialization["source"])
    initial_stat = initial_path.stat()
    require(initial_receipt.get("status") == "verified"
            and initial_receipt.get("relative_path") == "spnv2_runs/frozen/sysu_external_cache/" + INITIALIZATION_NAME
            and initial_receipt.get("sha256") == INITIALIZATION_SHA256 and initial_receipt.get("bytes") == 49475909
            and initial_receipt.get("file_stat") == dict(device=initial_stat.st_dev, inode=initial_stat.st_ino,
                size=initial_stat.st_size, mtime_ns=initial_stat.st_mtime_ns),
            "Frozen initialization-cache transfer receipt/stat binding differs; no weight re-read is performed")
    cache = config.get("frozen_image_cache")
    if cache is not None:
        manifest = metadata(binding(config, cache["manifest"], inside_namespace=inside_namespace))
        receipt = metadata(binding(config, cache["receipt"], inside_namespace=inside_namespace))
        require(manifest.get("schema") == "spnv2-required-image-spec-v1"
                and manifest.get("image_count") == 2171 and manifest.get("source_total_bytes") == 61849173
                and HEX.fullmatch(manifest.get("required_image_list_sha256") or "")
                and isinstance(manifest.get("entries"), list) and len(manifest["entries"]) == 2171,
                "Frozen image cache content manifest differs from the approved remaining-window scope")
        require(receipt.get("schema") == "spnv2-required-image-cache-v1"
                and receipt.get("status") == "verified_exact_frozen_bytes"
                and receipt.get("manifest_sha256") == cache["manifest"]["sha256"]
                and receipt.get("required_image_list_sha256") == manifest.get("required_image_list_sha256")
                and receipt.get("cache_root") == cache["source"]
                and receipt.get("image_count") == manifest["image_count"]
                and receipt.get("total_bytes") == manifest["source_total_bytes"]
                and receipt.get("mismatch_count") == 0,
                "Exact-byte image-cache verification receipt is missing or does not bind the manifest")
    accepted = config["accepted_run_ids"]
    require(isinstance(accepted, list) and len(accepted) == len(set(accepted))
            and all(isinstance(name, str) and NAME.fullmatch(name) for name in accepted)
            and accepted == handoff.get("accepted_run_ids")
            and len(accepted) == fence.get("accepted_pairs"), "Accepted pair protection list differs")
    require(NAME.fullmatch(config["new_task_revision"]) is not None,
            "New task revision must be one canonical directory name")
    require(config.get("approved_tasks") and config.get("approved_stores") is not None,
            "Exact new tasks and storage approvals are required")
    for item in config["approved_tasks"]:
        require(PurePosixPath(item["path"]).parent == PurePosixPath(plan + "/tasks/" + config["new_task_revision"]),
                "Only tasks in the explicit new technical revision are allowed")
        task = metadata(binding(config, item, inside_namespace=inside_namespace, scientific=True))
        require(task.get("schema") == "g50p-risk-diagnostic-task-v1"
                and task.get("action") in ("run_pair", "verify_pair", "verify_pilots", "summarize_diagnostic"),
                "Migration cannot rerun the completed A extraction/analysis or introduce another action")
        if task["action"] == "run_pair":
            require(task["run_id"] not in accepted
                    and task.get("output_dir") == plan + "/pairs/" + task["run_id"],
                    "Migration cannot replay an accepted pair or redirect scientific output")
            require(sum(store["root"] == task["output_dir"] + "/isolated_durable"
                        for store in config["approved_stores"]) == 1,
                    "Every new pair needs one exact storage approval before scientific execution")
    for item in config["approved_stores"]:
        root = absolute(item["root"])
        require(root.parent.parent == PurePosixPath(plan + "/pairs")
                and root.name == "isolated_durable" and root.parent.name not in accepted,
                "Storage approval must name one unaccepted diagnostic pair store")
        require(isinstance(item.get("identity"), dict) and isinstance(item.get("policy"), dict),
                "Each store requires an exact original identity and explicit policy")
        identity, policy = item["identity"], item["policy"]
        require({"run_id", "source_sha256", "config_sha256", "manifest_sha256"} <= set(identity)
                and isinstance(identity["run_id"], str) and identity["run_id"]
                and all(isinstance(key, str) and key and isinstance(value, str) and value
                        for key, value in identity.items())
                and all(HEX.fullmatch(identity[name] or "") for name in ("source_sha256", "config_sha256", "manifest_sha256")),
                "Storage approval must retain the complete original scientific identity")
        require(policy.get("schema") == "spnv2-storage-policy-v1" and policy.get("retain_latest") == 2
                and type(policy.get("estimated_peak_bytes")) is int and policy["estimated_peak_bytes"] > 0
                and isinstance(policy.get("protected_generations"), dict),
                "Storage policy/latest-two retention and a positive peak estimate are mandatory")
    binding(config, config["original_durable_state"], inside_namespace=inside_namespace)
    require(config["original_durable_state"]["path"] == LOGICAL_ROOT + "/spnv2_runs/plans/odr003_round2_20261023/implementation/durable_state.py",
            "Only the unchanged original frozen durable module may be adapted")
    streaming = config.get("streaming_migration")
    if streaming is not None:
        expected = LOGICAL_ROOT + "/spnv2/reproduction/sysu_migration_20261027/streaming_assets_v1.py"
        require(streaming.get("code", {}).get("path") == expected,
                "Streaming helper must be the exact approved project module")
        code_path = binding(config, streaming["code"], inside_namespace=inside_namespace)
        import importlib.util
        spec = importlib.util.spec_from_file_location("spnv2_streaming_bound", code_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        index = metadata(binding(config, config["diagnostic_index"], inside_namespace=inside_namespace))
        require(index.get("streaming_migration") == streaming, "Task config/dependency index binding differs")
        # Native composition resolves logical paths itself; inside the namespace
        # the same bound helper repeats the actual receipt/stat gate before import.
        if inside_namespace:
            for item in config["approved_tasks"]:
                task = metadata(binding(config, item, inside_namespace=True, scientific=True))
                require(module.readiness(index, task)["ready"], "Task dependencies are not all verified/published")
    for _, source, target in mounts(config, True):
        path = Path(target) if inside_namespace else Path(source)
        require(path.exists() and path.absolute() == path.resolve(), "Missing or aliased mount: " + str(path))
    return config


def config_location(config, path):
    root = config["paths"]["logical_control_root"]
    path = str(absolute(path))
    require(path == root + "/runtime_config.json"
            or (PurePosixPath(path).parent == PurePosixPath(root + "/config_versions")
                and NAME.fullmatch(PurePosixPath(path).name) and path.endswith(".json")),
            "Runtime config must be one immutable version in the explicit control bundle")
    return path


def build_bwrap(argv, config, *, scientific=False, config_sha256, config_path=None, _manager=False):
    """Build an argv only. No subprocess, mounts, imports, or dispatch occurs."""
    validate_config(config, scientific=scientific)
    require(HEX.fullmatch(config_sha256 or ""), "Explicit config hash required")
    p = config["paths"]
    require(not argv if not scientific else isinstance(argv, (list, tuple)),
            "Probe mode accepts no arbitrary executable or model command")
    inner = [p["logical_environment"] + "/bin/python", "-B", "-s", "-u",
             config["code_bindings"]["bootstrap"]["path"], "--config",
             config_location(config, config_path or p["logical_control_root"] + "/runtime_config.json"),
             "--config-sha256", config_sha256]
    if scientific:
        inner += ["--execute-host-entry", LOGICAL_ROOT + "/" + DIAGNOSTIC + "/implementation/host_entry.py", *argv]
    else:
        inner += ["--probe"]
    result = ["bwrap", "--unshare-user", "--uid", "1000", "--gid", "1000",
              "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--bind", "/proc", "/proc", "--tmpfs", "/home"]
    for writable, source, target in mounts(config, scientific, manager=_manager):
        if config.get("frozen_image_cache") is not None and target == p["logical_dataset"] + "/sunlamp/images":
            # Mount a private domain directory first: bwrap can then create the
            # cache mountpoint without writing into the actual RO shared data.
            result += ["--tmpfs", p["logical_dataset"] + "/sunlamp"]
        result += ["--bind" if writable else "--ro-bind", source, target]
    result += ["--chdir", LOGICAL_ROOT + "/spnv2"]
    environment = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MPLBACKEND": "Agg",
                   "TMPDIR": p["tmp"], "XDG_CACHE_HOME": p["cache"], "MPLCONFIGDIR": p["cache"] + "/matplotlib"}
    environment["TORCH_HOME"] = "/home/ps/.cache/torch"
    environment["SCREENDIR"] = config["screen_socket_dir"]
    for name, value in environment.items():
        result += ["--setenv", name, value]
    return result + inner


def build_manager_bwrap(config, deployment_binding, config_binding):
    """Build only the bound diagnostic metadata manager, with no arbitrary argv."""
    require(config.get("dispatch_provenance", {}).get("deployment") == deployment_binding,
            "Manager requires a config composed from this exact deployment")
    deployment = metadata(binding(config, deployment_binding))
    require(deployment.get("schema") == "spnv2-sysu-deployment-v1"
            and deployment.get("migration_id") == MIGRATION
            and deployment.get("index") == config["diagnostic_index"]
            and deployment.get("template") == config["dispatch_provenance"].get("template"),
            "Manager deployment/template/index bindings differ")
    index = metadata(binding(config, config["diagnostic_index"], scientific=True))
    require(index.get("migration_extensions") == config["code_bindings"], "Manager static extension binding differs")
    plan = LOGICAL_ROOT + "/" + DIAGNOSTIC
    manager = dict(path=plan + "/implementation/manager.py", sha256=index["implementation_sha256"]["manager.py"])
    binding(config, manager, scientific=True)
    require(metadata(binding(config, config_binding, scientific=True)) == config,
            "Manager input differs from its immutable first-task config")
    versions = Path(config["paths"]["control_root"]) / "config_versions"
    require(versions.is_dir() and versions.resolve() == versions.absolute(), "Explicit config_versions directory is required")
    result = build_bwrap([], config, scientific=True, config_sha256=config_binding["sha256"],
                         config_path=config_binding["path"], _manager=True)
    python = config["paths"]["logical_environment"] + "/bin/python"
    start = result.index(python)
    return result[:start] + [python, "-B", "-s", "-u", manager["path"], "--index", config["diagnostic_index"]["path"],
        "--deployment-manifest", deployment_binding["path"], "--deployment-sha256", deployment_binding["sha256"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--scientific", action="store_true")
    parser.add_argument("--run-probe", action="store_true")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, "Config changed")
    config = metadata(args.config)
    argv = build_bwrap(args.entry_args, config, scientific=args.scientific, config_sha256=args.config_sha256)
    # Scientific mode only emits the gated candidate command. Live rollout is
    # deliberately a separate controller decision; this CLI cannot dispatch it.
    if args.run_probe:
        require(not args.scientific, "This candidate CLI runs metadata probes only")
        raise SystemExit(subprocess.run(argv, check=False).returncode)
    print(json.dumps({"experimental_dispatch": False, "namespace_argv": argv}, indent=2))


if __name__ == "__main__":
    main()
