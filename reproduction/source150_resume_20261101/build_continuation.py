"""Prepare the explicitly authorized 150-epoch continuation; never deploy it.

Run with the server stdlib Python. This program reads metadata/code only, writes
new candidate files exclusively, and never launches or imports a model. Root
must hold/recheck the original manager lock and prove no scientific worker is
active before publishing candidate_state.json as the new production state.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

EXPECTED_OLD = "1628c1fa396b7ac51c06d53c34763b95d81787857ae4d72c523e1b718805e487"
EXPECTED_FORMAL = "f53581b9add1460ae4f5582bcbda3c84d647c901ffd946950aa343183b60b4a2"
RETIRED = ["odr003_O_lightbox", "odr003_O_sunlamp", "odr003_shared_calibration"]
CONFIGS = ["table2_h", "table2_e", "table2_eh", "table3_base", "table3_erase",
           "table3_flare", "table4_phi0_bn", "table4_phi3_gn"]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def canonical(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive(p, value):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def expected_paths(task):
    return [Path(x["path"] if isinstance(x, dict) else x)
            for x in task.get("expected_outputs", [])]


def verify_completed(task):
    need(task.get("status") == "accepted", "Prior task is not accepted: " + task["id"])
    phases = {x["phase"]: x for x in task.get("history", []) + [task.get("runtime", {})]}
    receipts = {}
    for phase in ("main", "verify"):
        r = phases.get(phase)
        need(r is not None, "Prior phase absent: " + task["id"] + "/" + phase)
        result_path = Path(r["result"])
        result = read(result_path)
        exit_path = Path(r["log"] + ".exitcode")
        actual_exit = int(exit_path.read_text().strip())
        need(result.get("exit_code") == 0 and not result.get("error") and actual_exit == 0,
             "Prior actual exit not successful: " + task["id"] + "/" + phase)
        need(r.get("exit_code") == r.get("wrapper_exit_code") == 0,
             "Prior recorded exit mismatch: " + task["id"] + "/" + phase)
        need(Path(r["record"]).is_file(), "Prior process identity missing")
        receipts[phase] = {"result": str(result_path), "result_sha256": sha(result_path),
                           "wrapper_exit": str(exit_path), "wrapper_exit_sha256": sha(exit_path),
                           "process_record": r["record"], "process_record_sha256": sha(r["record"])}
    for p in expected_paths(task):
        need(p.is_file() and p.stat().st_size > 0, "Prior expected evidence absent: " + str(p))
    return receipts


def observer_wrap(task, template, plan):
    before = copy.deepcopy(task)
    base = str(plan / "tasks/source_artifacts.py")
    entry = str(plan / "evaluation_entrypoints/capture_source.py")
    for key, mode in (("command", "evaluate"), ("verify_command", "verify_eval")):
        cmd = task[key]
        i = cmd.index(base)
        need(cmd[i + 1] == mode, "Unexpected source-evaluation template")
        task[key] = cmd[:i] + [entry, "--base-entry"] + cmd[i:]
    assets = task["metadata"]["asset_sha256"]
    template_assets = template["metadata"]["asset_sha256"]
    # Only transfer the separately validated same-forward observer package pins;
    # never transfer another task's config, source report, or output references.
    prefixes = tuple(str(plan / d) + "/" for d in ("evaluation_entrypoints", "extended_evaluation"))
    for p, digest in template_assets.items():
        if p.startswith(prefixes):
            need(p not in assets or assets[p] == digest, "Conflicting observer pin: " + p)
            assets[p] = digest
    task["metadata"]["pending_observer_entry"] = entry
    need(task["requires"] == before["requires"] and task["env"] == before["env"]
         and task["gpu"] == before["gpu"] and task["expected_outputs"] == before["expected_outputs"],
         "Observer wrapper changed an original execution/data boundary")
    return task


def build(plan, destination):
    old_manifest_path = plan / "queue_proposals/odr003_calibration_20261023/proposal.json"
    status_path = plan / "status.json"
    formal_path = plan / "queue_proposals/table2_formal_20261022/proposal.json"
    pending_path = plan / "queue_proposals/pending_source_tasks.json"
    old = read(old_manifest_path)
    previous = read(status_path)
    formal = read(formal_path)
    formal_receipt_path = formal_path.with_name("verified.json")
    formal_receipt = read(formal_receipt_path)
    pending = read(pending_path)
    need(canonical(old) == EXPECTED_OLD == previous["manifest_sha256"], "Current old55 binding changed")
    need(canonical(formal) == EXPECTED_FORMAL == formal_receipt["candidate_canonical_sha256"],
         "Accepted formal candidate binding changed")
    need(formal_receipt.get("status") == "passed" and sha(formal_path) == formal_receipt["candidate_file_sha256"],
         "Formal candidate receipt/bytes do not agree")
    need(len(old["tasks"]) == len(previous["tasks"]) == 55 and len(formal["tasks"]) == 64,
         "Unexpected original/formal task counts")
    need([t["id"] for t in old["tasks"][52:]] == RETIRED, "Unexpected research branch to retire")
    need(old["tasks"][:52] == formal["tasks"][:52], "Original accepted52 definitions differ")
    need(all(t["id"] == spec["id"] for t, spec in zip(previous["tasks"], old["tasks"])),
         "Original task identities/order changed")
    need(set(previous["completed"]) == {t["id"] for t in previous["tasks"][:52]},
         "Prior accepted set is not exactly the original52")
    receipts = {t["id"]: verify_completed(t) for t in previous["tasks"][:52]}
    need(previous["tasks"][52]["status"] == "failed", "Historical 003 failure no longer matches")
    need(all(t["status"] == "queued" and not t.get("runtime") for t in previous["tasks"][53:]),
         "Retired pending branch gained execution evidence; root must reconcile it")
    retired_phase = previous["tasks"][52]["runtime"]
    need(read(retired_phase["result"]).get("exit_code") == 1
         and int(Path(retired_phase["log"] + ".exitcode").read_text().strip()) == 1,
         "Historical failed branch actual terminal evidence differs")

    manifest = copy.deepcopy(formal)
    manifest["scope"] = "original52_preserved_plus_remaining150_source_epochs_and24_domain_evaluations"
    manifest["backlog"] = [
        "Only remaining8 source configurations/150epochs and24 source evaluations are selected by this continuation.",
        "Historical research branch003 is preserved in superseded state; no replay or acceptance of it is implied.",
        "Table6, B/N scans, cases, PRISMA25 and broader final reporting are not dispatched by this queue."
    ]
    observer_template = next(t for t in formal["tasks"] if t["id"] == "eval_table2_h_synthetic")
    for t in pending["pending_task_definitions"]:
        if t["id"].startswith(("train_table3_", "eval_table3_", "train_table4_", "eval_table4_")):
            copied = copy.deepcopy(t)
            if copied["kind"] == "full_source_evaluation":
                observer_wrap(copied, observer_template, plan)
            manifest["tasks"].append(copied)
    new_tasks = manifest["tasks"][52:]
    train = [t for t in new_tasks if t["id"].startswith("train_")]
    evaluate = [t for t in new_tasks if t["id"].startswith("eval_")]
    need(len(manifest["tasks"]) == 84 and len(train) == 8 and len(evaluate) == 24,
         "Continuation task count changed")
    need([t["metadata"]["config_name"] for t in train] == CONFIGS,
         "Continuation source configuration order changed")
    need(sum(t["metadata"]["epochs"] for t in train) == 150, "Continuation epoch count is not150")
    need(manifest["tasks"][:52] == old["tasks"][:52], "Accepted definitions changed")

    # Validate the existing runner's manifest contract without importing torch.
    sys.path.insert(0, str(plan / "pipeline"))
    from runner import validate_manifest, initialize_state
    validate_manifest(manifest)
    checked_assets = {}
    for task in new_tasks:
        need(not task.get("runtime") and task.get("status", "queued") == "queued",
             "New manifest embeds execution evidence")
        for p in expected_paths(task):
            need(not p.exists(), "New expected result already exists: " + str(p))
        for p in (task["log"], task["log"] + ".exitcode", task["log"] + ".verify", task["log"] + ".verify.exitcode"):
            need(not Path(p).exists(), "New task log already exists: " + p)
        for name, digest in task["metadata"]["asset_sha256"].items():
            if name not in checked_assets:
                actual = sha(name)
                need(actual == digest, "Bound asset drift: " + name)
                checked_assets[name] = digest
            else:
                need(checked_assets[name] == digest, "Conflicting new asset bindings: " + name)
    for t in train:
        cfg = Path(t["metadata"]["source"])
        content = cfg.read_text()
        need(re.search(r"(?m)^AUTO_RESUME: false$", content) and re.search(r"(?m)^  BEGIN_EPOCH: 0$", content)
             and re.search(r"(?m)^  RESUME_FILE: ''$", content), "Training is not independently initialized")
        need(re.search(r"(?m)^  END_EPOCH: " + str(t["metadata"]["epochs"]) + r"$", content),
             "Config epoch differs from task")
        out = re.search(r"(?m)^OUTPUT_DIR: (.+)$", content).group(1)
        exp = re.search(r"(?m)^EXP_NAME: (.+)$", content).group(1)
        architecture = re.search(r"(?m)^    NAME: (.+)$", content).group(1)
        output_dir = Path(out) / architecture / exp
        need(not output_dir.exists() or not any(output_dir.iterdir()),
             "New source output is not empty: " + str(output_dir))

    projected = copy.deepcopy(previous)
    projected["tasks"] = copy.deepcopy(previous["tasks"][:52]) + [dict(copy.deepcopy(t), status="queued") for t in new_tasks]
    projected["manifest_sha256"] = canonical(manifest)
    projected["completed"] = [t["id"] for t in projected["tasks"][:52]]
    projected["active"] = None
    projected["current"] = {}
    projected["error"] = None
    projected["resource_wait"] = None
    projected["status"] = "continuation_prepared_not_deployed"
    projected["scope"] = manifest["scope"]
    projected["backlog"] = copy.deepcopy(manifest["backlog"])
    projected.pop("downstream_blocked", None)
    projected["previous_manager"] = projected.pop("manager", None)
    projected["source_continuation_revision"] = str(destination / "supersede_audit.json")
    projected.setdefault("notes", []).append(
        "User2026-11-01 explicitly resumed remaining150 source epochs. Historical003 failed/pending research branch is archived, not accepted or replayed. Original52accepted task records are reused exactly.")
    initialized = initialize_state(projected, manifest)
    need(initialized["tasks"][:52] == previous["tasks"][:52], "Runtime/history preservation failed")
    need(len(initialized["completed"]) == 52 and all(t["status"] == "queued" for t in initialized["tasks"][52:]),
         "New queued work was incorrectly preaccepted")
    need(not set(RETIRED).intersection(initialized["completed"]), "Retired branch was incorrectly accepted")

    audit = {
        "schema": "spnv2-source150-explicit-branch-supersession-v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "authorization": "Direct human request2026-11-01 to finish previously deferred150 source epochs on Jiangnan host",
        "operation": "prepare_new_source_continuation_binding_only",
        "production_mutation": False, "no_scientific_worker_started": True,
        "old_manifest": str(old_manifest_path), "old_manifest_file_sha256": sha(old_manifest_path),
        "old_manifest_canonical_sha256": canonical(old), "old_status_file_sha256": sha(status_path),
        "old_status_canonical_sha256": canonical(previous),
        "formal_candidate": str(formal_path), "formal_candidate_sha256": sha(formal_path),
        "formal_acceptance_sha256": sha(formal_receipt_path), "pending_templates_sha256": sha(pending_path),
        "retired_from_selected_source_queue": copy.deepcopy(previous["tasks"][52:]),
        "retired_branch_policy": "Preserve original55 state/manifest and all original logs/results. Failure remains failure. No task in this branch is accepted, retried or counted as complete.",
        "preserved_accepted_ids": [t["id"] for t in previous["tasks"][:52]],
        "preserved_accepted_records_canonical_sha256": canonical(previous["tasks"][:52]),
        "original_phase_exit_receipts": receipts,
        "new_tasks": [t["id"] for t in new_tasks], "formal_epochs_remaining": 150,
        "source_configurations_remaining": CONFIGS, "evaluations_remaining": 24,
        "assets_checked": checked_assets, "candidate_manifest_canonical_sha256": canonical(manifest),
        "candidate_state_canonical_sha256": canonical(projected),
        "required_root_cutover": ["Hold original pipeline.lock and prove no active scientific worker",
            "Recheck exact old status/manifest hashes then preserve old bytes before atomic state replacement",
            "Apply separately validated storage-only entry revision before launch if required",
            "Launch existing runner with only --manifest selected_candidate; no old extension/revision flags",
            "Do not reopen completed G50-P pipeline, budget, lock seal, models or results"]
    }
    destination.mkdir(parents=True, exist_ok=True)
    # Preserve original bytes, not a JSON reserialization, with exclusive writes.
    for name, src in (("old55_status.json", status_path), ("old55_manifest.json", old_manifest_path)):
        with (destination / name).open("xb") as f:
            f.write(src.read_bytes()); f.flush(); os.fsync(f.fileno())
    exclusive(destination / "manifest.json", manifest)
    exclusive(destination / "candidate_state.json", projected)
    exclusive(destination / "supersede_audit.json", audit)
    exclusive(destination / "verification.json", {
        "status": "passed_preparation_only", "models_imported": False, "model_forwards": 0,
        "checks": ["original52actual_main_verify_exit0_and_exact_records_preserved",
                   "only_original8configs150epochs24evaluations_and_original_order",
                   "new_assets_pinned_outputs_empty_and_new_work_queued_only",
                   "historical003failure_preserved_never_preaccepted"],
        "files_sha256": {name: sha(destination / name) for name in
                         ("old55_status.json", "old55_manifest.json", "manifest.json", "candidate_state.json", "supersede_audit.json")}})
    print(json.dumps({"status": "prepared_not_deployed", "tasks": 84, "reused_accepted": 52,
                      "new_trainings": 8, "new_epochs": 150, "new_evaluations": 24,
                      "destination": str(destination), "manifest_sha256": canonical(manifest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.plan.resolve(), args.out.resolve())
