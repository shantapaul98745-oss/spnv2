"""CPU-only static graph/asset audit. Never execute a proposal command or publish state."""
import argparse
import ast
import copy
from collections import Counter
import hashlib
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import time

import yaml

PLAN = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
ODR = PLAN / "odr"
sys.path.insert(0, str(PLAN / "pipeline"))
import runner
from revision import extend_state

EXPECTED_V2 = "6562b36060f12e72dcdf9340b5262cf766cafea93c9a2fccc7ee1513c240d61b"


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arg(command, key):
    require(key in command and command.index(key) + 1 < len(command), "Missing CLI argument " + key)
    return command[command.index(key) + 1]


def layout(proposal, old, manifest):
    require(len(proposal["tasks"]) == 444, "Expected444 total parent tasks")
    require(json.dumps(proposal["tasks"][:9], ensure_ascii=False) == json.dumps(old["tasks"], ensure_ascii=False),
            "Exact nine-task bound prefix changed")
    fixed = [x for x in manifest["tasks"] if x["group"] == "fixed"]
    scans = [x for group in ("Bscan", "Nscan") for x in manifest["tasks"] if x["group"] == group]
    tasks = proposal["tasks"]
    require([t["id"] for t in tasks[9:49]] == [t["id"] for t in fixed], "Fixed40 block missing/reordered")
    require(tasks[49]["id"] == "pilot_table2_h" and tasks[83]["id"] == "eval_table4_phi3_gn_sunlamp",
            "All remaining sources must lie between fixed40 and scans")
    require([t["id"] for t in tasks[84:]] == [t["id"] for t in scans], "Scan360 block missing/reordered")
    require(tasks[49]["requires"] == [fixed[-1]["id"]], "Table2 is not gated by true final fixed task")
    by_id = {task["id"]: task for task in tasks}
    require(len(by_id) == 444, "Duplicate task identity")
    fixed_ids = {x["id"] for x in fixed}
    for offset, spec in enumerate(fixed + scans):
        task = by_id[spec["id"]]
        position = 9 + offset if offset < 40 else 84 + offset - 40
        require(tasks[position - 1]["id"] in task["requires"], "ODR serial predecessor missing")
        source_evals = {"eval_phi" + str(spec["phi"]) + "_" + d for d in ("synthetic", "lightbox", "sunlamp")}
        require(source_evals <= set(task["requires"]), "ODR lacks all three accepted source domains")
        if spec["group"] != "fixed":
            require(fixed_ids <= set(task["requires"]), "Scan lacks explicit all-fixed barrier")
        require(task["command"] == spec["queue_command"] and task["verify_command"] == spec["verify_command"],
                "ODR actual CLI changed/guessed")
        require(task["metadata"]["odr_manifest_task_sha256"] == runner.manifest_hash(spec),
                "ODR task definition binding mismatch")
        if spec.get("reuse_of"):
            require(spec["reuse_of"] in task["requires"], "Alias canonical acceptance missing")
            require(task["gpu"] is False and task["env"]["CUDA_VISIBLE_DEVICES"] == "",
                    "Alias should not start another GPU adaptation")
        else:
            require(task["gpu"] == manifest["gpu_gate"], "ODR GPU gate differs from reviewed parent")
    for pos in range(50, 84):
        require(tasks[pos]["requires"] == [tasks[pos - 1]["id"]], "Source serial dependency broken")
    require(proposal["backlog"] and any("GPU" in x for x in proposal["backlog"])
            and any("改善/退化" in x for x in proposal["backlog"])
            and any("PRISMA25" in x for x in proposal["backlog"]),
            "Resource/final analysis/material limits must remain backlog")
    runner.validate_manifest(proposal)
    return by_id


def audit():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTHONNOUSERSITE") == "1"
            and os.environ.get("STY"), "Independent guarded CPU environment required")
    if sha(PLAN / "pipeline/runner.py") != "72385d6ca9f9fc4a2e7005a57761e9393a93922921226bbb254402639b9a7c47":
        raise ValueError("Runner differs from accepted commit aa8e26fd; refresh proposal review")
    old_path = PLAN / "queue_v2.json"
    old = json.loads(old_path.read_text())
    manifest = json.loads((ODR / "manifest.json").read_text())
    proposal_path = DEST / "full_main_v3_proposal.json"
    proposal = json.loads(proposal_path.read_text())
    index = json.loads((DEST / "proposal_index.json").read_text())
    contracts = json.loads((DEST / "report_contracts.json").read_text())
    require(runner.manifest_hash(old) == EXPECTED_V2 == index["parent_canonical_sha256"], "Live v2 prefix drift")
    require(index["proposal_only"] is True and index["activated"] is False
            and index["odr_gpu_execution_evidence"] is False, "Proposal must remain unactivated/unexecuted")
    require(index["proposal_file_sha256"] == sha(proposal_path)
            and index["proposal_canonical_sha256"] == runner.manifest_hash(proposal), "Proposal digest mismatch")
    require(index["parent_file_sha256"] == sha(old_path), "Bound parent file changed")
    for name, expected in index["artifact_hashes"].items():
        require(sha(DEST / name) == expected, "Proposal auxiliary artifact drift")
    archive = index['archive']
    compressed_bytes = (DEST / 'proposal.json.gz').read_bytes()
    decompressed = gzip.decompress(compressed_bytes)
    require(decompressed == proposal_path.read_bytes(), 'Gzip decompression changes raw proposal')
    require(hashlib.sha256(decompressed).hexdigest() == archive['raw_sha256'] == archive['decompressed_raw_sha256'] == index['proposal_file_sha256'], 'Raw/gzip digest mismatch')
    require(sha(DEST / 'proposal.json.gz') == archive['gzip_sha256'] and compressed_bytes[4:8] == bytes(4) and not (compressed_bytes[3] & 8), 'Non-deterministic gzip header/digest')
    overview = (DEST / 'task_overview.md').read_text()
    rows = [line for line in overview.splitlines() if line.startswith('| ') and line.split('|', 2)[1].strip().isdigit()]
    require(len(rows) == 444 and sum('| 是 → ' in line for line in rows) == 30, 'Readable overview task/alias count mismatch')
    by_id = layout(proposal, old, manifest)
    checks = {"exact_bound_prefix_and_ordinary_manifest_validation": True,
              "fixed40_then_sources35_then_scan360_serial_graph": True,
              "every_odr_requires_source_three_domains_and_full_fixed_scan_barrier": True,
              "resource_analysis_and_material_backlog_preserved": True}

    # Validate the contract directly against the actual CPU verifier AST.
    verifier_source = (ODR / "verify_result.py").read_text()
    actual_contracts = {}
    for node in ast.walk(ast.parse(verifier_source)):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "report" for t in node.targets)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dict"):
            fields = sorted(x.arg for x in node.value.keywords)
            actual_contracts["alias" if "reuse_of" in fields else "execution"] = fields
    require(actual_contracts == contracts["fields"], "Actual CPU report field contract drift")

    sys.path.insert(0, str(ODR))
    import common as odr_common
    implementation = odr_common.verify_implementation()
    for line in ("A", "B1", "B2"):
        odr_common.verify_code(line)
    acceptance = json.loads((ODR / "reports/integration_cpu_acceptance.json").read_text())
    require(acceptance["status"] == "passed"
            and acceptance["implementation_manifest_sha256"] == sha(ODR / "implementation_manifest.json"),
            "ODR CPU engineering evidence does not bind current implementation")
    require(acceptance["real_GPU_queries"] is False and acceptance["real_ODR_executed"] is False,
            "Do not label CPU engineering evidence as real GPU acceptance")
    checks["actual_odr_schema_helpers_configs_and_engineering_acceptance_verified"] = True
    odr_tasks = [t for t in proposal["tasks"] if t["kind"].startswith("odr_")]
    require(len(odr_tasks) == 400 and Counter(t["metadata"]["odr_group"] for t in odr_tasks)
            == {"fixed": 40, "Bscan": 120, "Nscan": 240}, "ODR400 group counts incorrect")
    requested = manifest["tasks"]
    fixed = [t for t in requested if t["group"] == "fixed"]
    bscan = [t for t in requested if t["group"] == "Bscan"]
    nscan = [t for t in requested if t["group"] == "Nscan"]
    require(Counter((t["phi"], t["line"]) for t in fixed)
            == {(3, "A"): 10, (6, "A"): 10, (3, "B1"): 10, (3, "B2"): 10},
            "Main20 or repaired B1/B2 fixed lines omitted")
    seeds, domains = range(2021, 2026), ("lightbox", "sunlamp")
    expected_fixed = {(phi, line, d, s, 4096 if phi == 6 else 1024, 1 if phi == 6 else 4)
                      for phi, line in ((3, "A"), (6, "A"), (3, "B1"), (3, "B2"))
                      for d, s in itertools.product(domains, seeds)}
    key = lambda t: (t["phi"], t["line"], t["domain"], t["seed"], t["N"], t["B"])
    require({key(t) for t in fixed} == expected_fixed, "Fixed protocol Cartesian product mismatch")
    require({key(t) for t in bscan}
            == {(3, line, d, s, 1024, b) for line, d, s, b in
                itertools.product(("A", "B2"), domains, seeds, (1, 2, 4, 8, 16, 32))},
            "Bscan120 Cartesian product mismatch")
    require({key(t) for t in nscan}
            == {(phi, "A", d, s, n, 4) for phi, d, s, n in
                itertools.product((3, 6), domains, seeds, (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))},
            "Nscan240 Cartesian product mismatch")
    raw_by_id = {task["id"]: task for task in requested}
    aliases = [t for t in requested if t.get("reuse_of")]
    require(len(aliases) == 30 and len(requested) - len(aliases) == 370, "Unique370/alias30 incorrect")
    require(Counter(t["group"] for t in aliases) == {"Bscan": 20, "Nscan": 10}, "Alias group counts incorrect")
    for task in aliases:
        target = raw_by_id[task["reuse_of"]]
        require(not target.get("reuse_of") and key(task) == key(target), "Alias changes protocol or chains aliases")
        for field in ("source_final", "source_report", "code_sha256_manifest"):
            require(task[field] == target[field], "Alias does not share exact new source/code")
    checks["fixed40_Bscan120_Nscan240_and370_unique30_alias"] = True

    source_train = [t for t in proposal["tasks"] if t["kind"].startswith("source_training_")]
    source_eval = [t for t in proposal["tasks"] if t["kind"] == "full_source_evaluation"]
    pilots = [t for t in proposal["tasks"] if t["kind"] == "engineering_gpu_resume_pilot"]
    require(len(source_train) == 10 and len(source_eval) == 30 and len(pilots) == 4,
            "Expected10 formal sources/30 domain evaluations/4 pilots")
    epochs = []
    formal_finals = []
    for task in source_train:
        cfg_path = Path(arg(task["command"], "--cfg"))
        cfg = yaml.safe_load(cfg_path.read_text())
        epochs.append(cfg["TRAIN"]["END_EPOCH"])
        final = str(Path(cfg["OUTPUT_DIR"]) / cfg["MODEL"]["BACKBONE"]["NAME"] / cfg["EXP_NAME"] / "model_final.pth.tar")
        formal_finals.append(final)
        require(cfg["TRAIN"]["RESUME_FILE"] == "" and cfg["AUTO_RESUME"] is False
                and not cfg["MODEL"].get("PRETRAIN_FILE", "")
                and not cfg["MODEL"]["BACKBONE"].get("PRETRAINED", ""),
                "Formal source inherits prior/pilot weights")
        require(cfg["ROOT"] == str(PLAN / "source") and cfg["SEED"] == 2021
                and cfg["TRAIN"]["ACCUM_STEPS"] == 4, "Source protocol drift")
    require(sum(epochs) == 190 and Counter(epochs) == {20: 9, 10: 1}
            and len(set(formal_finals)) == 10, "Formal190 epochs or independent finals incorrect")
    require(sum(t["id"] == "train_phi3_EHS" for t in proposal["tasks"]) == 1, "Live phi3 duplicated")
    for pilot in pilots:
        require(arg(pilot["command"], "--attempts") == "32"
                and arg(pilot["command"], "--checkpoint") == "16", "Pilot32/16 protocol changed")
    for t in source_eval:
        require(arg(t["command"], "--expected-samples") in ("11994", "6740", "2791"), "Source domain denominator changed")
    require(Counter(arg(t["command"], "--expected-samples") for t in source_eval)
            == {"11994": 10, "6740": 10, "2791": 10}, "Each source needs all three domain evaluations")
    checks["formal10_configs190_epochs30_domains4_pilots_no_live_phi3_duplicate"] = True

    checked = {}
    source_assets = {str(p.resolve()) for folder in ("core", "tools")
                     for p in (PLAN / "source" / folder).rglob("*") if p.is_file()}
    for task in proposal["tasks"]:
        assets = task.get("metadata", {}).get("asset_sha256", {})
        if task in proposal["tasks"][:4]:
            continue  # Immutable original prefix predates asset metadata.
        require(source_assets <= set(assets), "New task lacks complete frozen parent source")
        for path, expected in assets.items():
            runner.inside(path)
            require(Path(path).is_absolute(), "Asset path not absolute")
            if path not in checked:
                checked[path] = sha(path)
            require(checked[path] == expected, "Pinned asset drift: " + path)
        require(task["verify_gpu"] is False and task["verify_env"]["CUDA_VISIBLE_DEVICES"] == "",
                "Independent verification must run on CPU")
        require(task["env"]["PYTHONNOUSERSITE"] == "1"
                and task["env"]["OMP_NUM_THREADS"] == task["env"]["OPENBLAS_NUM_THREADS"] == "2"
                and task["env"]["TORCH_HOME"] == str(PLAN.parents[1] / "env/cache/torch")
                and task["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "Execution environment changed")
    for spec in requested:
        task = by_id[spec["id"]]
        assets = task["metadata"]["asset_sha256"]
        report_path = str(ODR / "reports" / (spec["id"] + ".json"))
        kind = "alias" if spec.get("reuse_of") else "execution"
        require(task["expected_outputs"] == [report_path]
                and task["metadata"]["actual_report"] == report_path, "Actual CPU acceptance report path mismatch")
        require(task["metadata"]["actual_cpu_report_required_fields"] == contracts["fields"][kind],
                "Actual CPU acceptance schema mismatch")
        require(task["command"][4] == str(ODR / "run_task.py") and task["command"][-1] == "--execute"
                and arg(task["command"], "--task") == spec["id"], "Wrong actual queue command")
        require(task["verify_command"][4] == str(ODR / "verify_result.py")
                and arg(task["verify_command"], "--task") == spec["id"], "Wrong actual CPU verifier command")
        require(assets[spec["config"]] == spec["config_sha256"], "Actual selected ODR config hash mismatch")
        line_assets = {str(p.resolve()) for p in (ODR / ("source_" + spec["line"])).rglob("*") if p.is_file()}
        require(line_assets <= set(assets), "Selected A/B1/B2 actual code not fully pinned")
        for path in (ODR / "manifest.json", ODR / "implementation_manifest.json",
                     ODR / "reports/integration_cpu_acceptance.json", PLAN / "pipeline/runner.py"):
            require(str(path) in assets, "ODR reviewed manifest/helper/gate evidence not bound")
        cfg = yaml.safe_load(Path(spec["config"]).read_text())
        require(cfg["ROOT"] == str(ODR / ("source_" + spec["line"]))
                and cfg["ODR"]["NUM_TRAIN_SAMPLES"] == spec["N"]
                and cfg["ODR"]["IMAGES_PER_BATCH"] == spec["B"]
                and cfg["TRAIN"]["IMAGES_PER_GPU"] == 1 and cfg["TRAIN"]["ACCUM_STEPS"] == 1
                and cfg["FP16"] is False and cfg["TRAIN"]["REPRODUCIBLE"] is False
                and cfg["MODEL"]["PRETRAIN_FILE"] == spec["source_final"], "ODR actual config protocol mismatch")
        for path in (task["log"], task["log"] + ".exitcode", report_path,
                     str(PLAN / "events" / (task["id"] + ".main.config.json"))):
            require(not Path(path).exists(), "ODR proposal would overwrite execution evidence: " + path)
    checks["current_absolute_assets_actual_configs_code_and_report_contracts_bound"] = True
    current = json.loads((PLAN / "status.json").read_text())
    require(current["manifest_sha256"] == EXPECTED_V2, "Live binding changed during static audit")
    amended, revision = extend_state(current, old, proposal, caller_context={"static_v3b_proposal_audit": True})
    state = runner.initialize_state(amended, proposal)
    require(state["tasks"][0]["runtime"] == current["tasks"][0]["runtime"], "Live runtime evidence lost")
    for key, value in current["current"].items():
        if key != "process_state":
            require(state["current"][key] == value, "Current progress lost")
    require(state["current"]["pid"] == 3111276 and str(state["current"]["starttime"]) == "517862167",
            "Worker changed; refresh review context")
    checks["pure_append_binding_preserves_current_worker_and_progress"] = True

    negatives = []
    def rejects(name, mutate):
        changed = copy.deepcopy(proposal)
        mutate(changed)
        try:
            layout(changed, old, manifest)
        except (AssertionError, ValueError):
            negatives.append(name)
        else:
            raise AssertionError("Negative graph mutation accepted: " + name)
    rejects("immutable_prefix_edit", lambda p: p["tasks"][0].update(command=["false"]))
    rejects("missing_B1_fixed_task", lambda p: p["tasks"].pop(next(i for i, t in enumerate(p["tasks"]) if t["id"].startswith("fixed_phi3_B1"))))
    rejects("Table2_before_fixed_acceptance", lambda p: p["tasks"][49].update(requires=["eval_phi6_sunlamp"]))
    first_scan = proposal["tasks"][84]
    rejects("scan_missing_all_fixed_barrier", lambda p: p["tasks"][84].update(requires=[proposal["tasks"][83]["id"]] + first_scan["metadata"]["source_three_domain_tasks"]))
    first_fixed = proposal["tasks"][9]
    rejects("ODR_missing_source_three_domains", lambda p: p["tasks"][9].update(requires=[proposal["tasks"][8]["id"]]))
    alias_position = next(i for i, t in enumerate(proposal["tasks"]) if t["kind"] == "odr_alias_verification")
    rejects("alias_accidentally_GPU_executed", lambda p: p["tasks"][alias_position].update(gpu=manifest["gpu_gate"]))
    checks["negative_graph_mutations_rejected"] = True
    return {
        "status": "passed", "scope": "static proposal and graph integrity only; no real ODR/GPU execution",
        "proposal_only": True, "activated": False, "manager_changed": False, "checks": checks,
        "negative_graph_tests": negatives, "counts": index["counts"],
        "parent_tasks": 9, "proposal_tasks": 444, "appended_tasks": 435,
        "actual_cpu_report_contracts": contracts, "unique_assets_verified": len(checked),
        "archive": archive, "runner_code_git_commit": index["runner_code_git_commit"],
        "proposal_file_sha256": sha(proposal_path), "proposal_canonical_sha256": runner.manifest_hash(proposal),
        "parent_canonical_sha256": EXPECTED_V2,
        "odr_manifest_sha256": sha(ODR / "manifest.json"),
        "odr_implementation_manifest_sha256": sha(ODR / "implementation_manifest.json"),
        "backlog": proposal["backlog"], "current_worker": {"pid": state["current"]["pid"], "starttime": state["current"]["starttime"]},
        "code_sha256": {str(path): sha(path) for path in
                        (Path(__file__), DEST / "build_main_proposal.py", DEST / "package_proposal.py", PLAN / "pipeline/runner.py", PLAN / "pipeline/revision.py")},
        "screen": os.environ["STY"], "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "command": sys.argv, "finished": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    path = Path(args.report).resolve()
    require(DEST in path.parents and not path.exists(), "Use a new v3 static report path")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit()
    except Exception as error:
        result = {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                  "screen": os.environ.get("STY"), "command": sys.argv}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
