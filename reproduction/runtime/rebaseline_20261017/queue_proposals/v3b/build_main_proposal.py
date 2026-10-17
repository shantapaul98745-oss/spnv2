"""Create an unactivated append-only main-flow v3 proposal from reviewed real APIs."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import yaml

PLAN = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
ODR = PLAN / "odr"
EXPECTED_V2 = "6562b36060f12e72dcdf9340b5262cf766cafea93c9a2fccc7ee1513c240d61b"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(name, value):
    path = DEST / name
    if path.exists():
        raise FileExistsError("Never overwrite a proposal: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def report_contracts():
    source = (ODR / "verify_result.py").read_text()
    root = ast.parse(source)
    reports = [
        node.value for node in ast.walk(root)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "report" for t in node.targets)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict"
    ]
    contracts = {}
    for node in reports:
        fields = sorted(k.arg for k in node.keywords)
        kind = "alias" if "reuse_of" in fields else "execution"
        if kind in contracts:
            raise ValueError("Ambiguous actual CPU verifier report schema")
        contracts[kind] = fields
    if set(contracts) != {"alias", "execution"} or "N_seen" not in contracts["execution"]:
        raise ValueError("ODR CPU verifier API changed; root review required")
    expressions = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "atomic":
            expressions.append(ast.get_source_segment(source, node.args[0]))
    if not expressions or any("O/'reports'/(t['id']+'.json')" not in x.replace(" ", "") for x in expressions):
        raise ValueError("Actual CPU verifier report path changed; do not guess")
    return {"fields": contracts, "actual_atomic_report_path_expressions": expressions,
            "path_template": str(ODR / "reports") + "/{task_id}.json"}


ASSET_CACHE = {}


def bind(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        key = str(path)
        if key not in ASSET_CACHE:
            ASSET_CACHE[key] = sha(path)
        result[key] = ASSET_CACHE[key]
    return result


def source_assets():
    return [path for folder in ("core", "tools") for path in sorted((PLAN / "source" / folder).rglob("*")) if path.is_file()]


def unique(items):
    return list(dict.fromkeys(items))


def make_odr_task(spec, previous, all_fixed, contracts):
    alias = bool(spec.get("reuse_of"))
    phi = spec["phi"]
    name = "phi3_bn_ehs" if phi == 3 else "phi6_gn_ehs"
    domain_tasks = ["eval_phi" + str(phi) + "_" + domain for domain in ("synthetic", "lightbox", "sunlamp")]
    requirements = [previous] + domain_tasks
    if spec["group"] != "fixed":
        requirements += all_fixed
    if alias:
        requirements.append(spec["reuse_of"])
    env = copy.deepcopy(BASE_ENV)
    env["PYTHONHASHSEED"] = str(spec["seed"])
    if alias:
        env["CUDA_VISIBLE_DEVICES"] = ""
    assets = source_assets()
    assets += [PLAN / "source_manifest.json", PLAN / "pipeline/runner.py",
               ODR / "manifest.json", ODR / "implementation_manifest.json",
               ODR / "reports/integration_cpu_acceptance.json"]
    assets += sorted(ODR.glob("*.py"))
    assets += [ODR / (line + "_sha256.json") for line in ("A", "B1", "B2")]
    assets += [path for path in sorted((ODR / ("source_" + spec["line"])).rglob("*")) if path.is_file()]
    assets += [Path(spec["config"])]
    assets += [PLAN / "configs" / (name + suffix + ".yaml")
               for suffix in ("", "_eval_synthetic", "_eval_lightbox", "_eval_sunlamp")]
    if phi == 3:
        assets.append(PLAN.parents[1] / "env/cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth")
    if alias:
        canonical_spec = ODR_BY_ID[spec["reuse_of"]]
        assets.append(Path(canonical_spec["config"]))
    report = ODR / "reports" / (spec["id"] + ".json")
    contract_name = "alias" if alias else "execution"
    return {
        "id": spec["id"],
        "kind": "odr_alias_verification" if alias else "odr_" + spec["group"],
        "command": copy.deepcopy(spec["queue_command"]), "cwd": str(ODR),
        "requires": unique(requirements),
        "session": "spnv2_odr_" + spec["id"] + "_qv3",
        "log": str(PLAN / "logs" / ("queuev3_odr_" + spec["id"] + ".log")),
        "env": env, "gpu": False if alias else copy.deepcopy(ODR_MANIFEST["gpu_gate"]),
        "verify_gpu": False,
        "verify_env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        "verify_command": copy.deepcopy(spec["verify_command"]),
        "expected_outputs": [str(report)],
        "metadata": {
            "source": spec["config"], "source_final": spec["source_final"],
            "source_report": spec["source_report"], "source_three_domain_tasks": domain_tasks,
            "requires_source_three_domain_acceptance": True, "odr_group": spec["group"],
            "phi": phi, "line": spec["line"], "domain": spec["domain"],
            "observation_seed": spec["seed"], "N": spec["N"], "B": spec["B"],
            "physical_loader_batch": 1, "reuse_of": spec.get("reuse_of"),
            "actual_report": str(report), "actual_cpu_report_contract": contract_name,
            "actual_cpu_report_required_fields": contracts["fields"][contract_name],
            "odr_manifest_task_sha256": canonical(spec),
            "asset_sha256": bind(assets),
            "interpretation": "ODR observation seed; one independent formal source per architecture",
        },
    }


def main():
    global BASE_ENV, ODR_MANIFEST, ODR_BY_ID
    if sha(PLAN / "pipeline/runner.py") != "72385d6ca9f9fc4a2e7005a57761e9393a93922921226bbb254402639b9a7c47":
        raise ValueError("Runner differs from accepted commit aa8e26fd; refresh proposal review")
    old = json.loads((PLAN / "queue_v2.json").read_text())
    if canonical(old) != EXPECTED_V2 or len(old["tasks"]) != 9:
        raise ValueError("Live v2 differs from reviewed exact prefix")
    ODR_MANIFEST = json.loads((ODR / "manifest.json").read_text())
    ODR_BY_ID = {task["id"]: task for task in ODR_MANIFEST["tasks"]}
    if len(ODR_BY_ID) != 400 or ODR_MANIFEST["counts"] != {"fixed": 40, "Bscan": 120, "Nscan": 240}:
        raise ValueError("Unexpected actual ODR request set")
    BASE_ENV = copy.deepcopy(old["tasks"][4]["env"])
    contracts = report_contracts()
    fixed = [task for task in ODR_MANIFEST["tasks"] if task["group"] == "fixed"]
    scans = [task for group in ("Bscan", "Nscan") for task in ODR_MANIFEST["tasks"] if task["group"] == group]
    all_fixed = [task["id"] for task in fixed]
    proposed = copy.deepcopy(old)
    previous = proposed["tasks"][-1]["id"]
    for spec in fixed:
        item = make_odr_task(spec, previous, all_fixed, contracts)
        proposed["tasks"].append(item)
        previous = item["id"]
    pending = json.loads((PLAN / "queue_proposals/pending_source_tasks.json").read_text())
    source_tasks = copy.deepcopy(pending["pending_task_definitions"])
    if len(source_tasks) != 35 or source_tasks[0]["id"] != "pilot_table2_h":
        raise ValueError("Unexpected pending source definitions")
    changed_pending = []
    for index, task in enumerate(source_tasks):
        if index == 0:
            old_requires = task["requires"]
            task["requires"] = [previous]
            changed_pending.append({"task": task["id"], "field": "requires",
                                    "old": old_requires, "new": task["requires"]})
        # These definitions were never bound live; pin current reviewed assets again.
        previous_hashes = task["metadata"]["asset_sha256"]
        current_hashes = bind(previous_hashes)
        for path, value in current_hashes.items():
            if previous_hashes[path] != value:
                changed_pending.append({"task": task["id"], "field": "metadata.asset_sha256",
                                        "path": path, "old": previous_hashes[path], "new": value})
        task["metadata"]["asset_sha256"] = current_hashes
        proposed["tasks"].append(task)
        previous = task["id"]
    for spec in scans:
        item = make_odr_task(spec, previous, all_fixed, contracts)
        proposed["tasks"].append(item)
        previous = item["id"]
    proposed["scope"] = "main_sources_fixed40_and_BN_scans_pending_analysis"
    proposed["backlog"] = [
        item for item in old["backlog"]
        if item.startswith("Table6") or item.startswith("改善/退化") or item.startswith("PRISMA25")
    ]
    proposed["backlog"].append("GPU资源/吞吐测量、最终分析和完整证据矩阵仍待独立验收；本队列结束不代表整体目标完成")
    path = save("full_main_v3_proposal.json", proposed)
    save("report_contracts.json", contracts)
    save("asset_catalog.json", ASSET_CACHE)
    source_index = json.loads((PLAN / "queue_proposals/proposal_index.json").read_text())["source_configurations"]
    save("proposal_index.json", {
        "proposal_only": True, "activated": False,
        "root_review_required": ["ODR real CUDA execution implementation", "Current asset hashes and independent CPU evidence",
                                 "Exact worker reconciliation and explicit append-only manager restart"],
        "parent_manifest": str(PLAN / "queue_v2.json"),
        "parent_file_sha256": sha(PLAN / "queue_v2.json"), "parent_canonical_sha256": canonical(old),
        "proposal": str(path), "proposal_file_sha256": sha(path), "proposal_canonical_sha256": canonical(proposed),
        "proposal_task_count": len(proposed["tasks"]), "preserved_prefix_count": 9,
        "appended_task_count": len(proposed["tasks"]) - 9,
        "fixed_task_ids": all_fixed, "last_fixed_task_id": fixed[-1]["id"],
        "first_source_after_fixed": source_tasks[0]["id"], "last_source_before_scans": source_tasks[-1]["id"],
        "first_scan_task_id": scans[0]["id"], "last_scan_task_id": scans[-1]["id"],
        "counts": {"formal_source_configurations": 10, "formal_source_epochs": 190,
                   "source_domain_evaluations": 30, "engineering_pilots": 4,
                   "odr_requests": 400, "odr_fixed": 40, "odr_Bscan": 120, "odr_Nscan": 240,
                   "odr_unique_executions": 370, "odr_alias_requests": 30},
        "pending_definition_changes": changed_pending, "source_configurations": source_index,
        "odr_manifest_sha256": sha(ODR / "manifest.json"),
        "odr_implementation_manifest_sha256": sha(ODR / "implementation_manifest.json"),
        "odr_integration_cpu_report_sha256": sha(ODR / "reports/integration_cpu_acceptance.json"),
        "odr_gpu_execution_evidence": False,
        "backlog_preserved": proposed["backlog"],
        "artifact_hashes": {"report_contracts.json": sha(DEST / "report_contracts.json"),
                            "asset_catalog.json": sha(DEST / "asset_catalog.json")},
    })
    print(json.dumps({"proposal_tasks": len(proposed["tasks"]), "preserved_prefix": 9,
                      "fixed": len(fixed), "pending_sources": len(source_tasks), "scans": len(scans),
                      "aliases": sum(bool(t.get("reuse_of")) for t in ODR_MANIFEST["tasks"])}))


if __name__ == "__main__":
    main()
