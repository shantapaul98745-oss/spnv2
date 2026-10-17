"""Read real accepted fixed40 adaptation resource evidence; never perform ODR."""
import json
import math
from pathlib import Path
from bench_contract import PLAN, require, sha


def check_record(audit, task):
    require(audit["task"] == task and audit["status"] == "adaptation_passed_evaluation_pending",
            "ODR audit task/state mismatch")
    require(audit["N_seen"] == task["N"] and audit["optimizer_steps"] == task["N"] // task["B"],
            "ODR observation/update accounting mismatch")
    require(type(audit["adaptation_seconds"]) in (int, float)
            and math.isfinite(audit["adaptation_seconds"]) and audit["adaptation_seconds"] > 0,
            "Invalid actual ODR duration")
    require(type(audit["peak_allocated_bytes"]) is int and audit["peak_allocated_bytes"] > 0
            and type(audit["peak_reserved_bytes"]) is int
            and audit["peak_reserved_bytes"] >= audit["peak_allocated_bytes"], "Invalid actual ODR memory")
    require(type(audit["trainable_parameters"]) is int and audit["trainable_parameters"] > 0,
            "Invalid actual ODR parameter count")


def collect():
    root = PLAN / "odr"
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((PLAN / "status.json").read_text())
    accepted = set(state["completed"])
    records = []
    for task in [t for t in manifest["tasks"] if t["group"] == "fixed"]:
        require(task["id"] in accepted, "Fixed40 acceptance missing; do not invent ODR resources")
        proof_path = root / "reports" / (task["id"] + ".json")
        proof = json.loads(proof_path.read_text())
        path = Path(task["output"]) / "adaptation_audit.json"
        require(proof["status"] == "passed" and proof["task_id"] == task["id"]
                and proof["adaptation_audit_sha256"] == sha(path)
                and proof["implementation_manifest_sha256"] == sha(root / "implementation_manifest.json"),
                "ODR acceptance/file binding mismatch")
        audit = json.loads(path.read_text())
        check_record(audit, task)
        require(audit["source_sha256"] == proof["source_sha256"]
                and audit["implementation_manifest_sha256"] == proof["implementation_manifest_sha256"],
                "ODR source/implementation binding mismatch")
        records.append({
            "task_id": task["id"], "phi": task["phi"], "line": task["line"], "domain": task["domain"],
            "seed": task["seed"], "N": task["N"], "B": task["B"],
            "adaptation_seconds": audit["adaptation_seconds"],
            "peak_allocated_bytes": audit["peak_allocated_bytes"], "peak_reserved_bytes": audit["peak_reserved_bytes"],
            "trainable_parameters": audit["trainable_parameters"], "N_seen": audit["N_seen"],
            "optimizer_steps": audit["optimizer_steps"],
            "source_sha256": proof["source_sha256"], "audit": str(path), "audit_sha256": sha(path),
            "acceptance": str(proof_path), "acceptance_sha256": sha(proof_path),
        })
    require(len(records) == 40, "All fixed40 real resource records required")
    return {
        "scope": "Existing accepted fixed40 only; no new ODR, no aliases counted as executions",
        "timer_scope": "Actual audited_entry adaptation_seconds timer, including its audit/save overhead before domain evaluation; not pure-forward timing",
        "records": records, "count": 40, "phi0_and_phi3_GN_new_ODR_executions": 0,
        "odr_manifest_sha256": sha(root / "manifest.json"),
        "implementation_manifest_sha256": sha(root / "implementation_manifest.json"),
    }


def check_against_sources(collected, entries):
    by_name = {entry['name']: entry for entry in entries}
    for record in collected['records']:
        source = by_name['phi3_bn_ehs' if record['phi'] == 3 else 'phi6_gn_ehs']
        require(record['phi'] in (3, 6) and record['source_sha256'] == source['final_sha256'], 'ODR resource belongs to a different accepted source')
        require(record['trainable_parameters'] == source['parameter_counts']['odr_updated_backbone_norm_affine'], 'Actual ODR trainable parameter denominator differs from verified scope')
    return True
