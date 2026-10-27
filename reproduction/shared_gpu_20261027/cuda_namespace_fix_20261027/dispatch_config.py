"""Compose immutable single-task migration configs from bound deployment data.

No task/index/controller mutation, state deserialization, model import, dispatch,
or acceptance is performed. A caller may publish the returned config as a new
content-addressed file under the explicitly writable config_versions directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid

try:
    from . import namespace_runner as ns
except ImportError:
    import namespace_runner as ns

DEPLOYMENT_SCHEMA = "spnv2-sysu-deployment-v1"


def resolve_path(path, resolve=None):
    return Path(path) if resolve is None else Path(resolve(path))


def read_bound(item, resolve=None):
    ns.require(isinstance(item, dict) and set(item) == {"path", "sha256"}
               and ns.HEX.fullmatch(item["sha256"] or ""), "Invalid composer binding")
    ns.absolute(item["path"])
    path = resolve_path(item["path"], resolve)
    ns.require(ns.sha(path) == item["sha256"], "Composer bound metadata changed: " + item["path"])
    return ns.metadata(path)


def exact_run(index, item):
    rows = [row for row in index["tasks"] if row["task_path"] == item["path"]
            and row["task_sha256"] == item["sha256"]]
    ns.require(len(rows) == 1, "Task is not one exact frozen index row")
    return rows[0]


def same_scientific_code(actual, original, *, index=None, resolve=None):
    """Historical manager metadata is informational; all scientific bytes bind."""
    if index is not None and index.get('resource_revision') is not None:
        rev=read_bound(index['resource_revision'],resolve);code=rev['validator']
        ns.require(code['path']==ns.LOGICAL_ROOT+'/spnv2/reproduction/shared_gpu_20261027/resource_amendment.py','Unexpected resource bridge path')
        import importlib.util
        path=resolve_path(code['path'],resolve);ns.require(ns.sha(path)==code['sha256'],'Resource bridge code changed')
        spec=importlib.util.spec_from_file_location('spnv2_dc_resource_bridge',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.code_compatible(index,actual,original,resolve)
    return (isinstance(actual, dict) and set(actual) == set(original)
            and all(actual[name] == original[name] for name in original if name != "manager.py")
            and all(ns.HEX.fullmatch(digest or "") for digest in actual.values()))


def parent_identity(run_task, resolve=None):
    """Read only bound JSON metadata, never state.bin or a pickle."""
    spec = run_task["selection"]["parent_checkpoint"]
    directory = PurePosixPath(str(ns.absolute(spec["directory"])))
    producer = PurePosixPath(run_task["producer"]["implementation_dir"]).parent
    ns.require(directory.is_relative_to(producer / "runs")
               and directory.name == "generation-" + str(spec["generation"]).zfill(20),
               "Parent metadata escapes the unchanged producer checkpoint namespace")
    value = read_bound(dict(path=str(directory / "metadata.json"), sha256=spec["metadata_sha256"]), resolve)
    ns.require(value.get("format_version") == 1 and value.get("generation") == spec["generation"]
               and value.get("state_sha256") == spec["state_sha256"]
               and value.get("scientific_cursors") == spec["cursors"], "Parent generation metadata differs")
    identity = value["identity"]
    original_binding = run_task["selection"]["original_task"]
    original = read_bound(original_binding, resolve)
    expected = dict(run_id=original["run_id"], source_sha256=original["protocol"]["source_sha256"],
                    config_sha256=original_binding["sha256"],
                    manifest_sha256=original["baseline_task"]["bindings"]["stream_manifest"]["sha256"])
    ns.require(isinstance(identity, dict) and all(identity.get(key) == value for key, value in expected.items())
               and all(isinstance(key, str) and key and isinstance(value, str) and value
                       for key, value in identity.items()), "Parent metadata original identity differs")
    return copy.deepcopy(identity)


def check_saved_main(task, run_binding, run_task, resolve=None, expected_verify_run_id=None):
    """Bind a verifier to one saved complete main; this never accepts the pair."""
    run_id, output = run_task["run_id"], run_task["output_dir"]
    expected_verify_run_id = expected_verify_run_id or run_id + "_verify"
    ns.require(task.get("run_id") == expected_verify_run_id and task.get("run_task") == run_binding
               and task.get("proof_path") == output + "/proof.json", "Verifier is not for this exact indexed run")
    receipt = task["execution_receipt"]
    report_binding = receipt["report"]
    ns.require(receipt.get("run_id") == run_id and receipt.get("task") == run_binding
               and report_binding["path"] == output + "/report.json", "Verifier main receipt was redirected")
    report = read_bound(report_binding, resolve)
    ns.require(report.get("schema") == "g50p-risk-diagnostic-pair-report-v1"
               and report.get("status") == "pair_main_complete_pending_independent_verification"
               and report.get("run_id") == run_id and report.get("task_sha256") == run_binding["sha256"]
               and report.get("selection") == run_task["selection"] and report.get("producer") == run_task["producer"]
               and report.get("panel") == run_task["panel"], "Saved main report is not the exact complete indexed main")
    plan = ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC
    launch_path = PurePosixPath(str(ns.absolute(receipt["launch_record_path"])))
    ns.require(launch_path.parent.parent == PurePosixPath(plan + "/launches") and launch_path.name == "launch.json"
               and receipt["outcome_path"] == str(launch_path.with_name("outcome.json")),
               "Saved main launch/outcome escapes the owned launch namespace")
    launch = ns.metadata(resolve_path(str(launch_path), resolve))
    outcome = ns.metadata(resolve_path(receipt["outcome_path"], resolve))
    log = PurePosixPath(str(ns.absolute(receipt["log_path"])))
    ns.require(log.parent == PurePosixPath(plan + "/logs") and receipt["exit_code_path"] == str(log) + ".exitcode",
               "Saved main actual exit record escapes owned logs")
    exit_path = resolve_path(receipt["exit_code_path"], resolve)
    ns.require(exit_path.is_file() and not exit_path.is_symlink() and exit_path.stat().st_size <= 32
               and exit_path.read_text().strip() == "0", "Saved main has no actual successful wrapper exit")
    ns.require(launch.get("run_id") == run_id and launch.get("task_sha256") == run_binding["sha256"]
               and outcome.get("run_id") == run_id and outcome.get("status") == "pair_main_complete"
               and outcome.get("exit_code") == 0 and outcome.get("report") == report_binding["path"]
               and outcome.get("report_sha256") == report_binding["sha256"],
               "Saved main launch/outcome/report bindings disagree")


def check_audit_task(index, task, task_binding, status, *, resolve=None):
    """Admit only the existing exact audit template and fixed receipt cohort."""
    pilot = task["action"] == "verify_pilots"
    kind = "pilot_admission" if pilot else "summary"
    template = index[kind + "_template"]
    additions = {"execution_receipts", "index_path"}
    ns.require(additions <= set(task) and not (additions & set(template))
               and {key: value for key, value in task.items() if key not in additions} == template
               and task_binding["path"] == index["audit_task_paths"][kind]
               and task.get("frozen_inputs") == index["frozen_inputs"],
               "Audit task differs from the exact bound pilot/summary template")
    plan = ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC
    ns.require(task["index_path"] == plan + "/index.json", "Audit index path differs")
    receipt_path = PurePosixPath(str(ns.absolute(task["execution_receipts"]["path"])))
    prefix = "pilot_receipts" if pilot else "final_receipts"
    ns.require(receipt_path.parent == PurePosixPath(plan) and receipt_path.name.startswith(prefix)
               and receipt_path.name.endswith(".json") and ns.NAME.fullmatch(receipt_path.name),
               "Audit execution receipt escapes the fixed diagnostic receipt namespace")
    receipts = read_bound(task["execution_receipts"], resolve)
    selected = index["tasks"][:2] if pilot else index["tasks"]
    planned = [row["run_id"] for row in selected]
    ns.require(receipts.get("schema") == "g50p-diagnostic-execution-receipts-v1"
               and receipts.get("planned_run_ids") == planned,
               "Audit execution receipts changed the fixed planned IDs or scope")
    runs = receipts.get("runs")
    ns.require(isinstance(runs, list) and all(isinstance(run, dict) and run.get("run_id") in planned for run in runs)
               and len({run["run_id"] for run in runs}) == len(runs), "Audit includes duplicate or unplanned mains")
    main, accepted = status.get("main", {}), status.get("accepted", {})
    expected_main = {rid: main[rid] for rid in planned if rid in main}
    expected_accepted = {rid: accepted[rid] for rid in planned if rid in accepted}
    ns.require({run["run_id"]: run for run in runs} == expected_main
               and receipts.get("accepted") == expected_accepted
               and set(expected_accepted) <= set(expected_main)
               and receipts.get("missing_main_run_ids") == [rid for rid in planned if rid not in main]
               and receipts.get("missing_independent_acceptance_run_ids") == [rid for rid in planned if rid not in accepted],
               "Audit receipts omit/change actual main/accepted evidence or its missing denominators")
    for row in selected:
        if row["run_id"] in expected_main:
            ns.require(expected_main[row["run_id"]].get("task") == dict(path=row["task_path"], sha256=row["task_sha256"]),
                       "Audit main receipt is not the exact indexed window task")
    if pilot:
        ns.require(set(expected_main) == set(planned) and set(expected_accepted) == set(planned),
                   "Both fixed pilots need actual independent acceptance before metadata audit")
    return planned


def compose_single_task(deployment_binding, task_binding, *, resolve=None):
    """Return a fully validated config approving exactly the supplied task.

Bindings contain logical absolute paths. Inside the namespace, omit resolve.
Native metadata preparation may pass ``lambda path: ns.physical(base, path)``;
every translated file is still checked against its immutable SHA binding.
"""
    deployment = read_bound(deployment_binding, resolve)
    ns.require(deployment.get("schema") == DEPLOYMENT_SCHEMA
               and deployment.get("migration_id") == ns.MIGRATION, "Wrong immutable deployment manifest")
    template = read_bound(deployment["template"], resolve)
    index = read_bound(deployment["index"], resolve)
    ns.require(template.get("schema") == ns.SCHEMA and template.get("migration_id") == ns.MIGRATION
               and template.get("approved_tasks") == [] and template.get("approved_stores") == []
               and template.get("diagnostic_index") == deployment["index"], "Template is not an empty bound deployment template")
    ns.require(index.get("schema") == "g50p-risk-diagnostic-index-v1"
               and index.get("migration_extensions") == template["code_bindings"]
               and index.get("protocol") == template["effective_protocol"]
               and index.get("budget_policy") == template["budget_policy"]
               and index.get("frozen_inputs") == template["frozen_inputs"]
               and index.get("resource_revision") == template.get("resource_revision")
               and index.get("resource_policy") == template.get("resource_policy"),
               "Deployment changed original implementation, policy, protocol, inputs, or static extensions")
    current_code, original_code = index["implementation_sha256"], template["original_implementation_sha256"]
    ns.require(same_scientific_code(current_code, original_code,index=index,resolve=resolve),
               "Only the explicitly index-bound metadata manager may differ from the original implementation")
    ns.require(index.get("sysu_task_revision") == template["new_task_revision"], "Manager task revision differs from the template")
    rows = index["tasks"]
    ns.require(len(rows) == 48 and len({row["run_id"] for row in rows}) == 48
               and set(row["run_id"] for row in rows) == set(template["original_run_ids"]),
               "Deployment must preserve the original forty-eight window identities")
    plan = ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC
    for name, digest in index["implementation_sha256"].items():
        ns.require(PurePosixPath(name).name == name and name.endswith(".py")
                   and ns.sha(resolve_path(plan + "/implementation/" + name, resolve)) == digest,
                   "Original diagnostic implementation bytes changed")
    read_bound(index["frozen_inputs"], resolve)
    task = read_bound(task_binding, resolve)
    ns.require(task.get("schema") == "g50p-risk-diagnostic-task-v1"
               and task.get("action") in ("run_pair", "verify_pair", "verify_pilots", "summarize_diagnostic")
               and task.get("implementation_sha256") == index["implementation_sha256"]
               and task.get("budget_policy") == index["budget_policy"],
               "Composer allows only unchanged indexed main/verifier/original metadata-audit actions")
    ns.require(PurePosixPath(task_binding["path"]).parent == PurePosixPath(plan + "/tasks/" + template["new_task_revision"]),
               "Current task must be in the explicit new technical revision")
    status = ns.metadata(resolve_path(plan + "/status.json", resolve))
    ns.require(set(template["accepted_run_ids"]) <= set(status.get("accepted", {})),
               "Originally accepted pair preservation changed")
    if task["action"] in ("verify_pilots", "summarize_diagnostic"):
        planned = check_audit_task(index, task, task_binding, status, resolve=resolve)
        config = copy.deepcopy(template)
        config["approved_tasks"], config["approved_stores"] = [copy.deepcopy(task_binding)], []
        config["dispatch_provenance"] = dict(deployment=copy.deepcopy(deployment_binding),
            template=copy.deepcopy(deployment["template"]), index=copy.deepcopy(deployment["index"]),
            task=copy.deepcopy(task_binding), action=task["action"], run_id=task["run_id"], planned_run_ids=planned)
        ns.validate_config(config, scientific=True, inside_namespace=resolve is None)
        return config
    ns.require(task.get("index_binding") == index["frozen_inputs"], "Current main/verifier frozen-input binding differs")
    run_binding = task_binding if task["action"] == "run_pair" else task["run_task"]
    row = exact_run(index, run_binding)
    run_task = task if task["action"] == "run_pair" else read_bound(run_binding, resolve)
    code_allowed = (run_task.get("implementation_sha256") == current_code if task["action"] == "run_pair"
                    else same_scientific_code(run_task.get("implementation_sha256"), original_code,index=index,resolve=resolve))
    ns.require(run_task.get("action") == "run_pair" and run_task.get("run_id") == row["run_id"]
               and code_allowed
               and run_task.get("budget_policy") == index["budget_policy"]
               and run_task.get("index_binding") == index["frozen_inputs"]
               and run_task.get("producer") == template["original_producer"]
               and run_task.get("output_dir") == plan + "/pairs/" + row["run_id"],
               "Current indexed main changed producer, identity, inputs, policy, or output root")
    accepted = set(status.get("accepted", {}))
    ns.require(set(template["accepted_run_ids"]) <= accepted and row["run_id"] not in accepted,
               "Accepted pair preservation changed or this pair is already accepted")
    config = copy.deepcopy(template)
    config["approved_tasks"] = [copy.deepcopy(task_binding)]
    config["approved_stores"] = []
    if task["action"] == "run_pair":
        ns.require(not resolve_path(run_task["output_dir"] + "/report.json", resolve).exists(),
                   "A saved main report must be independently verified, never replayed")
        identity = parent_identity(run_task, resolve)
        policy = copy.deepcopy(template["store_policies"][row["run_id"]])
        config["approved_stores"] = [dict(root=run_task["output_dir"] + "/isolated_durable", identity=identity, policy=policy)]
    else:
        ns.require(task.get("index_path") == deployment["index"]["path"], "Verifier index path differs")
        check_saved_main(task, run_binding, run_task, resolve,
                         expected_verify_run_id=row["run_id"] + "_verify_" + deployment["index"]["sha256"][:12])
    config["dispatch_provenance"] = dict(deployment=copy.deepcopy(deployment_binding),
        template=copy.deepcopy(deployment["template"]), index=copy.deepcopy(deployment["index"]),
        task=copy.deepcopy(task_binding), action=task["action"], run_id=row["run_id"])
    # Full original namespace/fence/ledger-prefix/policy checks remain mandatory.
    ns.validate_config(config, scientific=True, inside_namespace=resolve is None)
    return config


def write_config_version(config, *, resolve=None):
    """Publish once under config_versions; never overwrite another version."""
    provenance = config["dispatch_provenance"]
    ns.require(config["approved_tasks"] == [provenance["task"]], "Config must approve only its composed task")
    raw = (json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    logical = config["paths"]["logical_control_root"] + "/config_versions/" + provenance["task"]["sha256"] + "_" + digest + ".json"
    ns.config_location(config, logical)
    path = resolve_path(logical, resolve)
    ns.require(path.parent.is_dir() and path.parent.resolve() == path.parent.absolute(),
               "Explicit owned config_versions directory must already exist without aliases")
    if path.exists():
        ns.require(not path.is_symlink() and path.read_bytes() == raw, "Immutable config version already differs")
        return dict(path=logical, sha256=digest)
    pending = path.with_name(".pending-" + uuid.uuid4().hex)
    descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    try:
        os.link(pending, path)  # atomic publication; no replacement of an older version
    except FileExistsError:
        ns.require(not path.is_symlink() and path.read_bytes() == raw, "Immutable config version already differs")
    pending.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=logical, sha256=digest)
