"""Affected stdlib metadata contracts for dynamic single-task composition."""
import hashlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.sysu_migration_20261027 import dispatch_config as dc
from reproduction.sysu_migration_20261027 import namespace_runner as ns


class ComposerTests(unittest.TestCase):
    def fixture(self, temporary, historical_main=False):
        logical = ns.LOGICAL_ROOT
        plan = logical + "/" + ns.DIAGNOSTIC
        control = logical + "/spnv2_runs/migrations/sysu_20261027/runtime_control"
        def resolve(path):
            self.assertTrue(path.startswith(logical + "/"))
            return Path(temporary) / path[len(logical) + 1:]
        def publish(path, value):
            target = resolve(path); target.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps(value, sort_keys=True) + "\n").encode()
            target.write_bytes(raw)
            return dict(path=path, sha256=hashlib.sha256(raw).hexdigest())
        frozen = publish(plan + "/frozen_inputs.json", {"frozen": True})
        policy = publish(plan + "/revisions/time_policy.json", {"time_hard_stop_enabled": False})
        protocol = publish(plan + "/revisions/protocol.json", {"frozen": True})
        source = publish(plan + "/implementation/host_entry.py", {"fixture": "stdlib bytes only"})
        manager = publish(plan + "/implementation/manager.py", {"fixture": "new metadata manager only"})
        implementation = {"host_entry.py": source["sha256"], "manager.py": manager["sha256"]}
        original_implementation = {**implementation, "manager.py": "6" * 64}
        run_implementation = {**implementation, "manager.py": "7" * 64} if historical_main else implementation
        producer_root = logical + "/spnv2_runs/plans/odr003_round2_20261023"
        producer = dict(implementation_dir=producer_root + "/implementation", commit="3ef-original")
        original = publish(producer_root + "/tasks/original.json", dict(run_id="original_P", protocol={"source_sha256": "1" * 64},
            baseline_task={"bindings": {"stream_manifest": {"sha256": "2" * 64}}}))
        identity = dict(run_id="original_P", source_sha256="1" * 64, config_sha256=original["sha256"],
                        manifest_sha256="2" * 64, additional_original_field="retained")
        cursors = dict(observed=64, candidates=1, commits=1, rollbacks=0, next_index=64)
        generation = producer_root + "/runs/original_P/durable/checkpoints/generation-00000000000000000001"
        parent = publish(generation + "/metadata.json", dict(format_version=1, generation=1, identity=identity,
            state_sha256="3" * 64, scientific_cursors=cursors))
        run_id = "current_pair"
        run_task = dict(schema="g50p-risk-diagnostic-task-v1", action="run_pair", run_id=run_id,
            output_dir=plan + "/pairs/" + run_id, implementation_sha256=run_implementation, budget_policy=policy,
            index_binding=frozen, producer=producer, panel={"frozen": True},
            selection={"original_task": original, "parent_checkpoint": dict(directory=generation, generation=1,
                metadata_sha256=parent["sha256"], state_sha256="3" * 64, cursors=cursors)})
        run_binding = publish(plan + "/tasks/sysu_new/current_pair.json", run_task)
        ids = [run_id] + ["unchanged_" + str(n) for n in range(47)]
        rows = [dict(run_id=run_id, task_path=run_binding["path"], task_sha256=run_binding["sha256"])]
        rows += [dict(run_id=name, task_path=plan + "/tasks/old/" + name + ".json", task_sha256="4" * 64) for name in ids[1:]]
        extensions = {"dispatch_config": {"path": logical + "/spnv2/reproduction/sysu_migration_20261027/dispatch_config.py", "sha256": "5" * 64}}
        pilot_template = dict(schema="g50p-risk-diagnostic-task-v1", action="verify_pilots", run_id="fixed_pilot_audit",
            implementation_sha256=implementation, budget_policy=policy, frozen_inputs=frozen,
            proof_path=plan + "/monitoring/pilot_proof.json")
        summary_template = dict(schema="g50p-risk-diagnostic-task-v1", action="summarize_diagnostic", run_id="fixed_summary",
            implementation_sha256=implementation, budget_policy=policy, frozen_inputs=frozen, output_dir=plan + "/delivery")
        index_binding = publish(plan + "/index.json", dict(schema="g50p-risk-diagnostic-index-v1", tasks=rows,
            implementation_sha256=implementation, protocol=protocol, budget_policy=policy, frozen_inputs=frozen,
            migration_extensions=extensions, sysu_task_revision="sysu_new", pilot_admission_template=pilot_template,
            summary_template=summary_template, audit_task_paths={"pilot_admission": plan + "/tasks/sysu_new/pilot.json",
                                                               "summary": plan + "/tasks/sysu_new/summary.json"}))
        template = dict(schema=ns.SCHEMA, migration_id=ns.MIGRATION, approved_tasks=[], approved_stores=[],
            diagnostic_index=index_binding, code_bindings=extensions, original_implementation_sha256=original_implementation,
            effective_protocol=protocol, budget_policy=policy, frozen_inputs=frozen, original_run_ids=ids,
            accepted_run_ids=["accepted_original"], original_producer=producer, new_task_revision="sysu_new",
            paths={"logical_control_root": control},
            source_fence_receipt={"unchanged": "common binding"}, budget_journal={"unchanged": "common prefix"},
            store_policies={run_id: dict(schema="spnv2-storage-policy-v1", retain_latest=2,
                estimated_peak_bytes=1000000, protected_generations={})})
        template_binding = publish(control + "/template.json", template)
        deployment = publish(control + "/deployment.json", dict(schema=dc.DEPLOYMENT_SCHEMA, migration_id=ns.MIGRATION,
            template=template_binding, index=index_binding))
        publish(plan + "/status.json", dict(accepted={"accepted_original": {}}))
        resolve(control + "/config_versions").mkdir()
        return dict(resolve=resolve, publish=publish, plan=plan, deployment=deployment, run_task=run_task,
                    run_binding=run_binding, identity=identity, template=template, current_implementation=implementation)

    def test_main_composes_one_store_with_full_metadata_identity_and_immutable_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary)
            with patch.object(ns, "validate_config") as validate:
                config = dc.compose_single_task(f["deployment"], f["run_binding"], resolve=f["resolve"])
            validate.assert_called_once_with(config, scientific=True, inside_namespace=False)
            self.assertEqual(config["approved_tasks"], [f["run_binding"]])
            self.assertEqual(config["approved_stores"][0]["identity"], f["identity"])
            self.assertEqual(config["source_fence_receipt"], f["template"]["source_fence_receipt"])
            version = dc.write_config_version(config, resolve=f["resolve"])
            self.assertEqual(dc.write_config_version(config, resolve=f["resolve"]), version)
            self.assertEqual(ns.sha(f["resolve"](version["path"])), version["sha256"])
            f["publish"](f["run_task"]["output_dir"] + "/report.json", {"already_saved": True})
            with self.assertRaisesRegex(ns.MigrationConfigError, "never replayed"):
                dc.compose_single_task(f["deployment"], f["run_binding"], resolve=f["resolve"])

    def test_dynamic_verifier_binds_new_saved_report_without_changing_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary, historical_main=True)
            run, binding, plan = f["run_task"], f["run_binding"], f["plan"]
            report = f["publish"](run["output_dir"] + "/report.json", dict(schema="g50p-risk-diagnostic-pair-report-v1",
                status="pair_main_complete_pending_independent_verification", run_id=run["run_id"],
                task_sha256=binding["sha256"], selection=run["selection"], producer=run["producer"], panel=run["panel"]))
            launch = plan + "/launches/nonce/launch.json"
            outcome = plan + "/launches/nonce/outcome.json"
            log = plan + "/logs/current_pair_nonce.log"
            f["publish"](launch, dict(run_id=run["run_id"], task_sha256=binding["sha256"]))
            f["publish"](outcome, dict(run_id=run["run_id"], status="pair_main_complete", exit_code=0,
                report=report["path"], report_sha256=report["sha256"]))
            exit_path = f["resolve"](log + ".exitcode"); exit_path.parent.mkdir(parents=True); exit_path.write_text("0\n")
            deployment = dc.read_bound(f["deployment"], f["resolve"])
            task = dict(schema="g50p-risk-diagnostic-task-v1", action="verify_pair",
                run_id=run["run_id"] + "_verify_" + deployment["index"]["sha256"][:12],
                implementation_sha256=f["current_implementation"], budget_policy=run["budget_policy"],
                index_binding=run["index_binding"], index_path=plan + "/index.json", run_task=binding,
                proof_path=run["output_dir"] + "/proof.json", execution_receipt=dict(run_id=run["run_id"], task=binding,
                    report=report, launch_record_path=launch, outcome_path=outcome, log_path=log, exit_code_path=log + ".exitcode"))
            verify = f["publish"](plan + "/tasks/sysu_new/current_pair_verify.json", task)
            with patch.object(ns, "validate_config"):
                config = dc.compose_single_task(f["deployment"], verify, resolve=f["resolve"])
            self.assertEqual(config["approved_tasks"], [verify])
            self.assertEqual(config["approved_stores"], [])
            self.assertNotIn("accepted", config)
            f["publish"](outcome, dict(run_id=run["run_id"], status="failed", exit_code=1))
            with self.assertRaisesRegex(ns.MigrationConfigError, "bindings disagree"):
                dc.compose_single_task(f["deployment"], verify, resolve=f["resolve"])

    def test_exact_existing_audit_templates_accept_only_fixed_receipt_scope(self):
        for action in ("verify_pilots", "summarize_diagnostic"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as temporary:
                f = self.fixture(temporary)
                deployment = dc.read_bound(f["deployment"], f["resolve"])
                index = dc.read_bound(deployment["index"], f["resolve"])
                pilot = action == "verify_pilots"
                kind = "pilot_admission" if pilot else "summary"
                selected = index["tasks"][:2] if pilot else index["tasks"]
                completed = selected if pilot else selected[:1]
                main = {row["run_id"]: dict(run_id=row["run_id"], task={"path": row["task_path"], "sha256": row["task_sha256"]})
                        for row in completed}
                accepted = {rid: {"preserved_independent_receipt_fixture": rid} for rid in main}
                f["publish"](f["plan"] + "/status.json", dict(main=main, accepted={"accepted_original": {}, **accepted}))
                planned = [row["run_id"] for row in selected]
                receipts = dict(schema="g50p-diagnostic-execution-receipts-v1", planned_run_ids=planned,
                    runs=list(main.values()), accepted=accepted,
                    missing_main_run_ids=[rid for rid in planned if rid not in main],
                    missing_independent_acceptance_run_ids=[rid for rid in planned if rid not in accepted])
                receipt_path = f["plan"] + ("/pilot_receipts_fixture.json" if pilot else "/final_receipts_fixture.json")
                receipt = f["publish"](receipt_path, receipts)
                task = copy.deepcopy(index[kind + "_template"])
                task.update(execution_receipts=receipt, index_path=deployment["index"]["path"])
                task_binding = f["publish"](index["audit_task_paths"][kind], task)
                with patch.object(ns, "validate_config"):
                    config = dc.compose_single_task(f["deployment"], task_binding, resolve=f["resolve"])
                self.assertEqual(config["approved_tasks"], [task_binding])
                self.assertEqual(config["approved_stores"], [])
                self.assertEqual(config["dispatch_provenance"]["planned_run_ids"], planned)
                bad = {**task, "unapproved_scope_override": True}
                bad_binding = f["publish"](task_binding["path"], bad)
                with self.assertRaisesRegex(ns.MigrationConfigError, "exact bound"):
                    dc.compose_single_task(f["deployment"], bad_binding, resolve=f["resolve"])
                changed = {**receipts, "planned_run_ids": planned + ["unapproved_extra_window"]}
                task["execution_receipts"] = f["publish"](receipt_path, changed)
                changed_binding = f["publish"](task_binding["path"], task)
                with self.assertRaisesRegex(ns.MigrationConfigError, "fixed planned IDs"):
                    dc.compose_single_task(f["deployment"], changed_binding, resolve=f["resolve"])


if __name__ == "__main__":
    unittest.main()
