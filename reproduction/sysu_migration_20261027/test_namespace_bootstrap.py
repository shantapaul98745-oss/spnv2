"""Bounded stdlib metadata tests: no namespace execution or scientific imports."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.sysu_migration_20261027 import namespace_runner as ns
from reproduction.sysu_migration_20261027 import bootstrap as boot


def digest(data):
    return hashlib.sha256(data).hexdigest()


class NamespaceTests(unittest.TestCase):
    def minimal(self, root):
        migration = root + "/spnv2_runs/migrations/sysu_20261027"
        logical_control = ns.LOGICAL_ROOT + "/spnv2_runs/migrations/sysu_20261027/runtime_control"
        runtime_tmp = tempfile.TemporaryDirectory(prefix="spnv2_sysu_metadata_")
        self.addCleanup(runtime_tmp.cleanup)
        screen = Path(runtime_tmp.name) / "screen"
        screen.mkdir(mode=0o700)
        paths = dict(target_root=root, source_snapshot=migration + "/source_snapshot",
                     repository=root + "/spnv2", environment="/home/ps/miniconda3/envs/SPNv2",
                     logical_environment="/home/ps/anaconda3/envs/SPNv2",
                     dataset="/home/ps/Downloads/DataSet/speedplus",
                     logical_dataset="/home/DATA/Downloads/DataSet/speedplus",
                     tmp=runtime_tmp.name, cache="/home/ps/.cache/spnv2_sysu_metadata_test",
                     logical_root=ns.LOGICAL_ROOT, control_root=migration + "/runtime_control",
                     logical_control_root=logical_control)
        code = {}
        for key, relative in (("namespace_runner", "sysu_migration_20261027/namespace_runner.py"),
                              ("bootstrap", "sysu_migration_20261027/bootstrap.py"),
                              ("dispatch_config", "sysu_migration_20261027/dispatch_config.py"),
                              ("storage_adapter", "sysu_migration_20261027/storage_adapter_v1.py"),
                              ("storage_policy", "storage_management_20261027/storage_policy.py")):
            target = Path(root) / "spnv2/reproduction" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"# metadata fixture\n")
            code[key] = dict(path=ns.LOGICAL_ROOT + "/spnv2/reproduction/" + relative,
                             sha256=digest(target.read_bytes()))
        return dict(schema=ns.SCHEMA, migration_id=ns.MIGRATION, paths=paths,
                    code_bindings=code, source_dispatch_fenced=False, screen_socket_dir=str(screen))

    def test_probe_is_read_only_and_arbitrary_command_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            argv = ns.build_bwrap([], config, config_sha256="a" * 64)
            self.assertEqual(argv[6:12], ["--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev"])
            self.assertIn("--probe", argv)
            self.assertNotIn("--execute-host-entry", argv)
            writable = [argv[n + 1] for n, value in enumerate(argv) if value == "--bind"]
            self.assertEqual(writable, [config["paths"]["tmp"], config["paths"]["cache"]])
            with self.assertRaises(ns.MigrationConfigError):
                ns.build_bwrap(["python", "model.py"], config, config_sha256="a" * 64)

    def test_scientific_mode_fails_before_dispatch_without_fence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            with self.assertRaisesRegex(ns.MigrationConfigError, "source fence"):
                ns.build_bwrap([], config, scientific=True, config_sha256="a" * 64)

    def test_owned_project_screen_sockets_are_exported_without_changing_sty(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            argv = ns.build_bwrap([], config, config_sha256="a" * 64)
            self.assertEqual(argv[argv.index("SCREENDIR") + 1], config["screen_socket_dir"])
            self.assertNotIn("STY", argv)
            Path(config["screen_socket_dir"]).chmod(0o755)
            with self.assertRaisesRegex(ns.MigrationConfigError, "mode0700"):
                ns.build_bwrap([], config, config_sha256="a" * 64)

    def test_writable_plan_has_readonly_history_and_shared_lock_backing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            config.update(new_task_revision="sysu_new", accepted_run_ids=["accepted_pair"])
            plan = ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC
            rows = ns.mounts(config, True)
            self.assertIn((False, config["paths"]["source_snapshot"] + "/" + ns.DIAGNOSTIC + "/pairs/accepted_pair",
                           plan + "/pairs/accepted_pair"), rows)
            self.assertEqual([target for writable, _, target in rows if writable and target.endswith("pipeline.lock")],
                             [ns.LOGICAL_ROOT + "/spnv2_runs/plans/" + name + "/pipeline.lock" for name in ns.LOCK_PLANS])
            for writable, source, target in rows:
                if writable and target.endswith("pipeline.lock"):
                    self.assertEqual(source, temporary + target[len(ns.LOGICAL_ROOT):])
            self.assertNotIn((True, config["paths"]["source_snapshot"], ns.LOGICAL_ROOT), rows)

    def test_frozen_image_cache_uses_private_domain_and_exact_readonly_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            domain = config["paths"]["logical_dataset"] + "/sunlamp"
            source = temporary + "/spnv2_runs/frozen/required_images_20261027/sunlamp/images_768x512_RGB"
            config["frozen_image_cache"] = dict(source=source, target=domain + "/images_768x512_RGB",
                                                manifest={}, receipt={})
            argv = ns.build_bwrap([], config, config_sha256="a" * 64)
            overlay = ["--tmpfs", domain,
                       "--ro-bind", config["paths"]["dataset"] + "/sunlamp/images", domain + "/images",
                       "--ro-bind", config["paths"]["dataset"] + "/sunlamp/test.json", domain + "/test.json",
                       "--ro-bind", source, domain + "/images_768x512_RGB"]
            position = argv.index(domain)
            self.assertEqual(argv[position - 1:position - 1 + len(overlay)], overlay)
            config["frozen_image_cache"]["target"] = domain + "/images"
            with self.assertRaisesRegex(ns.MigrationConfigError, "exact approved"):
                ns.build_bwrap([], config, config_sha256="a" * 64)

    def test_frozen_initialization_cache_alias_and_versioned_config_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            source = config["paths"]["source_snapshot"] + "/spnv2_runs/frozen/sysu_external_cache/" + ns.INITIALIZATION_NAME
            target = "/home/ps/.cache/torch/hub/checkpoints/" + ns.INITIALIZATION_NAME
            config["initialization_cache"] = dict(source=source, target=target, sha256=ns.INITIALIZATION_SHA256, receipt={})
            version = config["paths"]["logical_control_root"] + "/config_versions/task_config.json"
            argv = ns.build_bwrap([], config, config_sha256="a" * 64, config_path=version)
            position = argv.index(source)
            self.assertEqual(argv[position - 1:position + 2], ["--ro-bind", source, target])
            self.assertIn(version, argv)
            self.assertEqual(argv[argv.index("TORCH_HOME") + 1], "/home/ps/.cache/torch")
            config["initialization_cache"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ns.MigrationConfigError, "unchanged frozen artifact"):
                ns.build_bwrap([], config, config_sha256="a" * 64)

    def test_bound_metadata_manager_only_writes_config_versions_control_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            config.update(new_task_revision="sysu_new", accepted_run_ids=[])
            p = config["paths"]
            plan = ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC
            def publish(logical, value):
                path = ns.physical(config, logical, scientific=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, sort_keys=True))
                return dict(path=logical, sha256=ns.sha(path))
            manager_path = plan + "/implementation/manager.py"
            manager = publish(manager_path, {"metadata_manager_fixture": True})
            index = publish(plan + "/index.json", dict(migration_extensions=config["code_bindings"],
                implementation_sha256={"manager.py": manager["sha256"]}))
            template = dict(path=p["logical_control_root"] + "/template.json", sha256="b" * 64)
            deployment = publish(p["logical_control_root"] + "/deployment.json", dict(schema="spnv2-sysu-deployment-v1",
                migration_id=ns.MIGRATION, index=index, template=template))
            config.update(diagnostic_index=index, dispatch_provenance=dict(deployment=deployment, template=template))
            version = publish(p["logical_control_root"] + "/config_versions/first.json", config)
            with patch.object(ns, "validate_config"):
                argv = ns.build_manager_bwrap(config, deployment, version)
            self.assertEqual(argv[-7:], [manager_path, "--index", index["path"], "--deployment-manifest", deployment["path"],
                "--deployment-sha256", deployment["sha256"]])
            controls = [(argv[n + 1], argv[n + 2]) for n, value in enumerate(argv) if value == "--bind"
                        and argv[n + 1].startswith(p["control_root"])]
            self.assertEqual(controls, [(p["control_root"] + "/config_versions", p["logical_control_root"] + "/config_versions")])

    def test_canonical_asset_root_requires_exact_bound_rename_receipt_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(ns, "TARGET_ROOT", temporary):
            config = self.minimal(temporary)
            config["paths"]["source_snapshot"] = temporary
            control = config["paths"]["logical_control_root"]
            def publish(logical, value):
                path = ns.physical(config, logical)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, sort_keys=True))
                return dict(path=logical, sha256=ns.sha(path))
            manifest = publish(control + "/asset_transfer_manifest.json", {"files": [{"relative_path": "owned/file1"}, {"relative_path": "owned/file2"}]})
            receipt = dict(schema="spnv2-canonical-asset-layout-v1", status="verified_renamed_immutable_assets",
                target_root=temporary, source_staging=temporary + "/spnv2_runs/migrations/sysu_20261027/source_snapshot",
                transfer_manifest_sha256=manifest["sha256"], verified_file_count=2,
                no_file_content_changes=True, source_dispatch_fenced=True)
            config.update(asset_transfer_manifest=manifest, asset_layout_receipt=publish(control + "/layout.json", receipt))
            ns.validate_config(config)  # Legacy probe behavior also supports exact canonical U.
            ns.check_asset_layout(config)
            config["asset_layout_receipt"] = publish(control + "/layout.json", {**receipt, "no_file_content_changes": False})
            with self.assertRaisesRegex(ns.MigrationConfigError, "rename receipt"):
                ns.check_asset_layout(config)
            config["paths"]["source_snapshot"] = temporary + "/arbitrary_other_root"
            with self.assertRaisesRegex(ns.MigrationConfigError, "Source/control roots"):
                ns.validate_config(config)

    def test_final_journal_prefix_survives_new_costs_and_unresolved_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.minimal(temporary)
            config.update(new_task_revision="sysu_new", accepted_run_ids=[])
            path = Path(config["paths"]["source_snapshot"]) / ns.DIAGNOSTIC / "budget/journal.jsonl"
            path.parent.mkdir(parents=True)
            previous, rows = "0" * 64, []
            def append(kind, payload):
                nonlocal previous
                row = dict(sequence=len(rows) + 1, previous_sha256=previous, kind=kind, payload=payload, wall_time=0)
                previous = digest(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
                row["sha256"] = previous
                rows.append(json.dumps(row, sort_keys=True).encode() + b"\n")
                path.write_bytes(b"".join(rows))
            append("initial_cost_imported", dict(physical_forwards_upper=4748))
            prefix = path.read_bytes()
            config["budget_journal"] = dict(path=ns.LOGICAL_ROOT + "/" + ns.DIAGNOSTIC + "/budget/journal.jsonl", sha256=digest(prefix))
            handoff = dict(budget_journal_bytes=len(prefix))
            append("attempt_started", dict(attempt_id="target1"))
            with self.assertRaisesRegex(ns.MigrationConfigError, "unresolved"):
                ns.check_budget_journal(config, handoff)
            append("forward_reserved", dict(count=1))
            append("attempt_closed", dict(attempt_id="target1"))
            self.assertEqual(ns.check_budget_journal(config, handoff), path)
            path.write_bytes(path.read_bytes().replace(b"4748", b"4747", 1))
            with self.assertRaisesRegex(ns.MigrationConfigError, "prefix"):
                ns.check_budget_journal(config, handoff)


class HookTests(unittest.TestCase):
    def test_original_loader_verifies_before_explicit_storage_hook(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "durable_state.py"
            path.write_bytes(b"# frozen stdlib fixture\n")
            events = []
            module = SimpleNamespace(__file__=str(path), DurableRunStore=object)
            def original(task):
                events.append("full producer binding")
                def get(name):
                    events.append("path/hash shadow verification:" + name)
                    return module
                return get
            pair = SimpleNamespace(load_producer=original)
            adapter = SimpleNamespace(install_on_module=lambda old, approve: events.append("explicit storage extension"))
            boot.wrap_producer_loader(pair, adapter, object(), dict(path=str(path), sha256=digest(path.read_bytes())))
            old = pair.load_producer({})
            self.assertEqual(events, ["full producer binding"])
            self.assertIs(old("durable_state"), module)
            self.assertEqual(events[-2:], ["path/hash shadow verification:durable_state", "explicit storage extension"])
            path.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ns.MigrationConfigError, "different original"):
                old("durable_state")
            self.assertEqual(events.count("explicit storage extension"), 1)

    def test_exact_root_and_original_identity_are_both_required(self):
        config = dict(approved_stores=[dict(root="/tmp/owned/pair/isolated_durable", identity={"run_id": "original", "additional_original_field": "kept"}, policy={"explicit": True})])
        adapter = SimpleNamespace(WritableRootPolicy=lambda root, policy: (root, policy))
        approve = boot.policy_factory(config, adapter)
        self.assertIsNone(approve(Path("/tmp/owned"), {"run_id": "original", "additional_original_field": "kept"}))
        self.assertIsNone(approve(Path("/tmp/owned/pair/isolated_durable"), {"run_id": "changed"}))
        self.assertIsNone(approve(Path("/tmp/owned/pair/isolated_durable"), {"run_id": "original"}))
        self.assertIsNotNone(approve(Path("/tmp/owned/pair/isolated_durable"), {"run_id": "original", "additional_original_field": "kept"}))


if __name__ == "__main__":
    unittest.main()
