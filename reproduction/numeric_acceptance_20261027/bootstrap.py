"""Metadata-only migration bootstrap; scientific entry requires final receipts.

The extension preserves host_entry.execute and every original admission,
registration, budget, and strict identity check.  A narrow post-import hook
adapts only old('durable_state') after the original producer loader has verified
its path/SHA. No original scientific source file is rewritten.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

try:
    from . import namespace_runner as ns
except ImportError:
    import namespace_runner as ns


def policy_factory(config, adapter):
    approvals = {str(Path(item["root"])): item for item in config["approved_stores"]}
    ns.require(len(approvals) == len(config["approved_stores"]), "Duplicate writable store approval")
    def approve(root, identity):
        item = approvals.get(str(root))
        if item is None or identity != item["identity"]:
            return None
        return adapter.WritableRootPolicy(root, item["policy"])
    return approve


def wrap_producer_loader(pair_module, adapter, approve, durable_binding):
    """Install a single explicit hook; call the original verifier before use."""
    ns.require(not hasattr(pair_module, "_sysu_original_load_producer"), "Producer hook already installed")
    original = pair_module.load_producer
    def load_producer(task):
        verified_get = original(task)
        def old(name):
            module = verified_get(name)  # frozen path, full50 hash coverage, shadow check
            if name == "durable_state":
                ns.require(Path(module.__file__).resolve() == Path(durable_binding["path"])
                           and ns.sha(module.__file__) == durable_binding["sha256"],
                           "Storage hook received a different original durable module")
                adapter.install_on_module(module, approve)
            return module
        return old
    pair_module._sysu_original_load_producer = original
    pair_module.load_producer = load_producer


class PairStorageFinder(importlib.abc.MetaPathFinder):
    """No early scientific import: host_entry imports pair_runner after budget."""
    def __init__(self, pair_path, pair_sha256, adapter, approve, durable_binding, overrides=None):
        self.path, self.digest = Path(pair_path), pair_sha256
        self.adapter, self.approve, self.durable = adapter, approve, durable_binding
        self.overrides=overrides or {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ("pair_runner","verify_pair"):
            return None
        item=self.overrides.get(fullname)
        if item is None:return None
        source=Path(item["path"]); digest=item["sha256"]
        ns.require(ns.sha(source)==digest,"Revised comparison module binding changed")
        finder = self
        class Loader(importlib.machinery.SourceFileLoader):
            def exec_module(self, module):
                super().exec_module(module)
                ns.require(Path(module.__file__).resolve() == source
                           and ns.sha(module.__file__) == digest,
                           "Paired runner shadowed during migration import")
                if fullname=="pair_runner":wrap_producer_loader(module, finder.adapter, finder.approve, finder.durable)
        return importlib.util.spec_from_file_location(fullname, source, loader=Loader(fullname, str(source)))


def load_bound_module(name, path, digest):
    ns.require(name not in sys.modules, "Pre-existing module would bypass explicit migration binding: " + name)
    path = Path(path)
    ns.require(ns.sha(path) == digest, "Bound bootstrap import changed: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    ns.require(Path(module.__file__).resolve() == path and ns.sha(path) == digest,
               "Bound bootstrap import shadowed: " + name)
    return module


def prepare_runtime(config):
    """Read bounded metadata and install hook; no task execution or Torch import."""
    ns.validate_config(config, scientific=True, inside_namespace=True)
    plan = Path(ns.LOGICAL_ROOT) / ns.DIAGNOSTIC
    index = ns.metadata(ns.binding(config, config["diagnostic_index"], inside_namespace=True, scientific=True))
    ns.require(index.get("schema") == "g50p-risk-diagnostic-index-v1"
               and index.get("protocol") == config["effective_protocol"]
               and index.get("budget_policy") == config["budget_policy"],
               "Effective diagnostic index/protocol/policy differs from final migration config")
    implementation = index["implementation_sha256"]
    for name in ("host_entry.py", "budget.py", "pair_runner.py"):
        ns.require(ns.sha(plan / "implementation" / name) == implementation[name],
                   "Original diagnostic entry/import binding changed: " + name)
    ns.require(not ({"pair_runner", "durable_state", "host_entry", "budget"} & set(sys.modules)),
               "Original diagnostic modules must be fresh in this process")
    sys.path.insert(0, str(Path(ns.LOGICAL_ROOT) / "spnv2"))
    adapter = importlib.import_module("reproduction.sysu_migration_20261027.storage_adapter_v1")
    policy = importlib.import_module("reproduction.storage_management_20261027.storage_policy")
    for name, module in (("storage_adapter", adapter), ("storage_policy", policy)):
        item = config["code_bindings"][name]
        ns.require(Path(module.__file__).resolve() == Path(item["path"])
                   and ns.sha(module.__file__) == item["sha256"], "Extension module was shadowed: " + name)
    for item in config["approved_stores"]:
        policy.validate(item["policy"])
    approve = policy_factory(config, adapter)
    revision=ns.metadata(ns.binding(config,index["numeric_acceptance"],inside_namespace=True,scientific=True))
    for item in revision["overrides"].values():ns.binding(config,item,inside_namespace=True,scientific=True)
    ns.binding(config,revision["comparator"],inside_namespace=True,scientific=True)
    sys.path.insert(0,str(Path(revision["comparator"]["path"]).parent))
    finder = PairStorageFinder(plan / "implementation/pair_runner.py", implementation["pair_runner.py"],
                               adapter, approve, config["original_durable_state"],revision["overrides"])
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, str(plan / "implementation"))
    budget = load_bound_module("budget", plan / "implementation/budget.py", implementation["budget.py"])
    host = load_bound_module("host_entry", plan / "implementation/host_entry.py", implementation["host_entry.py"])
    return host, finder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe", action="store_true")
    group.add_argument("--execute-host-entry")
    parser.add_argument("--task-json")
    parser.add_argument("--index")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    ns.require(ns.sha(args.config) == args.config_sha256, "Migration runtime config changed")
    config = ns.metadata(args.config)
    ns.validate_config(config, scientific=False, inside_namespace=True)
    ns.config_location(config, str(Path(args.config).resolve()))
    for name, path in (("bootstrap", Path(__file__).resolve()), ("namespace_runner", Path(ns.__file__).resolve())):
        ns.require(str(path) == config["code_bindings"][name]["path"]
                   and ns.sha(path) == config["code_bindings"][name]["sha256"],
                   "Bootstrap/launcher self identity differs")
    if args.probe:
        ns.require(not any((args.task_json, args.index, args.log_path)), "Probe accepts no task arguments")
        print(json.dumps({"status": "metadata_namespace_probe_passed", "model_forwards": 0,
                          "experimental_dispatch": False, "torch_imported": "torch" in sys.modules,
                          "storage_hook_installed": False}))
        return 0
    plan = Path(ns.LOGICAL_ROOT) / ns.DIAGNOSTIC
    ns.require(args.execute_host_entry == str(plan / "implementation/host_entry.py")
               and args.index == str(plan / "index.json") and args.task_json and args.log_path,
               "Only the original host_entry with explicit task/index/log is allowed")
    ns.require(args.execute_host_entry in sys.argv, "Original host_entry must remain in the exact process command")
    approved = [item for item in config["approved_tasks"] if item["path"] == args.task_json]
    ns.require(len(approved) == 1, "Task lacks an explicit migration approval")
    ns.binding(config, approved[0], inside_namespace=True, scientific=True)
    ns.require(Path(args.log_path).parent == plan / "logs", "Log must remain in the owned diagnostic plan")
    host, finder = prepare_runtime(config)
    try:
        # No substitutions for two snapshots, four locks, exact launch records,
        # budget.begin_attempt, watchdog, producer hashes or independent proofs.
        return host.execute(args.task_json, args.index, args.log_path)["exit_code"]
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


if __name__ == "__main__":
    raise SystemExit(main())
