"""CPU-only append revision tests; execute through the project's guarded screen."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PLAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN / "pipeline"))
import runner
from revision import canonical_sha256, extend_state
from task_entry import process_alive, process_identity


def spec(ident):
    return {
        "id": ident, "command": [sys.executable, "-B", "-s", "-c", "pass"],
        "cwd": str(PLAN), "session": "revision_test_" + ident,
        "log": str(PLAN / "logs" / ("revision_fixture_" + ident + ".log")),
        "env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"},
        "gpu": False, "expected_outputs": [str(PLAN / "reports" / (ident + ".json"))],
        "requires": [],
    }


class Reporter:
    write_failed = False

    def publish(self, force=False):
        return True


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.old = {
            "version": 1, "plan_id": PLAN.name, "scope": "phase1",
            "report_path": str(runner.REPORT),
            "tasks": [spec("accepted"), spec("active")], "backlog": ["remaining"],
        }
        self.previous = runner.initialize_state({}, self.old)
        first, second = self.previous["tasks"]
        first.update(status="accepted", accepted_at="accepted_at",
                     runtime={"phase": "verify", "result": "retained-result.json", "exit_code": 0},
                     history=[{"phase": "main", "pid": 42, "starttime": "123", "exit_code": 0}])
        second.update(status="running", started_at="started_at", runtime={
            "phase": "main", "record": "record.json", "result": "result.json",
            "command": second["command"], "session": second["session"],
            "log": second["log"], "tail_offset": 77, "observed_live": True,
        }, history=[])
        self.previous.update(completed=["accepted"], active="active",
                             current={"task_id": "active", "pid": 99, "starttime": "456",
                                      "screen_id": "fixture.screen", "process_state": "verified"},
                             manager={"pid": 77, "starttime": "888"}, status="running")
        self.new = copy.deepcopy(self.old)
        self.new["tasks"].append(spec("next"))
        self.new.update(scope="full", backlog=[])

    def extend(self):
        return extend_state(self.previous, self.old, self.new,
                            caller_context={"test": True}, timestamp="2026-10-17T14:00:00+08:00")

    def test_only_hash_changes_and_inputs_are_unmodified(self):
        originals = copy.deepcopy((self.previous, self.old, self.new))
        amended, audit = self.extend()
        expected = copy.deepcopy(self.previous)
        expected["manifest_sha256"] = runner.manifest_hash(self.new)
        self.assertEqual(amended, expected)
        self.assertEqual((self.previous, self.old, self.new), originals)
        amended["current"]["pid"] = -1
        audit["appended_tasks"][0]["command"].append("changed")
        self.assertEqual((self.previous, self.old, self.new), originals)

    def test_active_runtime_pid_current_and_manager_preserved(self):
        amended, _ = self.extend()
        self.assertEqual(amended["tasks"][1], self.previous["tasks"][1])
        for field in ("current", "completed", "manager", "active", "status"):
            self.assertEqual(amended[field], self.previous[field])

    def test_accepted_evidence_and_initialize_state_compatibility(self):
        amended, _ = self.extend()
        state = runner.initialize_state(amended, self.new)
        self.assertEqual(state["tasks"][0], self.previous["tasks"][0])
        self.assertEqual(state["tasks"][1]["runtime"], self.previous["tasks"][1]["runtime"])
        self.assertEqual(state["completed"], ["accepted"])
        self.assertEqual(state["tasks"][1]["status"], "launching")
        self.assertEqual(state["current"]["pid"], 99)
        self.assertEqual(state["current"]["starttime"], "456")
        self.assertEqual(state["tasks"][2]["status"], "queued")
        self.assertNotIn("runtime", state["tasks"][2])
        self.assertEqual(state["scope"], "full")
        self.assertEqual(state["backlog"], [])

    def test_audit_is_complete_and_canonical_hash_matches_manager(self):
        amended, audit = self.extend()
        self.assertEqual(canonical_sha256(self.old), runner.manifest_hash(self.old))
        self.assertEqual(audit["original_tasks"], self.old["tasks"])
        self.assertEqual(audit["appended_tasks"], self.new["tasks"][2:])
        self.assertEqual(audit["old_task_count"], 2)
        self.assertEqual(audit["new_task_count"], 3)
        self.assertEqual(audit["accepted_task_ids"], ["accepted"])
        self.assertEqual(audit["active_task_id"], "active")
        self.assertEqual(audit["previous_state_sha256"], runner.manifest_hash(self.previous))
        self.assertEqual(audit["caller_context"], {"test": True})
        self.assertEqual(audit["new_manifest_sha256"], amended["manifest_sha256"])

    def test_prefix_definition_edits_rejected(self):
        for field, replacement in {
            "command": ["false"], "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "session": "changed", "log": "/changed", "cwd": "/changed",
            "requires": ["accepted"], "metadata": {"cfg": "/changed"},
            "verify_command": ["false"], "gpu": {"index": 0},
        }.items():
            with self.subTest(field=field):
                new = copy.deepcopy(self.new)
                new["tasks"][1][field] = replacement
                with self.assertRaises(ValueError):
                    extend_state(self.previous, self.old, new)

    def test_reorder_rejected(self):
        self.new["tasks"][:2] = list(reversed(self.new["tasks"][:2]))
        with self.assertRaises(ValueError):
            self.extend()

    def test_removal_rejected_even_with_longer_replacement(self):
        self.new["tasks"] = [self.new["tasks"][0], spec("other"), spec("more")]
        with self.assertRaises(ValueError):
            self.extend()

    def test_noop_and_shrink_rejected(self):
        for tasks in (self.old["tasks"], self.old["tasks"][:1]):
            with self.subTest(count=len(tasks)):
                self.new["tasks"] = copy.deepcopy(tasks)
                with self.assertRaises(ValueError):
                    self.extend()

    def test_top_level_execution_fields_immutable(self):
        for field, value in {"plan_id": "other", "report_path": "/other",
                             "version": 2, "env": {"A": "B"}}.items():
            with self.subTest(field=field):
                new = copy.deepcopy(self.new)
                new[field] = value
                with self.assertRaises(ValueError):
                    extend_state(self.previous, self.old, new)

    def test_mismatched_old_hash_and_missing_binding_rejected(self):
        for digest in ("bad", None, ""):
            self.previous["manifest_sha256"] = digest
            with self.assertRaises(ValueError):
                self.extend()

    def test_bad_manifest_identity_rejected(self):
        for ident in ("active", "", "../escape", None, 12):
            with self.subTest(ident=ident):
                self.new["tasks"][-1]["id"] = ident
                with self.assertRaises(ValueError):
                    self.extend()

    def test_embedded_new_runtime_and_preaccepted_status_rejected(self):
        for field, value in {"runtime": {"pid": 1}, "history": [], "accepted_at": "x",
                             "status": "accepted", "started_at": "x"}.items():
            with self.subTest(field=field):
                new = copy.deepcopy(self.new)
                new["tasks"][-1][field] = value
                with self.assertRaises(ValueError):
                    extend_state(self.previous, self.old, new)

    def test_old_state_missing_duplicate_reordered_or_changed_tasks_rejected(self):
        cases = [
            self.previous["tasks"][:1],
            [self.previous["tasks"][0], self.previous["tasks"][0]],
            list(reversed(self.previous["tasks"])),
        ]
        for tasks in cases:
            with self.subTest(tasks=tasks):
                previous = copy.deepcopy(self.previous)
                previous["tasks"] = copy.deepcopy(tasks)
                with self.assertRaises(ValueError):
                    extend_state(previous, self.old, self.new)
        self.previous["tasks"][1]["command"] = ["false"]
        with self.assertRaises(ValueError):
            self.extend()

    def test_malformed_old_state_evidence_rejected(self):
        changes = [
            ("tasks", "not a list"), ("completed", ["unknown"]),
            ("completed", ["accepted", "accepted"]), ("current", []),
            ("active", "unknown"), ("plan_id", "other"),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                previous = copy.deepcopy(self.previous)
                previous[key] = value
                with self.assertRaises(ValueError):
                    extend_state(previous, self.old, self.new)

    def test_malformed_task_runtime_status_history_rejected(self):
        for field, value in (("runtime", []), ("history", {}), ("history", [3]),
                             ("status", "invented")):
            with self.subTest(field=field):
                previous = copy.deepcopy(self.previous)
                previous["tasks"][1][field] = value
                with self.assertRaises(ValueError):
                    extend_state(previous, self.old, self.new)

    def test_nonfinite_and_malformed_context_rejected(self):
        self.new["tasks"][-1]["score"] = float("nan")
        with self.assertRaises(ValueError):
            self.extend()
        self.new["tasks"][-1].pop("score")
        with self.assertRaises(ValueError):
            extend_state(self.previous, self.old, self.new, caller_context=[])

    def test_ordinary_manifest_validation_still_required(self):
        runner.validate_manifest(self.old)
        runner.validate_manifest(self.new)
        amended, _ = self.extend()
        runner.initialize_state(amended, self.new)
        bad = copy.deepcopy(self.new)
        bad["tasks"][-1]["session"] = bad["tasks"][0]["session"]
        with self.assertRaises(ValueError):
            runner.validate_manifest(bad)

    def test_live_exact_cpu_child_is_observed_without_a_second_launch(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertEqual(os.environ.get("PYTHONNOUSERSITE"), "1")
        self.assertTrue(os.environ.get("STY"), "Runtime tests must be in screen")
        with tempfile.TemporaryDirectory(prefix="revision_cpu_", dir=PLAN / "reports") as folder:
            work = Path(folder)
            release = work / "release"
            # Child exits by itself after release or its deadline; no signal operations.
            child = subprocess.Popen([
                sys.executable, "-B", "-s", "-c",
                "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); "
                "end=time.monotonic()+8; "
                "exec('while not p.exists() and time.monotonic()<end: time.sleep(0.02)')",
                str(release),
            ], env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1"))
            try:
                record = process_identity(child.pid)
                self.assertTrue(process_alive(record))
                record["screen_id"] = os.environ["STY"]
                record_path = work / "process.json"
                record_path.write_text(json.dumps(record))
                self.previous["tasks"][1]["runtime"].update(
                    record=str(record_path), result=str(work / "absent.json"),
                    log=str(work / "empty.log"), tail_offset=0)
                amended, _ = self.extend()
                state = runner.initialize_state(amended, self.new)
                queue = runner.Queue(work / "manifest.json", self.new, state, Reporter())
                queue.events_path = work / "queue.events.jsonl"
                with patch.object(queue, "prepare_launch", side_effect=AssertionError("second launch")), \
                        patch.object(runner, "gpu_available", side_effect=AssertionError("GPU query")), \
                        patch.object(runner.subprocess, "run", side_effect=AssertionError("process operation")):
                    queue.observe(state["tasks"][1])
                self.assertEqual(state["tasks"][1]["status"], "running")
                self.assertEqual(state["current"]["pid"], child.pid)
                self.assertEqual(state["current"]["starttime"], record["starttime"])
                self.assertEqual(state["current"]["process_state"], "live PID and /proc starttime verified")
                self.assertEqual(state["tasks"][0], self.previous["tasks"][0])
                self.assertEqual(state["tasks"][2]["status"], "queued")
                self.assertEqual(state["completed"], ["accepted"])
                self.assertIsNone(child.poll())
            finally:
                release.touch()
                self.assertEqual(child.wait(timeout=10), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RevisionTests))
    evidence = {
        "tests_run": result.testsRun, "successful": result.wasSuccessful(),
        "failures": [(str(t), message) for t, message in result.failures],
        "errors": [(str(t), message) for t, message in result.errors],
        "screen": os.environ.get("STY"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "command": sys.argv,
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), PLAN / "pipeline/revision.py",
                      PLAN / "pipeline/runner.py", PLAN / "pipeline/task_entry.py")
        },
    }
    Path(args.summary).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
