"""Queue contracts with a fake clock/files/process runner: no server or launch."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import queue_round2 as queue
from round2_protocol import build_engineering_matrix, frozen_protocol


class StopFakeLoop(Exception):
    pass


class Clock:
    def __init__(self, max_sleeps=1000):
        self.now = 10000.0
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps += 1
        if self.sleeps >= self.max_sleeps:
            raise StopFakeLoop()


class QueueContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.plan = Path(self.temp.name)
        for name in ("tasks", "logs", "launches", "implementation"):
            (self.plan / name).mkdir()
        self.clock = Clock()
        self.launches, self.progress = [], []
        self.patches = [patch.object(queue, "PLAN", self.plan),
                        patch.object(queue, "atomic_json", side_effect=self.write_json),
                        patch.object(queue, "write_progress", side_effect=self.write_progress),
                        patch.object(queue, "time", self.clock),
                        patch.object(queue, "os", SimpleNamespace(name="posix", environ={"STY": "mock-owned"}, getpid=lambda: 12345)),
                        patch.object(queue, "project_lock", side_effect=lambda path: nullcontext()),
                        patch.object(queue.subprocess, "run", side_effect=AssertionError("No process may be launched by tests")),
                        patch.object(queue, "process_matches", return_value=False)]
        self.patches.append(patch.object(queue, "current_manager_identity", return_value={"pid":12345,"starttime":1}))
        for item in self.patches:
            item.start()
        rows = []
        for i, definition in enumerate(build_engineering_matrix()):
            task = dict(schema="odr003-round2-task-v1", stage="engineering", action="run",
                        run_id=f"synthetic_run_{i}", definition=definition,
                        implementation_sha256={}, protocol=frozen_protocol())
            task_path = self.plan / "tasks" / f"task_{i}.json"
            self.write_json(task_path, task)
            rows.append(dict(run_id=task["run_id"], definition=definition, task_path=str(task_path),
                             task_sha256=queue.sha256(task_path)))
        self.index = dict(schema="odr003-round2-engineering-index-v1", tasks=rows,
                          implementation_sha256={}, synthetic_contracts={"mock": True})
        self.index_path = self.plan / "engineering_index.json"
        self.write_json(self.index_path, self.index)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    @staticmethod
    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")

    def write_progress(self, state):
        snapshot = copy.deepcopy(state)
        self.progress.append(snapshot)
        self.write_json(self.plan / "status.json", snapshot)

    def state(self):
        return queue._new_state(self.index_path)

    def attempt(self, state, task_row, *, code=0, status="main_complete", point=None,
                missing_exit=False, missing_outcome=False, report=True):
        nonce = f"{len(state['attempts']) + 1:032x}"
        launch_dir = self.plan / "launches" / nonce
        launch_dir.mkdir()
        log = self.plan / "logs" / f"{nonce}.log"
        active = dict(run_id=task_row["run_id"], task_path=task_row["task_path"],
                      task_sha256=task_row["task_sha256"], nonce=nonce, session="mock-owned",
                      log_path=str(log), launch_directory=str(launch_dir), queued_at=self.clock.time(),
                      launch_state="intent_persisted")
        state["active"] = active
        state["attempts"].append(active)
        state["status"] = "running"
        outcome = dict(status=status, exit_code=code)
        if point is not None:
            outcome["point"] = point
        if status == "main_complete" and report:
            report_path = self.plan / f"report_{nonce}.json"
            self.write_json(report_path, {"run_id": task_row["run_id"], "mock": True})
            outcome["report"] = str(report_path)
            self.write_json(launch_dir / "launch.json", {"run_id": task_row["run_id"], "run_nonce": nonce})
        if not missing_exit:
            Path(str(log) + ".exitcode").write_text(str(code), encoding="utf8")
        if not missing_outcome:
            self.write_json(launch_dir / "outcome.json", outcome)
        return active

    def fake_launch(self, state, task_path, run_id):
        self.launches.append(run_id)
        row = dict(run_id=run_id, task_path=str(task_path), task_sha256=queue.sha256(task_path))
        status = "engineering_held" if run_id == "g50p2_engineering_verification" else "main_complete"
        self.attempt(state, row, status=status)
        self.write_progress(state)

    def allow_resources(self):
        return patch.object(queue, "assess_snapshot", return_value=SimpleNamespace(allow_start=True, reasons=()))

    def test_exactly_eight_plus_independent_verification_then_held_no_restart(self):
        with patch.object(queue, "collect_snapshot", return_value={}), self.allow_resources(), \
             patch.object(queue, "_launch", side_effect=self.fake_launch):
            self.assertEqual(queue.main(self.index_path), 0)
            expected = [row["run_id"] for row in self.index["tasks"]] + ["g50p2_engineering_verification"]
            self.assertEqual(self.launches, expected)
            state = queue.read(self.plan / "status.json")
            self.assertEqual(state["status"], "engineering_held")
            self.assertTrue(state["verified"])
            self.assertEqual(len(state["completed"]), 8)
            self.assertEqual(queue.main(self.index_path), 0)
            self.assertEqual(self.launches, expected)

    def test_completed_original_trajectories_are_not_relaunched(self):
        state = self.state()
        for row in self.index["tasks"][:3]:
            self.attempt(state, row)
            queue._finish_attempt(state)
        self.write_progress(state)
        with patch.object(queue, "collect_snapshot", return_value={}), self.allow_resources(), \
             patch.object(queue, "_launch", side_effect=self.fake_launch):
            self.assertEqual(queue.main(self.index_path), 0)
        self.assertEqual(self.launches, [row["run_id"] for row in self.index["tasks"][3:]]
                         + ["g50p2_engineering_verification"])

    def test_exit_75_waits_at_least_thirty_minutes(self):
        state = self.state()
        self.attempt(state, self.index["tasks"][0], code=75, status="waiting_resources")
        queue._finish_attempt(state)
        self.assertIsNone(state["active"])
        self.assertEqual(state["status"], "waiting_resources")
        self.assertGreaterEqual(state["next_resource_check"] - self.clock.time(), 1800)
        self.assertEqual(state["completed"], [])

    def test_busy_monitor_only_reads_resources_every_thirty_minutes(self):
        checks = []
        self.clock.max_sleeps = 361
        def snapshot(*args, **kwargs):
            checks.append(self.clock.time())
            return {}
        with patch.object(queue, "collect_snapshot", side_effect=snapshot), \
             patch.object(queue, "assess_snapshot", return_value=SimpleNamespace(allow_start=False, reasons=("foreign_compute",))), \
             patch.object(queue, "_launch", side_effect=AssertionError("Busy host cannot launch")):
            with self.assertRaises(StopFakeLoop):
                queue.main(self.index_path)
        self.assertEqual(checks, [10000.0, 11800.0])

    def test_only_two_distinct_declared_fault_points_may_resume(self):
        state = self.state()
        row = self.index["tasks"][3]
        for point in ("candidate_2_tentative", "observation_31"):
            self.attempt(state, row, code=76, status="controlled_interruption", point=point)
            # JSON readback breaks in-memory dict aliasing like a real restart.
            state = json.loads(json.dumps(state))
            queue._finish_attempt(state)
            self.assertEqual(state["status"], "resume_required")
            self.assertIsNone(state["active"])
        self.attempt(state, row, code=76, status="controlled_interruption", point="candidate_2_tentative")
        queue._finish_attempt(state)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["completed"], [])

    def test_unknown_fault_or_non_interrupted_trajectory_cannot_request_resume(self):
        for row, point in ((self.index["tasks"][0], "candidate_2_tentative"),
                           (self.index["tasks"][3], "new_unapproved_fault")):
            state = self.state()
            # Keep distinct directories while preserving the state schema.
            state["attempts"] = [{"nonce": "reserved", "run_id": "unrelated"}] if point.startswith("new") else []
            self.attempt(state, row, code=76, status="controlled_interruption", point=point)
            queue._finish_attempt(state)
            self.assertEqual(state["status"], "blocked")

    def test_nonzero_failure_and_disagreeing_outcome_block(self):
        state = self.state()
        self.attempt(state, self.index["tasks"][0], code=1, status="failed")
        queue._finish_attempt(state)
        self.assertEqual(state["status"], "blocked")
        self.assertIsNotNone(state["active"])
        state = self.state()
        state["attempts"] = [{"nonce": "reserved", "run_id": "unrelated"}]
        active = self.attempt(state, self.index["tasks"][1], code=75, status="waiting_resources")
        self.write_json(Path(active["launch_directory"]) / "outcome.json", {"exit_code": 0, "status": "main_complete"})
        queue._finish_attempt(state)
        self.assertEqual(state["status"], "blocked")

    def test_intent_only_restart_never_spawns_a_duplicate(self):
        state = self.state()
        active = self.attempt(state, self.index["tasks"][0], missing_exit=True,
                              missing_outcome=True, report=False)
        active["queued_at"] -= 121
        state["status"] = "launching"
        self.write_progress(state)
        with patch.object(queue, "collect_snapshot", side_effect=AssertionError("Active ambiguous launch must be resolved first")), \
             patch.object(queue, "_launch", side_effect=AssertionError("Cannot duplicate a persisted launch intent")):
            self.assertEqual(queue.main(self.index_path), 1)
        saved = queue.read(self.plan / "status.json")
        self.assertEqual(saved["status"], "blocked")
        self.assertEqual(saved["active"]["nonce"], active["nonce"])

    def test_wrapper_timeout_preserves_launch_intent_before_any_retry(self):
        state = self.state()
        task = self.index["tasks"][0]
        def timeout(*args, **kwargs):
            saved = queue.read(self.plan / "status.json")
            self.assertEqual(saved["active"]["launch_state"], "intent_persisted")
            raise subprocess.TimeoutExpired("synthetic-no-launch", 20)
        with patch.object(queue.subprocess, "run", side_effect=timeout):
            with self.assertRaises(subprocess.TimeoutExpired):
                queue._launch(state, Path(task["task_path"]), task["run_id"])
        saved = queue.read(self.plan / "status.json")
        self.assertEqual(len(saved["attempts"]), 1)
        self.assertIsNotNone(saved["active"])

    def test_index_cannot_expand_or_mix_complete_runs(self):
        expanded = copy.deepcopy(self.index)
        expanded["tasks"].append(copy.deepcopy(expanded["tasks"][0]))
        with self.assertRaises(ValueError):
            queue.validate_engineering_index(expanded)
        duplicate = copy.deepcopy(self.index)
        duplicate["tasks"][1]["run_id"] = duplicate["tasks"][0]["run_id"]
        with self.assertRaises(ValueError):
            queue.validate_engineering_index(duplicate)
        item = self.index["tasks"][0]
        task = queue.read(item["task_path"])
        task["stage"] = "complete"
        self.write_json(item["task_path"], task)
        item["task_sha256"] = queue.sha256(item["task_path"])
        with self.assertRaises(ValueError):
            queue.validate_engineering_index(self.index)

    def test_duplicate_completed_run_ids_fail_closed(self):
        state = self.state()
        self.attempt(state, self.index["tasks"][0])
        queue._finish_attempt(state)
        state["completed"].append(copy.deepcopy(state["completed"][0]))
        self.write_progress(state)
        with patch.object(queue, "_launch", side_effect=AssertionError("Invalid completed ledger cannot launch")):
            with self.assertRaises(ValueError):
                queue.main(self.index_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
