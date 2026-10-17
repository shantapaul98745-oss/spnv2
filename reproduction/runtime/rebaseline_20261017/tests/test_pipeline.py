"""CPU-only pipeline acceptance. Execute this file through the project screen wrapper."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REAL_PLAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_PLAN / "pipeline"))
import progress
import runner
import task_entry


class PipelineTests(unittest.TestCase):
    def setUp(self):
        evidence = REAL_PLAN / "reports/pipeline_tests"
        evidence.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="pipeline_test_", dir=str(evidence)))
        self.plan_patch = patch.object(runner, "PLAN", self.root)
        self.plan_patch.start()
        self.manifest_path = self.root / "queue.json"

    def tearDown(self):
        self.plan_patch.stop()
        # Keep exact process, logs, commands and output evidence from nested screen tests.
        pass

    def manifest(self, tasks):
        return {"version": 1, "plan_id": self.root.name, "tasks": tasks}

    def task(self, ident="first", code="print('okay')", **extra):
        task = {
            "id": ident, "kind": "cpu_test",
            "command": [runner.PYTHON, "-B", "-s", "-c", code],
            "cwd": str(runner.BASE / "spnv2"),
            "session": self.root.name + "_" + ident,
            "log": str(self.root / (ident + ".log")),
            "requires": [], "verify_command": [runner.PYTHON, "-B", "-s", "-c", "print('verified')"],
            "gpu": False,
        }
        task.update(extra)
        return task

    def queue(self, tasks):
        manifest = self.manifest(tasks)
        runner.validate_manifest(manifest)
        progress.atomic_json(self.manifest_path, manifest)
        state = runner.initialize_state({"tasks": [], "notes": []}, manifest)
        reporter = progress.Reporter(self.root / "status.json", self.root / "progress.md", state)
        queue = runner.Queue(self.manifest_path, manifest, state, reporter)
        return queue, reporter, state

    def drive(self, queue, state, timeout=30):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            outcome = queue.step()
            if outcome in {"complete", "failed"}:
                return outcome
            time.sleep(0.15)
        self.fail("CPU queue did not finish: " + json.dumps(state, default=str))

    def test_atomic_replace_failure_preserves_prior_file(self):
        path = self.root / "atomic.json"
        progress.atomic_json(path, {"version": 1})
        with patch("progress.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                progress.atomic_json(path, {"version": 2})
        self.assertEqual(progress.read_json(path), {"version": 1})
        self.assertEqual(list(self.root.glob(".atomic.json.*")), [])

    def test_report_failure_blocks_advancement_without_killing(self):
        queue, reporter, state = self.queue([self.task()])
        with patch("progress.atomic_write", side_effect=OSError("read only")), \
                patch.object(queue, "prepare_launch") as launch:
            self.assertFalse(reporter.publish(force=True))
            self.assertEqual(queue.step(), "report_blocked")
            launch.assert_not_called()
        self.assertTrue(reporter.publish(force=True))
        self.assertTrue(reporter.warning_path.exists())

    def test_report_publication_keeps_task_references_and_revision(self):
        queue, reporter, state = self.queue([self.task()])
        task = state["tasks"][0]
        self.assertTrue(reporter.publish(force=True))
        self.assertIs(task, state["tasks"][0])
        task["status"] = "running"
        reporter.publish(force=True)
        snapshot = progress.read_json(self.root / "status.json")
        text = (self.root / "progress.md").read_text()
        self.assertEqual(snapshot["tasks"][0]["status"], "running")
        self.assertIn(str(snapshot["revision"]), text)

    def test_render_full_state_stale_and_paths(self):
        state = {
            "heartbeat_at": "2026-01-01T00:00:00+08:00",
            "updated_at": "2026-01-01T00:00:00+08:00",
            "tasks": [{"id": value, "status": value} for value in
                      ["queued", "running", "awaiting_verification", "accepted", "failed", "limited"]],
            "current": {"last_event_at": "2026-01-01T00:00:00+08:00",
                        "epoch": 2, "epochs": 20, "batch": 4, "batches": 5,
                        "lr": 0.001, "loss": 1.5, "update_attempts": 8,
                        "optimizer_updates": 7, "amp_skips": 1, "log": "/log/path"},
        }
        rendered = progress.render(state, self.root, self.root / "progress.md")
        for value in ["2026-01-01T00:00:00+08:00", "2 / 20", "4 / 5", "0.001", "1.5",
                      "超过180秒", "/log/path", "awaiting verification", "accepted", "limited"]:
            self.assertIn(value, rendered)

    def test_manifest_hash_refuses_changed_work(self):
        manifest = self.manifest([self.task()])
        state = runner.initialize_state({"tasks": []}, manifest)
        state["tasks"][0]["status"] = "accepted"
        accepted = runner.initialize_state(state, manifest)
        self.assertEqual(accepted["completed"], ["first"])
        modified = copy.deepcopy(manifest)
        modified["tasks"][0]["command"].append("different")
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            runner.initialize_state(state, modified)

    def test_manifest_rejects_preaccepted_and_missing_evidence(self):
        for task in [self.task(status="accepted"), self.task(verify_command=None)]:
            with self.assertRaises(ValueError):
                runner.validate_manifest(self.manifest([task]))

    def test_actual_process_starttime_and_stale_pid(self):
        identity = task_entry.process_identity(os.getpid())
        self.assertTrue(task_entry.process_alive(identity))
        self.assertFalse(task_entry.process_alive(dict(identity, starttime="wrong")))
        self.assertFalse(task_entry.process_alive({"pid": 999999999, "starttime": "1"}))

    def test_gpu_gate_blocks_compute_and_allows_desktop(self):
        memory = subprocess.CompletedProcess([], 0, stdout="200, 24000\n")
        occupied = subprocess.CompletedProcess([], 0, stdout="123, /usr/bin/python, 100\n")
        desktop = subprocess.CompletedProcess([], 0, stdout="123, /usr/lib/xorg/Xorg, 100\n")
        with patch("runner.subprocess.run", side_effect=[memory, occupied]):
            self.assertFalse(runner.gpu_available({"min_free_mb": 1000})[0])
        with patch("runner.subprocess.run", side_effect=[memory, desktop]):
            self.assertTrue(runner.gpu_available({"min_free_mb": 1000})[0])
        with patch("runner.subprocess.run", side_effect=[memory, desktop]):
            self.assertFalse(runner.gpu_available({"max_used_mb": 100})[0])

    def test_events_are_incremental_and_error_is_retained(self):
        queue, reporter, state = self.queue([self.task()])
        task = state["tasks"][0]
        task["runtime"] = {"log": str(self.root / "events.log"), "tail_offset": 0}
        event = {"event": "checkpoint", "epoch": 1, "loss": 2.5, "optimizer_updates": 2, "amp_skips": 0}
        Path(task["runtime"]["log"]).write_text("SPNV2_EVENT " + json.dumps(event) + "\n")
        queue.tail_events(task)
        self.assertEqual(state["current"]["optimizer_updates"], 2)
        revision = state["revision"]
        queue.tail_events(task)
        self.assertEqual(state["revision"], revision)
        with Path(task["runtime"]["log"]).open("a") as stream:
            stream.write('SPNV2_EVENT {"event":"error","message":"NaN"}\n')
        queue.tail_events(task)
        self.assertEqual(task["event_error"]["event"], "error")

    def test_restart_launch_intent_never_relaunches(self):
        queue, reporter, state = self.queue([self.task()])
        task = state["tasks"][0]
        task["status"] = "launching"
        task["runtime"] = {"phase": "main", "record": str(self.root / "missing.process"),
                           "result": str(self.root / "missing.result"), "log": str(self.root / "missing.log"),
                           "missing_since": "2026-01-01T00:00:00+08:00"}
        with patch.object(queue, "prepare_launch") as launch:
            queue.step()
            self.assertEqual(task["status"], "failed")
            queue.step()
            launch.assert_not_called()
        self.assertIn("no automatic relaunch", task["error"])

    def test_live_exact_pid_is_reconciled_on_restart(self):
        queue, reporter, state = self.queue([self.task()])
        task = state["tasks"][0]
        identity = task_entry.process_identity(os.getpid())
        record_path = self.root / "existing.process"
        progress.atomic_json(record_path, identity)
        task["status"] = "launching"
        task["runtime"] = {"phase": "main", "record": str(record_path),
                           "result": str(self.root / "missing.result"), "log": str(self.root / "missing.log")}
        with patch.object(queue, "prepare_launch") as launch:
            queue.step()
            launch.assert_not_called()
        self.assertEqual(task["status"], "running")
        self.assertEqual(state["current"]["pid"], os.getpid())

    def test_changed_manifest_forbids_new_launch(self):
        queue, reporter, state = self.queue([self.task()])
        modified = self.manifest([self.task(code="print('changed')")])
        progress.atomic_json(self.manifest_path, modified)
        with patch("runner.subprocess.run") as command:
            queue.prepare_launch(state["tasks"][0], "main")
            command.assert_not_called()
        self.assertEqual(state["tasks"][0]["status"], "failed")

    def test_expected_outputs_require_nonempty_file_and_matching_hash(self):
        path = self.root / "output"
        task = self.task(expected_outputs=[str(path)])
        self.assertIn("not a file", runner.expected_outputs_error(task))
        path.write_bytes(b"")
        self.assertIn("too small", runner.expected_outputs_error(task))
        path.write_bytes(b"evidence")
        task["expected_outputs"] = [{"path": str(path), "sha256": hashlib.sha256(b"evidence").hexdigest()}]
        self.assertIsNone(runner.expected_outputs_error(task))

    def test_screen_main_verify_and_accept(self):
        output = self.root / "artifact.json"
        code = ("from pathlib import Path; import json; "
                "print('SPNV2_EVENT '+json.dumps({'event':'checkpoint','epoch':1,'epochs':1,"
                "'update_attempts':1,'optimizer_updates':1,'amp_skips':0}),flush=True); "
                "Path(" + repr(str(output)) + ").write_text('accepted evidence')")
        task = self.task(code=code, expected_outputs=[str(output)])
        queue, reporter, state = self.queue([task])
        self.assertEqual(self.drive(queue, state), "complete")
        actual = state["tasks"][0]
        self.assertEqual(actual["status"], "accepted")
        self.assertEqual(len(actual["history"]), 1)
        main_record = progress.read_json(actual["history"][0]["record"])
        self.assertTrue(main_record["starttime"])
        self.assertTrue(main_record["screen_id"])
        self.assertEqual(main_record["command"], task["command"])
        self.assertEqual(progress.read_json(actual["runtime"]["result"])["exit_code"], 0)

    def test_screen_failure_stops_later_tasks_no_retry(self):
        queue, reporter, state = self.queue([
            self.task(code="raise SystemExit(7)"),
            self.task("second", requires=["first"]),
        ])
        self.assertEqual(self.drive(queue, state), "failed")
        failed = state["tasks"][0]
        self.assertEqual(failed["status"], "failed")
        for _ in range(3):
            self.assertEqual(queue.step(), "failed")
        self.assertNotIn("runtime", state["tasks"][1])
        self.assertNotIn("history", failed)
        self.assertEqual(failed["runtime"]["exit_code"], 7)

    def test_screen_zero_exit_missing_file_is_not_accepted(self):
        task = self.task(expected_outputs=[str(self.root / "never_created")])
        queue, reporter, state = self.queue([task])
        self.assertEqual(self.drive(queue, state), "failed")
        self.assertIn("Expected output", state["tasks"][0]["error"])

    def test_progress_sink_warning_latches_without_killing_current_task(self):
        warning = json.dumps({"event": "progress_write_warning", "phase": "train",
                              "error": "disk full", "path": "/optional/sink", "queue_blocked": True})
        code = ("import time; print('SPNV2_EVENT ' + " + repr(warning)
                + ",flush=True); time.sleep(0.3); print('HEALTHY_COMMAND_FINISHED',flush=True)")
        queue, reporter, state = self.queue([self.task(code=code), self.task("second", requires=["first"])])
        until = time.monotonic() + 20
        while time.monotonic() < until and state.get("status") != "reporting_blocked":
            queue.step()
            time.sleep(0.1)
        self.assertEqual(state["status"], "reporting_blocked")
        self.assertTrue(state["downstream_blocked"])
        task = state["tasks"][0]
        self.assertEqual(task["status"], "awaiting_verification")
        self.assertEqual(task["runtime"]["phase"], "main")
        self.assertEqual(progress.read_json(task["runtime"]["result"])["exit_code"], 0)
        self.assertIn("HEALTHY_COMMAND_FINISHED", Path(task["runtime"]["log"]).read_text())
        self.assertNotIn("event_error", task)
        with patch.object(queue, "prepare_launch") as launch:
            for _ in range(3):
                queue.step()
            launch.assert_not_called()
        self.assertNotIn("runtime", state["tasks"][1])


    def test_phase_complete_keeps_full_goal_backlog(self):
        queue,reporter,state=self.queue([])
        state["backlog"]=["remaining source/ODR/scans"]
        self.assertEqual(queue.step(),"complete")
        self.assertEqual(state["status"],"phase_complete")
        self.assertEqual(len(state["backlog"]),1)

    def test_stopped_manager_never_launches(self):
        queue,reporter,state=self.queue([self.task()])
        queue.stop_requested=True
        with patch("runner.subprocess.run") as launch:
            self.assertEqual(queue.step(),"stopped")
            self.assertFalse(queue.prepare_launch(state["tasks"][0],"main"))
            self.assertFalse(queue.prepare_launch(state["tasks"][0],"verify"))
            launch.assert_not_called()

    def test_signal_during_resource_gate_never_launches(self):
        queue,reporter,state=self.queue([self.task(gpu={"index":0})])
        def gate(spec):queue.stop_requested=True;return True,None
        with patch("runner.gpu_available",side_effect=gate),patch("runner.subprocess.run") as launch:
            queue.step();launch.assert_not_called()
        self.assertNotIn("runtime",state["tasks"][0])

    def test_signal_during_publish_never_spawns(self):
        queue,reporter,state=self.queue([self.task()])
        publish=reporter.publish
        def stop_after_publish(*a,**kw):
            result=publish(*a,**kw);queue.stop_requested=True;return result
        with patch.object(reporter,"publish",side_effect=stop_after_publish),patch("runner.subprocess.run") as launch:
            self.assertFalse(queue.prepare_launch(state["tasks"][0],"main"));launch.assert_not_called()
        self.assertTrue(state["tasks"][0]["runtime"]["launch_cancelled_before_spawn"])

    def test_remote_desktop_is_bounded_baseline(self):
        memory=subprocess.CompletedProcess([],0,"350, 15717\n","")
        desktop=subprocess.CompletedProcess([],0,"7890, /usr/libexec/gnome-remote-desktop-daemon, 120\n","")
        large=subprocess.CompletedProcess([],0,"7890, /usr/libexec/gnome-remote-desktop-daemon, 999\n","")
        with patch("runner.subprocess.run",side_effect=[memory,desktop]):
            self.assertTrue(runner.gpu_available({"min_free_mb":14500,"max_used_mb":1200})[0])
        with patch("runner.subprocess.run",side_effect=[memory,large]):
            self.assertFalse(runner.gpu_available({"min_free_mb":14500,"max_used_mb":1200})[0])

    def test_checkpoint_and_missing_score_display(self):
        queue,reporter,state=self.queue([self.task()]);task=state["tasks"][0]
        log=self.root/"mapping.log";log.write_text('SPNV2_EVENT '+json.dumps({"event":"checkpoint","path":"/new/checkpoint"})+'\n'+'SPNV2_EVENT '+json.dumps({"event":"validation_end","n_samples":3,"heatmap_rejections":3,"score":None,"metrics":{"heat_pose":{"mean":None,"count":0,"unit":"score"}}})+'\n')
        task["runtime"]={"log":str(log),"tail_offset":0}
        queue.tail_events(task)
        self.assertEqual(state["current"]["checkpoint"],"/new/checkpoint")
        self.assertIsNone(state["last_evaluation"]["score"])
        text=progress.render(state,self.root,self.root/"progress.md")
        self.assertIn("heat_pose",text);self.assertIn("有效分母",text);self.assertIn("/new/checkpoint",text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
