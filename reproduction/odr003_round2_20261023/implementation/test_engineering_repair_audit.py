"""Pure saved-record audit contracts; no model, image, SSH or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import verify_engineering_phase as phase


def zero_work():
    return dict(attempts_started=0, attempts_finished=0,
                inflight_unknown_attempt_ids=[], forwards_confirmed_completed={},
                forwards_execution_lower_bound={}, forwards_execution_upper_bound={},
                forwards_inflight_unknown_upper_bound={}, replay_forwards_confirmed_completed={},
                compute_seconds_confirmed=0.0, compute_time_incomplete=False)


def history_row():
    work = zero_work()
    work.update(attempts_started=1, attempts_finished=0,
                inflight_unknown_attempt_ids=["unfinished-image-read"], compute_time_incomplete=True)
    return dict(nonce="be01859de69743f181c337a9a17fcf01", run_id="reference_lightbox",
                actual_exit_code=1,
                actual_outcome=dict(status="failed", exit_code=1,
                                    error="Frozen inference preprocessing unexpectedly consumed RNG",
                                    execution_attempt_elapsed_seconds=2.0),
                physical_work=work,
                zero_forward_proof=dict(model_forward_upper_bound=0, unconfirmed_non_model_work=[dict(
                    attempt_id="unfinished-image-read", work_kind="image_read", forwards_reserved={})]),
                scientific_cursors=dict(observed=0, candidates=0, commits=0, rollbacks=0, next_index=0))


class RepairAuditContracts(unittest.TestCase):
    def setUp(self):
        self.binding = dict(path="/saved/preforward_import_proof.json", sha256="a" * 64)
        self.index = dict(tasks=[dict(run_id="reference_lightbox")], pre_forward_repair=self.binding)
        self.task = dict(pre_forward_repair=copy.deepcopy(self.binding))
        self.evidence = dict(status="passed", prior_failed_engineering_attempts=[history_row()])

    def audit(self, evidence=None):
        verifier = Mock(return_value=self.evidence if evidence is None else evidence)
        result = phase._repair_audit(self.index, self.task, verifier=verifier)
        verifier.assert_called_once_with(self.binding, expected_index=self.index, plan=phase.PLAN)
        return result

    def test_ordinary_index_has_no_repair_exception(self):
        verifier = Mock(side_effect=AssertionError("must not verify absent exception"))
        self.assertEqual(phase._repair_audit(dict(tasks=[]), {}, verifier=verifier), (None, []))
        verifier.assert_not_called()

    def test_verified_exit_one_remains_failure_and_no_new_scientific_run(self):
        evidence, rows = self.audit()
        self.assertEqual(rows[0]["actual_exit_code"], 1)
        self.assertEqual(rows[0]["actual_outcome"]["status"], "failed")
        self.assertEqual(evidence, self.evidence)
        rows[0]["actual_outcome"]["status"] = "tampered"
        self.assertEqual(self.evidence["prior_failed_engineering_attempts"][0]["actual_outcome"]["status"], "failed")

    def test_different_task_binding_fails_before_helper(self):
        self.task["pre_forward_repair"]["sha256"] = "b" * 64
        verifier = Mock()
        with self.assertRaisesRegex(ValueError, "binding differs"):
            phase._repair_audit(self.index, self.task, verifier=verifier)
        verifier.assert_not_called()

    def test_failed_independent_proof_blocks(self):
        self.evidence["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "independently pass"):
            self.audit()

    def test_more_than_one_implementation_failure_is_not_excused(self):
        self.evidence["prior_failed_engineering_attempts"].append(history_row())
        with self.assertRaisesRegex(ValueError, "exactly its one"):
            self.audit()

    def test_exit_one_cannot_be_relabelled_wait_or_interruption(self):
        for replacement in (0, 75, 76):
            doc = copy.deepcopy(self.evidence)
            doc["prior_failed_engineering_attempts"][0]["actual_exit_code"] = replacement
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "genuine exit 1"):
                self.audit(doc)

    def test_old_outcome_exit_is_also_checked(self):
        self.evidence["prior_failed_engineering_attempts"][0]["actual_outcome"]["exit_code"] = 75
        with self.assertRaisesRegex(ValueError, "genuine exit 1"):
            self.audit()

    def test_dispatched_or_unknown_forward_cannot_use_zero_forward_exception(self):
        for key, value in (("forwards_execution_upper_bound", {"arrival_scoring": 1}),):
            doc = copy.deepcopy(self.evidence)
            doc["prior_failed_engineering_attempts"][0]["physical_work"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "model forward"):
                self.audit(doc)

    def test_unconfirmed_image_read_is_retained_as_unknown_cost(self):
        _, rows = self.audit()
        self.assertEqual(rows[0]["physical_work"]["inflight_unknown_attempt_ids"], ["unfinished-image-read"])
        self.assertTrue(rows[0]["physical_work"]["compute_time_incomplete"])

    def test_unconfirmed_work_cannot_be_reclassified_from_model_work_or_erased(self):
        for key, value in (("attempts_started", 0), ("inflight_unknown_attempt_ids", []),
                           ("compute_time_incomplete", False)):
            doc = copy.deepcopy(self.evidence)
            doc["prior_failed_engineering_attempts"][0]["physical_work"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unconfirmed image-read"):
                self.audit(doc)
        for key, value in (("work_kind", "arrival_scoring"), ("forwards_reserved", {"model": 1})):
            doc = copy.deepcopy(self.evidence)
            doc["prior_failed_engineering_attempts"][0]["zero_forward_proof"]["unconfirmed_non_model_work"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unconfirmed image-read"):
                self.audit(doc)

    def test_prior_scientific_commit_or_observation_blocks(self):
        for key in ("observed", "candidates", "commits", "rollbacks", "next_index"):
            doc = copy.deepcopy(self.evidence)
            doc["prior_failed_engineering_attempts"][0]["scientific_cursors"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "scientific progress"):
                self.audit(doc)

    def test_prior_failure_must_belong_to_the_first_real_trajectory(self):
        self.evidence["prior_failed_engineering_attempts"][0]["run_id"] = "other_run"
        with self.assertRaisesRegex(ValueError, "first trajectory"):
            self.audit()

    def test_costs_preserve_failure_elapsed_and_unknown_memory_without_counting_ninth_run(self):
        current = dict(run=[dict(nonce="c" * 32, actual_exit_code=0,
                                actual_outcome=dict(exit_code=0, execution_attempt_elapsed_seconds=10.0,
                                                    peak_gpu_allocated_bytes=100, peak_gpu_reserved_bytes=200))])
        ledger = zero_work()
        ledger["forwards_confirmed_completed"] = {"arrival_scoring": 128}
        ledger["forwards_execution_lower_bound"] = {"arrival_scoring": 128}
        ledger["forwards_execution_upper_bound"] = {"arrival_scoring": 128}
        ledger["compute_seconds_confirmed"] = 8.0
        cost = phase._phase_costs(current, [history_row()], {"run": ledger})
        self.assertEqual(cost["actual_attempt_count"], 2)
        self.assertEqual(cost["actual_exit_codes"], {"0": 1, "1": 1})
        self.assertEqual(cost["scientific_trajectories"], 8)
        self.assertEqual(cost["logical_observations"], 1024)
        self.assertFalse(cost["prior_failure_is_additional_scientific_trajectory"])
        self.assertEqual(cost["forward_costs"]["forwards_confirmed_completed"], {"arrival_scoring": 128})
        self.assertEqual(cost["compute_seconds_confirmed"], 8.0)
        self.assertTrue(cost["compute_time_incomplete"])
        actual = cost["attempt_measurements"]
        self.assertEqual(actual["execution_attempt_elapsed_seconds_sum_recorded"], 12.0)
        self.assertEqual(actual["peak_gpu_allocated_bytes_unknown_attempts"], [history_row()["nonce"]])
        self.assertFalse(actual["memory_maximum_complete_across_attempts"])

    def test_duplicate_attempt_in_cost_history_is_not_double_counted(self):
        row = history_row()
        with self.assertRaisesRegex(ValueError, "duplicate attempt"):
            phase._phase_costs({"current": [row]}, [row], {})

    def test_an_unbound_current_exit_one_still_blocks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(phase, "PLAN", Path(directory)):
            plan = Path(directory)
            nonce = "a" * 32
            launch = plan / "launches" / nonce
            launch.mkdir(parents=True)
            log = plan / "failed.log"
            log.write_text("saved failure", encoding="utf8")
            Path(str(log) + ".exitcode").write_text("1", encoding="ascii")
            outcome = dict(status="failed", exit_code=1)
            (launch / "outcome.json").write_text(json.dumps(outcome), encoding="utf8")
            row = dict(nonce=nonce, run_id="run", task_path="task.json", task_sha256="f" * 64,
                       log_path=str(log), launch_directory=str(launch), outcome=outcome, exit_code=1)
            receipts = dict(all_attempts=[row], queue_wait_seconds=0)
            expected = {"run": dict(task_path="task.json", task_sha256="f" * 64)}
            with self.assertRaisesRegex(ValueError, "unexpected failed attempt"):
                phase._attempt_evidence(receipts, expected)


if __name__ == "__main__":
    unittest.main()
