"""Synthetic evidence/runner unit tests, not hardware performance measurements."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_results import artifact_path, report_evidence
from hardware.run_record import record_path, record_run, sha, verify_run_record
from verify_rtl import expected_commands, main as verify_main, read_measurements


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.binary = self.directory / "kernel"
        self.simulator = self.directory / "simulator"
        self.log = self.directory / "kernel.log"
        self.binary.write_bytes(b"native ELF fixture")
        self.simulator.write_bytes(b"RTL executable fixture")
        self.binary.with_suffix(".c").write_text("/* synthetic unit fixture */\n")
        self.provenance = {"workload": "two_gemm", "mode": "act", "dataset": 901,
                           "repetitions": 2, "calls": [],
                           "source_sha256": sha(self.binary.with_suffix(".c"))}
        self.binary.with_suffix(".json").write_text(json.dumps(self.provenance))
        build = {"binary_sha256": sha(self.binary), "source_sha256": self.provenance["source_sha256"]}
        Path(str(self.binary) + ".build.json").write_text(json.dumps(build))
        self.command = [str(self.simulator), f"+loadmem={self.binary}", str(self.binary)]
        self.stdout = (
            "ACT_PASS name=two_gemm mode=act dataset=901 iteration=0 cycles=1006 begin=100 end=1106 base=2164260928\n"
            "ACT_PASS name=two_gemm mode=act dataset=901 iteration=1 cycles=512 begin=2000 end=2512 base=2164260928\n"
        )
        self.trace = "\n".join(
            f"ACT_COMMAND opcode={op} cycle={i + 1} funct={func} rs1={rs1} rs2={rs2}"
            for i, (op, func, rs1, rs2) in enumerate(expected_commands([], 0x81000040, 2))
        ) + "\nACT_GEMMINI opcode=123 cycle=10 busy=0 mac00_nonzero=0\nACT_GEMMINI opcode=91 cycle=10 busy=0 mac00_nonzero=0\n"

    def run_fixture(self, returncode=0, during=None):
        def simulate(command, **kwargs):
            # The execution identity must already exist when the child starts.
            before = json.loads(record_path(self.log).read_text())
            self.assertEqual(before["status"], "started")
            self.assertEqual(before["rtl_sha256"], sha(self.simulator))
            self.assertEqual(before["binary_sha256"], sha(self.binary))
            self.assertEqual(before["command"], command)
            self.assertNotIn("stdout_sha256", before)
            if during:
                during()
            kwargs["stdout"].write(self.stdout)
            kwargs["stderr"].write(self.trace)
            return subprocess.CompletedProcess(command, returncode)

        with patch("hardware.run_record.subprocess.run", side_effect=simulate), redirect_stdout(io.StringIO()):
            return record_run(self.binary, self.log, self.command)

    def report(self):
        run = verify_run_record(self.log, self.binary)
        return {**{key: run[key] for key in ("rtl_sha256", "binary_sha256", "stdout_sha256", "trace_sha256")},
                "run_record_sha256": sha(record_path(self.log)), "cycles": [1006, 512],
                "workload": "two_gemm", "mode": "act", "dataset": "901", "provenance": self.provenance}

    def test_valid_record_precedes_execution_and_raw_cycles_match(self):
        self.assertEqual(self.run_fixture(), 0)
        cycles, trace = report_evidence(self.report(), self.log)
        self.assertEqual(cycles, [1006, 512])
        self.assertEqual(trace, self.trace)

    def test_report_only_cycle_change_is_rejected(self):
        self.run_fixture()
        report = self.report()
        report["cycles"] = [1, 1]
        with self.assertRaisesRegex(AssertionError, "cycles differ"):
            report_evidence(report, self.log)

    def test_changed_raw_stdout_is_rejected(self):
        self.run_fixture()
        report = self.report()
        self.log.write_text(self.stdout.replace("cycles=1006", "cycles=1"))
        with self.assertRaisesRegex(AssertionError, "Raw stdout changed"):
            report_evidence(report, self.log)

    def test_changed_raw_trace_is_rejected(self):
        self.run_fixture()
        report = self.report()
        Path(str(self.log) + ".trace").write_text("different trace")
        with self.assertRaisesRegex(AssertionError, "Raw trace changed"):
            report_evidence(report, self.log)

    def test_report_cannot_substitute_current_simulator_hash(self):
        self.run_fixture()
        report = self.report()
        self.simulator.write_bytes(b"new RTL installed after the run")
        report["rtl_sha256"] = sha(self.simulator)
        with self.assertRaisesRegex(AssertionError, "rtl_sha256 differs"):
            report_evidence(report, self.log)

    def test_verifier_keeps_original_rtl_identity_after_simulator_replacement(self):
        self.run_fixture()
        original_hash = sha(self.simulator)
        self.simulator.write_bytes(b"new RTL installed after the run")
        report_path = self.directory / "verified.rtl.json"
        argv = ["verify_rtl.py", "--binary", str(self.binary), "--log", str(self.log), "--report", str(report_path)]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            verify_main()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["rtl_sha256"], original_hash)
        self.assertNotEqual(report["rtl_sha256"], sha(self.simulator))
        self.assertEqual(report_evidence(report, self.log)[0], [1006, 512])

    def test_simulator_changed_during_execution_invalidates_run(self):
        with self.assertRaisesRegex(AssertionError, "changed during"):
            self.run_fixture(during=lambda: self.simulator.write_bytes(b"replacement"))
        with self.assertRaisesRegex(AssertionError, "invalidated"):
            verify_run_record(self.log)

    def test_native_binary_changed_during_execution_invalidates_run(self):
        with self.assertRaisesRegex(AssertionError, "changed during"):
            self.run_fixture(during=lambda: self.binary.write_bytes(b"replacement"))

    def test_native_binary_changed_after_execution_is_rejected(self):
        self.run_fixture()
        self.binary.write_bytes(b"replacement")
        with self.assertRaisesRegex(AssertionError, "Native binary differs"):
            verify_run_record(self.log, self.binary)

    def test_failed_process_cannot_be_verified(self):
        self.assertEqual(self.run_fixture(returncode=7), 7)
        with self.assertRaisesRegex(AssertionError, "did not finish"):
            verify_run_record(self.log)

    def test_interrupted_process_leaves_unverifiable_started_record(self):
        with patch("hardware.run_record.subprocess.run", side_effect=OSError("launch failed")):
            with self.assertRaisesRegex(OSError, "launch failed"):
                record_run(self.binary, self.log, self.command)
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            verify_run_record(self.log)

    def test_existing_run_is_never_overwritten(self):
        self.run_fixture()
        before = record_path(self.log).read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh run"):
            record_run(self.binary, self.log, self.command)
        self.assertEqual(record_path(self.log).read_bytes(), before)

    def test_missing_execution_record_is_not_backfilled(self):
        self.log.write_text(self.stdout)
        with self.assertRaisesRegex(AssertionError, "rerun RTL"):
            verify_run_record(self.log)
        self.assertFalse(record_path(self.log).exists())

    def test_manifest_replacement_is_rejected_by_report_binding(self):
        self.run_fixture()
        report = self.report()
        record = json.loads(record_path(self.log).read_text())
        record["command"].append("changed")
        record_path(self.log).write_text(json.dumps(record))
        with self.assertRaisesRegex(AssertionError, "not bound"):
            report_evidence(report, self.log)

    def test_wrong_elf_or_loadmem_argument_is_rejected_before_execution(self):
        for command in ([str(self.simulator), f"+loadmem={self.binary}", "wrong-elf"],
                        [str(self.simulator), "+loadmem=wrong-elf", str(self.binary)]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "ELF and"):
                record_run(self.binary, self.log, command)
        self.assertFalse(record_path(self.log).exists())

    def test_firmware_cycle_invariants(self):
        mutations = {
            "elapsed": self.stdout.replace("cycles=1006", "cycles=1"),
            "end": self.stdout.replace("end=1106", "end=100"),
            "zero": self.stdout.replace("cycles=1006 begin=100 end=1106", "cycles=0 begin=100 end=100"),
            "identity": self.stdout.replace("mode=act", "mode=sequential"),
            "iteration": self.stdout.replace("iteration=1", "iteration=0"),
            "missing": self.stdout.splitlines()[0],
            "extra": self.stdout + self.stdout.splitlines()[0],
            "malformed": self.stdout.replace("cycles=1006", "cycles=invalid"),
            "base": self.stdout.replace("2164260928", "2164260992"),
            "failure": self.stdout + "ACT_FAIL\n",
        }
        for label, stdout in mutations.items():
            with self.subTest(label=label), self.assertRaises(AssertionError):
                read_measurements(stdout, self.provenance)

    def test_report_labels_cannot_relabel_a_run(self):
        self.run_fixture()
        report = deepcopy(self.report())
        report["dataset"] = "zero"
        with self.assertRaisesRegex(AssertionError, "labels differ"):
            report_evidence(report, self.log)

    def test_relocation_without_mapping_keeps_the_recorded_path(self):
        self.assertEqual(artifact_path(self.log), self.log)

    def test_relocation_matches_path_components_not_text_prefixes(self):
        roots = (Path("/old/experiment"), Path("/new/experiment"))
        self.assertEqual(artifact_path("/old/experiment/build/run.json", roots),
                         Path("/new/experiment/build/run.json"))
        other = Path("/old/experiment-extra/build/run.json")
        self.assertEqual(artifact_path(other, roots), other)

    def test_relocation_rejects_relative_roots_and_parent_traversal(self):
        invalid = [("relative", "/new"), ("/old", "relative"),
                   ("/old/../other", "/new"), ("/old", "/new/../other")]
        for roots in invalid:
            with self.subTest(roots=roots), self.assertRaises(ValueError):
                artifact_path("/old/run.json", roots)
        with self.assertRaises(ValueError):
            artifact_path("/old/../outside/run.json", ("/old", "/new"))

    def test_relocated_evidence_keeps_hash_checks_without_the_old_path(self):
        self.run_fixture()
        report = self.report()
        original_record = record_path(self.log).read_bytes()
        moved = self.directory / "moved"
        moved.mkdir()
        for path in list(self.directory.iterdir()):
            if path.is_file():
                path.rename(moved / path.name)
        self.assertFalse(self.log.exists())
        relocated = artifact_path(self.log, (self.directory, moved))
        self.assertEqual(report_evidence(report, relocated)[0], [1006, 512])
        self.assertEqual(record_path(relocated).read_bytes(), original_record)
        with self.assertRaisesRegex(AssertionError, "Missing execution-time run record"):
            report_evidence(report, self.log)
        report["cycles"] = [1, 1]
        with self.assertRaisesRegex(AssertionError, "cycles differ"):
            report_evidence(report, relocated)


if __name__ == "__main__":
    unittest.main()
