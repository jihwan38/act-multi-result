"""A changed source plan must never relabel an already generated compiler."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import prepare_sources
from _internal import run
from _internal.common import fingerprint, source_hashes


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mode = "multipattern"
        self.base = self.root / "build" / self.mode
        for module in (prepare_sources, run):
            mocked = patch.object(module, "ROOT", self.root)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(prepare_sources, "COMPONENTS", ("act-backend",))
        mocked.start()
        self.addCleanup(mocked.stop)

    def plan(self, revision="old-revision", patches=()):
        return {"toolchain": self.mode, "upstreams": {"act-backend": revision},
                "patches": list(patches)}

    def write(self, relative, content):
        path = self.base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def prepared(self, plan):
        self.write("src/act-backend/module.py", "original source\n")
        record = {**plan, "fingerprint": fingerprint(plan),
                  "files": source_hashes(self.base / "src")}
        self.write("src/provenance.json", json.dumps(record))
        return record

    def generated(self):
        binary = self.write("backends/QKV", "original compiler\n")
        self.write("backends/QKVDoubleGemm", "other generated compiler\n")
        target = self.base / "targets/QKV"
        self.write("targets/QKV/oracle/build/print_handler_ext.test.so", "Oracle runtime\n")
        self.write("targets/QKV/backend/src/discovery.rs", "generated source\n")
        self.write("runs/qkv/attention/assembly.py", "previous assembly\n")
        self.write("cache/keep", "cached dependency\n")
        return target, binary

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def archive(self, command, **kwargs):
        """Exercise real tar extraction without git or any repository writes."""
        self.assertEqual(command[:2], ["git", "-C"])
        self.assertIn("archive", command)
        output = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
        payload = ("source from " + command[-1] + "\n").encode()
        with tarfile.open(output, "w") as bundle:
            member = tarfile.TarInfo("module.py")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        return subprocess.CompletedProcess(command, 0)

    def test_changed_plan_invalidates_all_compilers_but_keeps_targets_runs_and_cache(self):
        self.prepared(self.plan())
        self.generated()
        retained = {name: self.snapshot(self.base / name) for name in ("targets", "runs", "cache")}
        other_mode = self.root / "build/multi-result/backends/QKV"
        other_mode.parent.mkdir(parents=True)
        other_mode.write_text("different toolchain compiler\n")
        current = self.plan("new-revision")
        with patch.object(prepare_sources, "source_plan", return_value=current), \
                patch.object(prepare_sources.subprocess, "run", side_effect=self.archive):
            record = prepare_sources.prepare(self.mode)
        self.assertFalse((self.base / "backends").exists())
        self.assertEqual((self.base / "src/act-backend/module.py").read_text(),
                         "source from new-revision\n")
        self.assertEqual(record["fingerprint"], fingerprint(current))
        self.assertEqual(json.loads((self.base / "src/provenance.json").read_text()), record)
        for name, contents in retained.items():
            self.assertEqual(self.snapshot(self.base / name), contents)
        self.assertEqual(other_mode.read_text(), "different toolchain compiler\n")

    def test_unchanged_plan_preserves_compilers_without_staging(self):
        plan = self.plan()
        record = self.prepared(plan)
        self.generated()
        before = self.snapshot(self.base)
        with patch.object(prepare_sources, "source_plan", return_value=plan), \
                patch.object(prepare_sources.subprocess, "run") as command:
            self.assertEqual(prepare_sources.prepare(self.mode), record)
        command.assert_not_called()
        self.assertEqual(self.snapshot(self.base), before)

    def test_failed_patch_staging_preserves_existing_sources_and_compilers(self):
        self.prepared(self.plan())
        self.generated()
        before = self.snapshot(self.base)
        current = self.plan("new-revision", [{
            "path": "patches/multipattern/act-backend/0001-broken.patch", "sha256": "changed"}])

        def fail_patch(command, **kwargs):
            if "archive" in command:
                return self.archive(command, **kwargs)
            self.assertIn("apply", command)
            raise subprocess.CalledProcessError(1, command)

        with patch.object(prepare_sources, "source_plan", return_value=current), \
                patch.object(prepare_sources.subprocess, "run", side_effect=fail_patch):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare_sources.prepare(self.mode)
        self.assertEqual(self.snapshot(self.base), before)
        self.assertFalse(list(self.base.glob(".prepare-*")))

    def test_forced_reprepare_invalidates_compilers_even_when_plan_is_unchanged(self):
        plan = self.plan()
        self.prepared(plan)
        self.generated()
        with patch.object(prepare_sources, "source_plan", return_value=plan), \
                patch.object(prepare_sources.subprocess, "run", side_effect=self.archive):
            prepare_sources.prepare(self.mode, force=True)
        self.assertFalse((self.base / "backends").exists())
        self.assertTrue((self.base / "runs/qkv/attention/assembly.py").is_file())

    def test_skip_build_rejects_stale_plan_before_accepting_existing_compiler(self):
        self.prepared(self.plan())
        self.generated()
        before = self.snapshot(self.base)
        with patch.object(run, "source_plan", return_value=self.plan("new-revision")), \
                patch.object(run, "built_target", wraps=run.built_target) as target:
            with self.assertRaisesRegex(ValueError, "Source configuration changed;.*--skip-build.*regenerate"):
                run.existing_build({"target": "QKV"}, self.mode)
        target.assert_not_called()
        self.assertEqual(self.snapshot(self.base), before)

    def test_skip_build_accepts_current_plan_and_returns_its_provenance(self):
        plan = self.plan()
        record = self.prepared(plan)
        target, binary = self.generated()
        before = self.snapshot(self.base)
        with patch.object(run, "source_plan", return_value=plan):
            self.assertEqual(run.existing_build({"target": "QKV"}, self.mode),
                             (target, binary, record))
        self.assertEqual(self.snapshot(self.base), before)


if __name__ == "__main__":
    unittest.main()
