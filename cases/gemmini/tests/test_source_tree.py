"""Check source ownership and pinning with tiny, local real Git repositories."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.source_tree import check_sources, prepare_sources


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], stderr=subprocess.PIPE, text=True
    ).strip()


def repository(path, submodules=()):
    path.mkdir(parents=True)
    git(path, "init", "--quiet")
    git(path, "config", "user.name", "Source Test")
    git(path, "config", "user.email", "source-test@example.invalid")
    (path / "README").write_text("upstream\n")
    if submodules:
        (path / ".gitmodules").write_text("".join(
            f'[submodule "{name}"]\n\tpath = {name}\n\turl = {source}\n'
            for name, source in submodules
        ))
    git(path, "add", ".")
    for name, source in submodules:
        revision = git(source, "rev-parse", "HEAD")
        git(path, "update-index", "--add", "--cacheinfo", f"160000,{revision},{name}")
    git(path, "commit", "--quiet", "-m", "fixture")
    return path


class SourceTreeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="act05-source-test-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        # Explicitly allow local fixture URLs in this test process only.
        environment = patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"})
        environment.start()
        self.addCleanup(environment.stop)
        tests = repository(directory / "upstreams/tests")
        gemmini = repository(directory / "upstreams/gemmini", [
            ("software/gemmini-rocc-tests", tests),
        ])
        rocket = repository(directory / "upstreams/rocket")
        chipyard = repository(directory / "upstreams/chipyard", [
            ("generators/gemmini", gemmini), ("generators/rocket-chip", rocket),
        ])
        self.repo = repository(directory / "project", [("third_party/chipyard", chipyard)])
        self.experiment = self.repo / "cases/gemmini"
        self.experiment.mkdir(parents=True)
        self.upstream = self.repo / "third_party/chipyard"
        self.working = self.repo / "build/multi-result-legacy/hardware/chipyard"
        self.pins = {"chipyard": git(chipyard, "rev-parse", "HEAD"),
                     "gemmini_revision": git(gemmini, "rev-parse", "HEAD"),
                     "rocket_chip_revision": git(rocket, "rev-parse", "HEAD"),
                     "gemmini_tests_revision": git(tests, "rev-parse", "HEAD")}

    def prepare_nested(self):
        prepare_sources(self.experiment)
        git(self.working, "submodule", "update", "--init", "--recursive")

    def test_preparation_is_opt_in_and_original_stays_clean(self):
        self.assertFalse((self.upstream / ".git").exists())
        prepare_sources(self.experiment)
        self.assertEqual(git(self.upstream, "status", "--porcelain"), "")
        # Nested hardware is not fetched by the generic source-copy helper.
        self.assertFalse((self.working / "generators/gemmini/.git").exists())
        self.assertFalse((self.working / ".git/objects/info/alternates").exists())
        self.assertEqual(git(self.working, "rev-parse", "HEAD"), self.pins["chipyard"])
        self.assertEqual(git(self.working, "remote", "get-url", "origin"),
                         git(self.upstream, "remote", "get-url", "origin"))

    def test_nested_pins_come_from_gitlinks(self):
        self.prepare_nested()
        actual = check_sources(self.experiment)
        self.assertEqual(actual.pop("chipyard")["revision"], self.pins["chipyard"])
        self.assertEqual(actual, {k: v for k, v in self.pins.items() if k != "chipyard"})

    def test_existing_working_changes_are_preserved(self):
        self.prepare_nested()
        (self.working / "README").write_text("experiment patch\n")
        prepare_sources(self.experiment)
        check_sources(self.experiment)
        self.assertEqual((self.working / "README").read_text(), "experiment patch\n")
        self.assertEqual(git(self.upstream, "status", "--porcelain"), "")

    def test_dirty_original_is_rejected(self):
        prepare_sources(self.experiment)
        (self.upstream / "README").write_text("unexpected original edit\n")
        with self.assertRaisesRegex(ValueError, "Keep .* clean"):
            prepare_sources(self.experiment)

    def test_working_revision_mismatch_is_not_reset(self):
        prepare_sources(self.experiment)
        git(self.working, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "--allow-empty", "-m", "unexpected revision")
        changed = git(self.working, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "Revision mismatch"):
            prepare_sources(self.experiment)
        self.assertEqual(git(self.working, "rev-parse", "HEAD"), changed)

    def test_nested_revision_mismatch_is_rejected(self):
        self.prepare_nested()
        tests = self.working / "generators/gemmini/software/gemmini-rocc-tests"
        git(tests, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "--allow-empty", "-m", "unexpected software revision")
        with self.assertRaisesRegex(ValueError, "Revision mismatch"):
            check_sources(self.experiment)

    def test_unregistered_source_is_rejected(self):
        git(self.repo, "update-index", "--force-remove", "third_party/chipyard")
        with self.assertRaisesRegex(ValueError, "registered"):
            prepare_sources(self.experiment)

    def test_working_symlink_cannot_redirect_patches_to_original(self):
        git(self.repo, "submodule", "update", "--init", "--", "third_party/chipyard")
        self.working.parent.mkdir(parents=True)
        self.working.symlink_to(self.upstream, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "not a symlink"):
            prepare_sources(self.experiment)

    def test_common_setup_initializes_act_but_not_chipyard(self):
        modules = self.repo / ".gitmodules"
        names = ("act", "taidl", "act-oracle", "act-backend")
        for name in names:
            source = repository(self.repo.parent / f"upstreams/{name}")
            path = f"third_party/{name}"
            entry = f'[submodule "{path}"]\n\tpath = {path}\n\turl = {source}\n'
            modules.write_text(modules.read_text() + entry)
            git(self.repo, "update-index", "--add", "--cacheinfo",
                f"160000,{git(source, 'rev-parse', 'HEAD')},{path}")
        git(self.repo, "add", ".gitmodules")
        scripts = self.repo / "scripts"
        scripts.mkdir()
        project = Path(__file__).resolve().parents[3]
        shutil.copyfile(project / "scripts/setup.sh", scripts / "setup.sh")
        # Only dependency initialization is under test. Skip environment/package
        # installation; Git still performs real clones for the four ACT repos.
        for name in ("install_third_party.sh", "build_ortools.sh"):
            script = scripts / "_internal" / name
            script.parent.mkdir(exist_ok=True)
            script.write_text("#!/usr/bin/env bash\nexit 0\n")
            script.chmod(0o755)
        conda = scripts / "fixture-conda"
        conda.write_text("#!/usr/bin/env bash\nprintf '%s\\n' 'conda() { return 0; }'\n")
        conda.chmod(0o755)
        result = subprocess.run(["bash", str(scripts / "setup.sh")],
                                env={**os.environ, "ACT_CONDA_EXE": str(conda)},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in names:
            self.assertTrue((self.repo / "third_party" / name / ".git").exists())
        self.assertFalse((self.upstream / ".git").exists())


if __name__ == "__main__":
    unittest.main()
