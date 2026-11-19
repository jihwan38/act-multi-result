"""Prepare a private Chipyard checkout; Git links are the source revision pins.

The registered upstream stays clean. Existing experiment checkouts may contain
our patches/build products, but a different HEAD is never reset or overwritten.
Nested dependencies are initialized by Chipyard's own selective setup script.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


UPSTREAM = "third_party/chipyard"


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def require_revision(directory, expected):
    if not (directory / ".git").exists():
        raise ValueError(f"Uninitialized repository: {directory}; run hardware/setup.sh")
    actual = git(directory, "rev-parse", "HEAD")
    if actual != expected:
        raise ValueError(f"Revision mismatch at {directory}: expected {expected}, got {actual}; not overwriting")


def upstream_pin(experiment, initialize=False):
    repo = experiment.parents[1]
    # The index supports both a committed gitlink and its staged initial addition.
    entries = git(repo, "ls-files", "--stage", "--", UPSTREAM).splitlines()
    if len(entries) != 1:
        raise ValueError(f"Expected one registered {UPSTREAM} submodule")
    fields = entries[0].split()
    if len(fields) != 4 or fields[0] != "160000" or fields[2] != "0" or fields[3] != UPSTREAM:
        raise ValueError(f"Expected an unconflicted gitlink for {UPSTREAM}")
    revision = fields[1]
    upstream = repo / UPSTREAM
    if initialize and not (upstream / ".git").exists():
        subprocess.run(
            ["git", "-C", str(repo), "submodule", "update", "--init", "--", UPSTREAM],
            check=True, stdout=sys.stderr,
        )
    require_revision(upstream, revision)
    if git(upstream, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"Keep {upstream} clean; put experiment changes in patches/")
    url = git(repo, "config", "--file", ".gitmodules", "--get", f"submodule.{UPSTREAM}.url")
    return upstream, {"path": UPSTREAM, "url": url, "revision": revision}


def prepare_sources(experiment):
    upstream, pin = upstream_pin(experiment, initialize=True)
    working = experiment.parents[1] / "build/multi-result-legacy/hardware/chipyard"
    if working.is_symlink():
        raise ValueError(f"Working source must be an independent checkout, not a symlink: {working}")
    if not working.exists():
        working.parent.mkdir(parents=True, exist_ok=True)
        # Independent Git objects: deleting build/ must not damage the original,
        # and the worktree must not borrow objects from a temporary checkout.
        subprocess.run(
            ["git", "clone", "--no-hardlinks", "--no-checkout", "--no-recurse-submodules",
             str(upstream), str(working)], check=True, stdout=sys.stderr,
        )
        subprocess.run(
            ["git", "-C", str(working), "remote", "set-url", "origin", pin["url"]], check=True,
        )
        subprocess.run(
            ["git", "-C", str(working), "checkout", "--detach", pin["revision"]],
            check=True, stdout=sys.stderr,
        )
    require_revision(working, pin["revision"])
    return {"chipyard": pin}


def gitlink(directory, revision, path):
    fields = git(directory, "ls-tree", revision, "--", path).split()
    if len(fields) != 4 or fields[0:2] != ["160000", "commit"] or fields[3] != path:
        raise ValueError(f"No gitlink for {path} in {directory} at {revision}")
    return fields[2]


def check_sources(experiment):
    upstream, pin = upstream_pin(experiment)
    working = experiment.parents[1] / "build/multi-result-legacy/hardware/chipyard"
    if working.is_symlink():
        raise ValueError(f"Working source must be an independent checkout, not a symlink: {working}")
    require_revision(working, pin["revision"])
    result = {"chipyard": pin}
    for name, path in (("gemmini_revision", "generators/gemmini"),
                       ("rocket_chip_revision", "generators/rocket-chip")):
        expected = gitlink(upstream, pin["revision"], path)
        require_revision(working / path, expected)
        result[name] = expected
    # Read the software pin from the pinned Gemmini commit, not its dirty files.
    gemmini = working / "generators/gemmini"
    tests_path = "software/gemmini-rocc-tests"
    expected = gitlink(gemmini, result["gemmini_revision"], tests_path)
    require_revision(gemmini / tests_path, expected)
    result["gemmini_tests_revision"] = expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1]
    action = prepare_sources if args.action == "prepare" else check_sources
    try:
        print(json.dumps(action(experiment), indent=2))
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
