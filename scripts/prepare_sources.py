#!/usr/bin/env python3
"""Apply a named patch series to fresh archives of the pinned upstreams."""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile

from _internal.common import (ROOT, COMPONENTS, TOOLCHAINS, fingerprint, main_guard,
                    source_hashes, source_plan, toolchain_lock, write_json)


def prepare(toolchain, force=False):
    plan = source_plan(toolchain)
    signature = fingerprint(plan)
    base = ROOT / "build" / toolchain
    output = base / "src"
    previous = output / "provenance.json"
    if output.exists():
        if not previous.is_file():
            if not force:
                raise ValueError(f"Unmanaged source directory: {output}; use --force to replace it")
        else:
            record = json.loads(previous.read_text())
            if source_hashes(output) != record["files"] and not force:
                raise ValueError(f"Prepared sources were edited: {output}; export your patch or use --force")
            if record["fingerprint"] == signature and not force:
                return record
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=base) as temporary:
        from pathlib import Path
        stage = Path(temporary) / "src"
        stage.mkdir()
        for component in COMPONENTS:
            destination = stage / component
            destination.mkdir()
            archive = Path(temporary) / f"{component}.tar"
            subprocess.run(["git", "-C", str(ROOT / "third_party" / component), "archive",
                            "--format=tar", f"--output={archive}", plan["upstreams"][component]], check=True)
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    path = Path(member.name)
                    if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unexpected archive member: {member.name}")
                bundle.extractall(destination)
            env = {**os.environ, "GIT_CEILING_DIRECTORIES": temporary}
            for patch in plan["patches"]:
                if Path(patch["path"]).parts[-2] != component:
                    continue
                print(f"Applying {patch['path']}", flush=True)
                base_command = ["git", "-C", str(destination), "apply", "--no-index"]
                for flags in (["--check"], [], ["--reverse", "--check"]):
                    subprocess.run([*base_command, *flags, str(ROOT / patch["path"])], env=env, check=True)
        record = {**plan, "fingerprint": signature, "files": source_hashes(stage)}
        write_json(stage / "provenance.json", record)
        if output.exists():
            shutil.rmtree(output)
        stage.replace(output)
    print(f"Prepared {toolchain}: {output}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toolchain", choices=TOOLCHAINS)
    parser.add_argument("--force", action="store_true", help="Replace edited prepared sources")
    args = parser.parse_args()
    with toolchain_lock(args.toolchain):
        prepare(args.toolchain, args.force)


if __name__ == "__main__":
    main_guard(main)
