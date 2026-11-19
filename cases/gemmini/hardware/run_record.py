"""Record executable identity before RTL execution, then seal its raw evidence.

Hashes detect stale or accidentally changed artifacts, not malicious replacement
of both the evidence and this unsigned record. No legacy logs are backfilled.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_path(log):
    return Path(str(log) + ".run.json")


def record_run(binary, log, command):
    """Run the supplied simulator command; interrupted runs remain incomplete."""
    binary, log = binary.resolve(), log.resolve()
    simulator = Path(command[0]).resolve()
    command = [str(simulator), *command[1:]]
    if command[-1] != str(binary) or f"+loadmem={binary}" not in command:
        raise ValueError("Recorded native binary must match both ELF and +loadmem arguments")
    trace, exit_path = Path(str(log) + ".trace"), Path(str(log) + ".exit")
    manifest = record_path(log)
    for path in (log, trace, exit_path, manifest):
        if path.exists():
            raise ValueError(f"Use fresh run artifacts: {path}")
    record = {"schema_version": 1, "status": "started", "command": command,
              "simulator": str(simulator), "binary": str(binary),
              "rtl_sha256": sha(simulator), "binary_sha256": sha(binary)}
    log.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting an existing run. This record is
    # on disk before the simulator starts, not reconstructed by the verifier.
    with manifest.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    with log.open("x") as stdout, trace.open("x") as stderr:
        process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    with exit_path.open("x") as stream:
        stream.write(f"{process.returncode}\n")
    record.update({"returncode": process.returncode,
                   "rtl_sha256_after": sha(simulator), "binary_sha256_after": sha(binary),
                   "stdout_sha256": sha(log), "trace_sha256": sha(trace)})
    unchanged = all(record[key] == record[key + "_after"] for key in ("rtl_sha256", "binary_sha256"))
    record["status"] = "complete" if unchanged else "invalidated"
    # A crash during finalization leaves either the started or complete record,
    # never a partially written manifest that could be mistaken for a pass.
    with tempfile.NamedTemporaryFile(mode="w", dir=log.parent, prefix=".run-record.", delete=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, manifest)
    if not unchanged:
        raise AssertionError("Simulator or native binary changed during RTL execution")
    print(f"RTL simulator exit status: {process.returncode}")
    print(f"Run record: {manifest}")
    for key, path in (("rtl_sha256", simulator), ("binary_sha256", binary),
                      ("stdout_sha256", log), ("trace_sha256", trace)):
        print(f"{record[key]}  {path}")
    return process.returncode


def verify_run_record(log, binary=None):
    """Check saved execution evidence without reading today's simulator binary."""
    manifest = record_path(log)
    if not manifest.is_file():
        raise AssertionError("Missing execution-time run record; rerun RTL, do not backfill old logs")
    record = json.loads(manifest.read_text())
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise AssertionError("Incomplete or invalidated RTL run record")
    if record["returncode"] != 0 or Path(str(log) + ".exit").read_text().strip() != "0":
        raise AssertionError("RTL process did not finish successfully")
    for key in ("rtl_sha256", "binary_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", record[key]) or record[key] != record[key + "_after"]:
            raise AssertionError("Executable changed during RTL execution")
    for key, path in (("stdout_sha256", log), ("trace_sha256", Path(str(log) + ".trace"))):
        if sha(path) != record[key]:
            raise AssertionError(f"Raw {key.removesuffix('_sha256')} changed since RTL execution")
    if binary is not None and sha(binary) != record["binary_sha256"]:
        raise AssertionError("Native binary differs from the execution-time run record")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a simulator command after --")
    code = record_run(args.binary, args.log, command)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
