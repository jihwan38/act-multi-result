"""Shared paths and process isolation for the two ACT toolchains."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("act", "taidl", "act-backend", "act-oracle")
TOOLCHAINS = {"baseline": ("baseline",), "multi-result": ("baseline", "multi-result")}
# The run shortcut and verify's default paths use this list. ISA scripts run directly.
CASES = {
    "qkv": {"target": "QKV", "isa": "QKV.py",
            "default_workload": "workloads/attention.hlo", "default_toolchain": "baseline",
            "toolchains": ("baseline", "multi-result")},
    "double_gemm": {"target": "QKVDoubleGemm", "isa": "QKVDoubleGemm.py",
                    "default_workload": "workloads/two_gemm_concat.hlo", "default_toolchain": "multi-result",
                    "toolchains": ("multi-result",)},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def source_hashes(directory):
    """Exclude generated native/Cargo builds; retain all source files."""
    result = {}
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "build", "target", "__pycache__"))
        for name in sorted(files):
            path = Path(base) / name
            if name != "provenance.json" and not name.endswith((".pyc", ".pyo")):
                result[str(path.relative_to(directory))] = digest(path)
    return result


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def upstreams():
    result = {}
    for name in COMPONENTS:
        relative = f"third_party/{name}"
        entry = git("ls-files", "--stage", "--", relative).split()
        if len(entry) != 4 or entry[0] != "160000":
            raise ValueError(f"Missing pinned submodule: {relative}")
        directory = ROOT / relative
        if not (directory / ".git").exists():
            raise ValueError("Initialize upstreams with ./scripts/setup.sh")
        revision = git("rev-parse", "HEAD", cwd=directory)
        if revision != entry[1] or git("status", "--porcelain", cwd=directory):
            raise ValueError(f"{relative} must be clean at its recorded commit {entry[1]}")
        result[name] = revision
    return result


def patch_series(group):
    directory = ROOT / "patches" / group
    paths = []
    for line in (directory / "series").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        path = (directory / line).resolve()
        if (not path.is_relative_to(directory) or len(Path(line).parts) != 2
                or Path(line).parts[0] not in COMPONENTS[1:] or path.suffix != ".patch"):
            raise ValueError(f"Invalid patch entry in {group}/series: {line}")
        paths.append(path)
    if len(paths) != len(set(paths)) or set(paths) != set(directory.glob("*/*.patch")):
        raise ValueError(f"{group}/series must list every patch exactly once")
    return paths


def source_plan(toolchain):
    patches = [path for group in TOOLCHAINS[toolchain] for path in patch_series(group)]
    return {"toolchain": toolchain, "upstreams": upstreams(),
            "patches": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in patches]}


def case_config(name, toolchain=None):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid case name: {name}")
    directory = ROOT / "cases" / name
    if name not in CASES:
        raise ValueError(f"Unknown case: {name}")
    config = CASES[name]
    selected = toolchain or config["default_toolchain"]
    if selected not in TOOLCHAINS or selected not in config["toolchains"]:
        raise ValueError(f"Case {name} supports: {', '.join(config['toolchains'])}; requested {selected}")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", config["target"]):
        raise ValueError("Invalid target name")
    for key in ("isa", "default_workload"):
        path = (directory / config[key]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Invalid {key} for {name}")
    return directory, config, selected


def process_env(toolchain):
    if toolchain not in TOOLCHAINS:
        raise ValueError(f"Unknown toolchain: {toolchain}")
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix or not Path(sys.executable).resolve().is_relative_to(Path(prefix).resolve()):
        raise ValueError("Activate the Conda environment first: conda activate act")
    ortools = ROOT / "build/deps/ortools"
    if not (ortools / "lib/cmake/ortools/ortoolsConfig.cmake").is_file():
        raise ValueError("Project-local OR-Tools is missing; run ./scripts/setup.sh")
    if not (Path(prefix) / "include/nlohmann/json.hpp").is_file():
        raise ValueError("nlohmann_json is missing from the active environment")
    env = dict(os.environ)
    for key in ("LD_PRELOAD", "PYTHONHOME", "CARGO_TARGET_DIR"):
        env.pop(key, None)
    src = ROOT / "build" / toolchain / "src"
    env.update(PYTHONPATH=os.pathsep.join(str(src / component) for component in COMPONENTS),
               ACT_TOOLCHAIN=toolchain, ACT_BUILD_DIR=str(src.parent),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               ORTOOLS_ROOT=str(ortools), NLOHMANN_JSON_ROOT=prefix,
               CARGO_BUILD_JOBS=os.environ.get("ACT_BUILD_JOBS", "4"))
    # A host C++ runtime must not take precedence over the selected environment.
    env["LD_LIBRARY_PATH"] = os.pathsep.join((str(Path(prefix) / "lib"), str(ortools / "lib")))
    return env


@contextmanager
def toolchain_lock(toolchain):
    base = ROOT / "build" / toolchain
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another {toolchain} build/run is active") from None
        yield


def command(args, *, env, log, cwd=ROOT):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-50:]), file=sys.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}); log: {log}")


def main_guard(function):
    try:
        function()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
