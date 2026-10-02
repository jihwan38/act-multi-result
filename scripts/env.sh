#!/usr/bin/env bash
# Source this file after Conda activation and prepare_sources.py.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Use: source scripts/env.sh baseline|multi-result' >&2
    exit 2
fi

_act_select_environment() {
    local repo_root exports
    if [[ $# != 1 ]]; then
        echo 'Use: source scripts/env.sh baseline|multi-result' >&2
        return 2
    fi
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    exports="$(python - "$repo_root" "$1" <<'PY'
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from _internal.common import ROOT, main_guard, process_env

def main():
    selected = sys.argv[2]
    env = process_env(selected)
    if not (ROOT / "build" / selected / "src/provenance.json").is_file():
        raise ValueError(f"Run python scripts/prepare_sources.py {selected} first")
    keys = ("ACT_TOOLCHAIN", "ACT_BUILD_DIR", "PYTHONPATH", "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE", "ORTOOLS_ROOT", "NLOHMANN_JSON_ROOT",
            "CARGO_BUILD_JOBS", "LD_LIBRARY_PATH")
    for key in keys:
        print(f"export {key}={shlex.quote(env[key])}")
    print("unset LD_PRELOAD PYTHONHOME CARGO_TARGET_DIR")

main_guard(main)
PY
    )" || return
    eval "$exports"
    printf 'ACT source: %s/src\nGenerated files: %s\n' "$ACT_BUILD_DIR" "$ACT_BUILD_DIR"
}

if _act_select_environment "$@"; then
    unset -f _act_select_environment
else
    unset -f _act_select_environment
    return 1
fi
