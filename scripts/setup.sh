#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: scripts/setup.sh

Initialize pinned upstreams, create/update the Conda environment, install ACT
packages, and build project-local OR-Tools. No case is built by this command.

Environment:
  ACT_ENV_NAME    Conda environment name (default: act)
  ACT_CONDA_EXE   Explicit Conda executable, if it cannot be found automatically
  ACT_BUILD_JOBS  Native build parallelism (default: 4)
HELP
    exit 0
fi
if (( $# )); then echo 'Usage: scripts/setup.sh' >&2; exit 2; fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${ACT_ENV_NAME:-act}"
find_conda() {
    local candidate
    for candidate in "${ACT_CONDA_EXE:-}" "${CONDA_EXE:-}" \
        "$(command -v conda || true)" \
        "${HOME}/miniforge3/bin/conda" "${HOME}/miniconda3/bin/conda"; do
        if [[ -n ${candidate} && -x ${candidate} ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}
if ! conda_executable="$(find_conda)"; then
    echo 'Conda was not found. Install Miniforge or set ACT_CONDA_EXE.' >&2
    exit 1
fi
eval "$("${conda_executable}" shell.bash hook)"
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export ACT_BUILD_JOBS="${ACT_BUILD_JOBS:-4}"
unset PYTHONPATH LD_PRELOAD

echo '[1/4] Initialize pinned upstream submodules'
git submodule update --init --recursive -- \
    third_party/act third_party/taidl third_party/act-backend third_party/act-oracle

if conda run -n "$env_name" python -c pass >/dev/null 2>&1; then
    echo "[2/4] Update Conda environment: $env_name"
    conda env update -n "$env_name" -f environment.yml
else
    echo "[2/4] Create Conda environment: $env_name"
    conda env create -n "$env_name" -f environment.yml
fi
conda activate "$env_name"

echo '[3/4] Install pinned ACT packages'
./scripts/_internal/install_third_party.sh
python - <<'PY'
from importlib.metadata import version
import jax

expected = {"numpy": "1.26.4", "jax": "0.4.33", "jaxlib": "0.4.33", "ml-dtypes": "0.4.1"}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Dependency versions differ from environment.yml: {actual}")
print(f"Numerical dependencies: {actual}")
print(f"XLA CPU devices: {jax.devices('cpu')}")
PY

echo '[4/4] Build project-local OR-Tools'
./scripts/_internal/build_ortools.sh
printf '\nSetup complete.\n  conda activate %s\n  python scripts/prepare_sources.py baseline\n  source scripts/env.sh baseline\n  python cases/qkv/QKV.py\n' "$env_name"
