#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/_internal/install_third_party.sh [CONDA_ENV]

Install the pinned ACT packages from third_party/ without resolving their
Git-based dependencies.

  CONDA_ENV  Install through `conda run -n CONDA_ENV`.
             If omitted, use the Python interpreter active in this shell.

Examples:
  scripts/_internal/install_third_party.sh act
  conda activate act
  scripts/_internal/install_third_party.sh
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

packages=(
    "third_party/taidl"
    "third_party/act-oracle"
    "third_party/act-backend"
    "third_party/act"
)

for package in "${packages[@]}"; do
    if [[ ! -f "${repo_root}/${package}/pyproject.toml" ]]; then
        echo "error: missing ${package}/pyproject.toml" >&2
        echo "Initialize the repository submodules before installing." >&2
        exit 1
    fi
done

staging_root="$(mktemp -d "${TMPDIR:-/tmp}/act-install.XXXXXX")"
cleanup() {
    rm -rf -- "${staging_root}"
}
trap cleanup EXIT

if (( $# == 1 )); then
    if ! command -v conda >/dev/null 2>&1; then
        echo "error: conda is not available in this shell" >&2
        echo "Initialize Conda, or activate the environment and omit CONDA_ENV." >&2
        exit 1
    fi
    python_cmd=(conda run --no-capture-output -n "$1" python)
else
    python_cmd=(python)
fi

"${python_cmd[@]}" -c \
    'import sys; print(f"Installing with Python: {sys.executable}")'

for package in "${packages[@]}"; do
    package_name="${package##*/}"
    archive_path="${staging_root}/${package_name}.tar"
    source_path="${staging_root}/${package_name}"

    mkdir -p "${source_path}"
    git -C "${repo_root}/${package}" archive \
        --format=tar \
        --output="${archive_path}" \
        HEAD
    tar -xf "${archive_path}" -C "${source_path}"

    echo "Installing ${package}"
    "${python_cmd[@]}" -m pip install \
        --no-deps \
        --no-build-isolation \
        "${source_path}"
done

"${python_cmd[@]}" -c \
    'from importlib.metadata import version; [print(f"{p}: {version(p)}") for p in ("taidl", "taidl-to", "act-backend", "act-ecosystem")]'
