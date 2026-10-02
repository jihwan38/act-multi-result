#!/usr/bin/env bash

set -euo pipefail

readonly ORTOOLS_REPOSITORY="https://github.com/google/or-tools.git"
readonly ORTOOLS_TAG="v9.12"
readonly ORTOOLS_REVISION="b8e881fbde473a9e33e0dac475e498559eb0459d"

usage() {
    cat <<'EOF'
Usage: scripts/_internal/build_ortools.sh [CONDA_ENV]

Build the OR-Tools revision used by the official ACT v1.2 Docker image and
install it under build/deps/ortools.

  CONDA_ENV  Run the native build tools through `conda run -n CONDA_ENV`.
             If omitted, use the tools active in this shell.

Environment:
  ACT_BUILD_JOBS  Number of parallel build jobs. Defaults to 4.

Examples:
  scripts/_internal/build_ortools.sh act
  conda activate act
  scripts/_internal/build_ortools.sh
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
deps_root="${repo_root}/build/deps"
source_dir="${deps_root}/src/or-tools"
cmake_build_dir="${deps_root}/build/or-tools"
install_dir="${deps_root}/ortools"
revision_stamp="${install_dir}/.act-ortools-revision"

use_conda=0
conda_env=""
if (( $# == 1 )); then
    if ! command -v conda >/dev/null 2>&1; then
        echo "error: conda is not available in this shell" >&2
        echo "Initialize Conda, or activate the environment and omit CONDA_ENV." >&2
        exit 1
    fi
    use_conda=1
    conda_env="$1"
fi

# Native build tools must not inherit a host-injected C++ runtime. This changes
# only the environment of the child command; the caller's shell is untouched.
run_native() {
    if (( use_conda == 1 )); then
        env -u LD_PRELOAD -u LD_LIBRARY_PATH \
            conda run --no-capture-output -n "${conda_env}" \
            env -u LD_PRELOAD -u LD_LIBRARY_PATH "$@"
    else
        env -u LD_PRELOAD -u LD_LIBRARY_PATH "$@"
    fi
}

if [[ -n ${LD_PRELOAD:-} ]]; then
    echo "Ignoring inherited LD_PRELOAD for native build child processes"
fi

run_native git --version
run_native cmake --version
run_native ninja --version

if [[ -n ${ACT_BUILD_JOBS:-} ]]; then
    build_jobs="${ACT_BUILD_JOBS}"
else
    build_jobs=4
fi

if [[ ! ${build_jobs} =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ACT_BUILD_JOBS must be a positive integer" >&2
    exit 1
fi

has_ortools_library=0
for candidate in "${install_dir}"/lib/libortools.so*; do
    if [[ -e ${candidate} ]]; then
        has_ortools_library=1
        break
    fi
done

if [[ -f ${revision_stamp} ]] \
    && [[ $(<"${revision_stamp}") == "${ORTOOLS_REVISION}" ]] \
    && [[ -f ${install_dir}/lib/cmake/ortools/ortoolsConfig.cmake ]] \
    && (( has_ortools_library == 1 )); then
    echo "OR-Tools ${ORTOOLS_TAG} is already installed at ${install_dir}"
    exit 0
fi

mkdir -p "$(dirname -- "${source_dir}")" "${cmake_build_dir}" "${install_dir}"

if [[ ! -d ${source_dir}/.git ]]; then
    if [[ -e ${source_dir} ]]; then
        echo "error: ${source_dir} exists but is not a Git checkout" >&2
        exit 1
    fi

    run_native git clone \
        --depth 1 \
        --branch "${ORTOOLS_TAG}" \
        --single-branch \
        "${ORTOOLS_REPOSITORY}" \
        "${source_dir}"
fi

actual_revision="$(run_native git -C "${source_dir}" rev-parse HEAD)"
if [[ ${actual_revision} != "${ORTOOLS_REVISION}" ]]; then
    echo "error: OR-Tools revision mismatch" >&2
    echo "expected: ${ORTOOLS_REVISION}" >&2
    echo "actual:   ${actual_revision}" >&2
    echo "Remove the ignored source checkout and rerun the script." >&2
    exit 1
fi

# ACT uses constraint_solver. These optional MIP backends are unnecessary;
# SCIP's bundled build also assumes its source path contains no spaces.
run_native cmake \
    -S "${source_dir}" \
    -B "${cmake_build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_DEPS=ON \
    -DUSE_SCIP=OFF \
    -DUSE_COINOR=OFF \
    -DUSE_HIGHS=OFF \
    -DBUILD_SAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PYTHON=OFF \
    -DBUILD_FLATZINC=OFF \
    -DBUILD_TESTING=OFF

run_native cmake --build "${cmake_build_dir}" --parallel "${build_jobs}"
run_native cmake --install "${cmake_build_dir}"

if [[ ! -f ${install_dir}/lib/cmake/ortools/ortoolsConfig.cmake ]]; then
    echo "error: OR-Tools CMake package was not installed" >&2
    exit 1
fi

has_ortools_library=0
for candidate in "${install_dir}"/lib/libortools.so*; do
    if [[ -e ${candidate} ]]; then
        has_ortools_library=1
        break
    fi
done

if (( has_ortools_library == 0 )); then
    echo "error: OR-Tools shared library was not installed" >&2
    exit 1
fi

printf '%s\n' "${ORTOOLS_REVISION}" > "${revision_stamp}"

echo "OR-Tools ${ORTOOLS_TAG} installed successfully"
echo "Revision: ${ORTOOLS_REVISION}"
echo "Prefix:   ${install_dir}"
