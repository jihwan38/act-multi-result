#!/usr/bin/env bash
# Run as a process, not with source. ACT's Conda environment is not modified.
set -euo pipefail

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$(cd "$experiment_dir/../.." && pwd)/build/multi-result-legacy"
chipyard_dir="$build_dir/hardware/chipyard"
rtl_env="$build_dir/deps/rtl"
lock_env="$build_dir/deps/conda-lock"
conda_exe=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$conda_exe" || ! -x "$conda_exe" ]]; then
    echo "Activate Conda first, or set CONDA_EXE to its executable." >&2
    exit 1
fi
python_exe="$(dirname -- "$conda_exe")/python"
[[ -x "$python_exe" ]] || { echo "Cannot find Conda's Python" >&2; exit 1; }

# Avoid importing ACT modules or host-preloaded C++ runtimes into the RTL tools.
unset PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1
export NPROC=${NPROC:-8}
mkdir -p "$build_dir/hardware" "$build_dir/deps"

read_version() {
    "$python_exe" -c 'import json,sys; v=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): v=v[key]
print(v)' "$experiment_dir/hardware/versions.json" "$1"
}

"$python_exe" "$experiment_dir/hardware/source_tree.py" prepare
lockfile="$chipyard_dir/$(read_version conda_lock.path)"
[[ $(sha256sum "$lockfile" | cut -d ' ' -f 1) == "$(read_version conda_lock.sha256)" ]] || {
    echo "Chipyard environment lock checksum differs" >&2; exit 1;
}

if [[ ! -x "$lock_env/bin/conda-lock" ]]; then
    "$conda_exe" create -y -p "$lock_env" -c conda-forge \
        "conda-lock=$(read_version conda_lock.installer_version)"
fi
if [[ ! -f "$rtl_env/.chipyard-lock.sha256" ]]; then
    "$lock_env/bin/conda-lock" install --conda "$conda_exe" -p "$rtl_env" "$lockfile"
    sha256sum "$lockfile" > "$rtl_env/.chipyard-lock.sha256"
fi
[[ $(cut -d ' ' -f 1 "$rtl_env/.chipyard-lock.sha256") == "$(read_version conda_lock.sha256)" ]] || {
    echo "Existing RTL environment has a different lock" >&2; exit 1;
}

# The upstream initializer intentionally avoids Linux/FireSim's large recursive
# trees. Do not replace it with an unrestricted recursive submodule checkout.
cd "$chipyard_dir"
if [[ ! -f "$build_dir/submodules.initialized" ]]; then
    ./scripts/init-submodules-no-riscv-tools.sh
    # Verify the software checkout too: the upstream script has permissive error handling.
    test -f generators/gemmini/software/gemmini-rocc-tests/include/gemmini.h
    "$python_exe" "$experiment_dir/hardware/source_tree.py" check
    touch "$build_dir/submodules.initialized"
fi
# Check even when reusing an existing initialization stamp.
"$python_exe" "$experiment_dir/hardware/source_tree.py" check

source "$(dirname -- "$conda_exe")/../etc/profile.d/conda.sh"
# Chipyard's packaged activation hooks read unset optional variables (RISCV).
set +u
conda activate "$rtl_env"
set -u
export RISCV="$rtl_env/riscv-tools"
export PATH="$RISCV/bin:$PATH"
if [[ ! -f "$build_dir/toolchain-core.initialized" ]]; then
    # Use pinned official build steps; omit only unused FPGA/Linux/navigation work
    # and Scala precompilation (the actual simulator build compiles it later).
    ./build-setup.sh --use-lean-conda --skip-conda --skip-submodules \
        --skip-precompile --skip-clean --skip-circt
    test -x "$RISCV/bin/riscv64-unknown-elf-gcc"
    test -f "$RISCV/lib/libfesvr.a"
    touch "$build_dir/toolchain-core.initialized"
fi

# Same upstream release and bytes. curl works in environments where wget's
# direct release-assets connection stalls; no compiler/RTL source is substituted.
circt_version=$(read_version circt.version)
[[ $("$python_exe" -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' conda-reqs/circt.json) == "$circt_version" ]]
archive="$build_dir/downloads/circt-full-static-linux-x64-1.75.0.tar.gz"
mkdir -p "$build_dir/downloads"
if [[ ! -f "$archive" ]]; then
    curl -fL --retry 2 --connect-timeout 10 --max-time 600 \
        "https://github.com/llvm/circt/releases/download/$circt_version/$(read_version circt.artifact)" \
        -o "$archive.partial"
    [[ $(sha256sum "$archive.partial" | cut -d ' ' -f 1) == "$(read_version circt.sha256)" ]]
    mv "$archive.partial" "$archive"
fi
[[ $(sha256sum "$archive" | cut -d ' ' -f 1) == "$(read_version circt.sha256)" ]]
if [[ ! -f "$build_dir/circt.initialized" ]]; then
    tar -xzf "$archive" -C "$RISCV" --strip-components 1
    test -x "$RISCV/bin/firtool"
    touch "$build_dir/circt.initialized"
fi
verilator --version
"$RISCV/bin/riscv64-unknown-elf-gcc" --version
"$RISCV/bin/firtool" --version
echo "RTL dependencies prepared. This is not a simulator or performance test result."
