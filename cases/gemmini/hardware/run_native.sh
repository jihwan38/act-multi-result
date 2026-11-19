#!/usr/bin/env bash
# One fresh simulator process/reset per binary. No host-time performance claims.
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(cd "$experiment_dir/../.." && pwd)/build/multi-result-legacy"
if [[ $# != 2 || ! -f "$1" || -e "$2" || -e "$2.trace" || -e "$2.exit" || -e "$2.run.json" ]]; then
    echo "Usage: bash hardware/run_native.sh binary fresh-log" >&2
    exit 2
fi
binary="$(realpath "$1")"
log="$(realpath -m "$2")"
chipyard="$build_dir/hardware/chipyard"
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ ! -x "$conda_exe" ]]; then
    echo "Conda is required; activate the ACT environment first" >&2
    exit 2
fi
set +u
eval "$("$conda_exe" shell.bash hook)"
conda activate "$build_dir/deps/rtl"
set -u
unset LD_PRELOAD PYTHONPATH
export RISCV="$CONDA_PREFIX/riscv-tools"
export PATH="$RISCV/bin:$PATH"
# Cached simulators may retain an RPATH from the checkout's former location.
# Select the relocated toolchain libraries without changing the measured ELF.
export LD_LIBRARY_PATH="$RISCV/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
simulator="$chipyard/sims/verilator/simulator-chipyard.harness-ActDualGemminiRocketConfig"
mkdir -p "$(dirname "$log")"
# DRAMSim2 writes auxiliary files in its working directory; isolate each run.
run_dir="$(mktemp -d "$build_dir/rtl-run.XXXXXX")"
# Remove empty run directories on exit, but preserve any DRAMSim2 auxiliary data.
trap 'rmdir -- "$run_dir" 2>/dev/null || true' EXIT
cd "$run_dir"
python "$experiment_dir/hardware/run_record.py" --binary "$binary" --log "$log" -- \
    "$simulator" +permissive +dramsim \
    "+dramsim_ini_dir=$chipyard/generators/testchipip/src/main/resources/dramsim2_ini" \
    +max-cycles=2000000 +verbose "+loadmem=$binary" +permissive-off "$binary"
