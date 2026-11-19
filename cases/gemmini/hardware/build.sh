#!/usr/bin/env bash
# Dependency setup is separate: bash hardware/setup.sh
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(cd "$experiment_dir/../.." && pwd)/build/multi-result-legacy"
chipyard="$build_dir/hardware/chipyard"
gemmini="$chipyard/generators/gemmini"
mode=${1:-dual}
[[ "$mode" == single || "$mode" == dual ]] || { echo "Expected single or dual" >&2; exit 2; }
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
[[ -x "$conda_exe" ]] || { echo "Activate Conda first" >&2; exit 2; }
set +u
eval "$("$conda_exe" shell.bash hook)"
conda activate "$build_dir/deps/rtl"
set -u
unset LD_PRELOAD PYTHONPATH
export RISCV="$CONDA_PREFIX/riscv-tools"
export PATH="$RISCV/bin:$PATH"
python "$experiment_dir/hardware/source_tree.py" check

if [[ "$mode" == dual ]]; then
    observation="$experiment_dir/hardware/patches/0001-observe-gemmini-rtl-activity.patch"
    if git -C "$gemmini" apply --reverse --check "$observation" 2>/dev/null; then
        echo "Observation patch already applied"
    else
        git -C "$gemmini" apply --check "$observation"
        git -C "$gemmini" apply "$observation"
    fi
    config="$chipyard/generators/chipyard/src/main/scala/config/ActGemminiConfigs.scala"
    if [[ -e "$config" ]]; then
        cmp "$experiment_dir/hardware/ActGemminiConfigs.scala" "$config"
    else
        cp "$experiment_dir/hardware/ActGemminiConfigs.scala" "$config"
    fi
    export ACT_GEMMINI_TRACE=1
    config_name=ActDualGemminiRocketConfig
else
    # This condition removes all observation circuitry at elaboration, including
    # when build/ contains an already-patched clone from a prior dual build.
    export ACT_GEMMINI_TRACE=0
    config_name=GemminiRocketConfig
fi
cd "$chipyard/sims/verilator"
make -j"${NPROC:-8}" "CONFIG=$config_name" JAVA_HEAP_SIZE=12G

if [[ "$mode" == single ]]; then
    if rg -q 'ACT_GEMMINI opcode=' "generated-src/chipyard.harness.TestHarness.$config_name/gen-collateral" -g '*.sv'; then
        echo "Single baseline was cached with observers enabled; use a fresh build" >&2
        exit 1
    fi
    tests="$gemmini/software/gemmini-rocc-tests"
    (cd "$tests" && ./build.sh -j4 BAREMETAL_ONLY=1 bareMetalC)
    make run-binary CONFIG=GemminiRocketConfig LOADMEM=1 \
        "BINARY=$tests/build/bareMetalC/tiled_matmul_ws_full_C-baremetal"
else
    # The two elaborated headers must differ ONLY in their routing opcode.
    python - "$gemmini/software/gemmini-rocc-tests/include" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
a = (root/'gemmini_lane0_params.h').read_text()
b = (root/'gemmini_lane1_params.h').read_text()
assert '#define XCUSTOM_ACC 3' in a and '#define XCUSTOM_ACC 2' in b
assert a == b.replace('#define XCUSTOM_ACC 2', '#define XCUSTOM_ACC 3')
PY
    # Detect a cached build made without the observation switch. Make does not
    # know that an environment variable changes Scala elaboration.
    observations=$(rg -l 'ACT_GEMMINI opcode=' "generated-src/chipyard.harness.TestHarness.$config_name/gen-collateral" -g '*.sv' | wc -l)
    [[ "$observations" == 2 ]] || { echo "Expected two RTL observers; use a fresh build" >&2; exit 1; }
fi
python - "$experiment_dir" "$config_name" "$build_dir" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
config = sys.argv[2]
build = pathlib.Path(sys.argv[3])
binary = build/'hardware/chipyard/sims/verilator'/('simulator-chipyard.harness-'+config)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = json.loads(subprocess.check_output(
    [sys.executable, str(root/'hardware/source_tree.py'), 'check'], text=True))
report = {'config': config, 'rtl_sha256': digest(binary),
          'versions': {**json.loads((root/'hardware/versions.json').read_text()), **sources},
          'config_sha256': digest(root/'hardware/ActGemminiConfigs.scala'),
          'observer_sha256': digest(root/'hardware/patches/0001-observe-gemmini-rtl-activity.patch'),
          'verilator': subprocess.check_output(['verilator', '--version'], text=True).strip(),
          'note': 'Build provenance; single mode also ran the official full-C numerical test.'}
(build/('hardware-'+config+'.json')).write_text(json.dumps(report, indent=2)+'\n')
PY
