#!/usr/bin/env bash
# Use the same upstream bare-metal startup, linker script and libc as its tests.
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(cd "$experiment_dir/../.." && pwd)/build/multi-result-legacy"
if [[ $# != 2 || ! -f "$1" || -e "$2" ]]; then
    echo "Usage: bash native/build.sh generated.c fresh-binary" >&2
    exit 2
fi
rtl_prefix="$build_dir/deps/rtl"
test_dir="$build_dir/hardware/chipyard/generators/gemmini/software/gemmini-rocc-tests"
common_dir="$test_dir/riscv-tests/benchmarks/common"
compiler="$rtl_prefix/riscv-tools/bin/riscv64-unknown-elf-gcc"
"$compiler" -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu11 -O2 \
    -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
    -march=rv64gc -Wa,-march=rv64gc -I"$test_dir/riscv-tests" \
    -I"$test_dir/riscv-tests/env" -I"$test_dir" -I"$common_dir" -I"$experiment_dir/native" \
    -DPRINT_TILE=0 -nostdlib -nostartfiles -static -T "$common_dir/test.ld" \
    -Wl,--section-start=.act_memory=0x81000000 \
    -DBAREMETAL=1 "$1" "$common_dir"/*.c "$common_dir"/*.S -lm -lgcc -o "$2"
"$rtl_prefix/riscv-tools/bin/riscv64-unknown-elf-objdump" -d "$2" > "$2.disassembly"
"$rtl_prefix/bin/python" - "$1" "$2" "$compiler" "$experiment_dir/native/gemmini_dual.h" <<'PY'
import hashlib, json, pathlib, subprocess, sys
source, binary, compiler, header = map(pathlib.Path, sys.argv[1:])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {"source_sha256": digest(source), "binary_sha256": digest(binary),
          "header_sha256": digest(header), "compiler_sha256": digest(compiler),
          "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True),
          "memory_section": "0x81000000", "flags": "native/build.sh"}
pathlib.Path(str(binary) + ".build.json").write_text(json.dumps(record, indent=2) + "\n")
PY
sha256sum "$1" "$2" "$experiment_dir/native/gemmini_dual.h"
