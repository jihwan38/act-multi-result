"""Exercise the shared verification CLI against real generated assemblies."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    qkv_hlo = ROOT / "cases/qkv/workloads/attention.hlo"
    double_hlo = ROOT / "cases/double_gemm/workloads/two_gemm_concat.hlo"
    multi_qkv = ROOT / "build/multi-result/runs/qkv/attention/assembly.py"
    double = ROOT / "build/multi-result/runs/double_gemm/two_gemm_concat/assembly.py"
    qkv_oracle = ROOT / "build/multi-result/targets/QKV/oracle"
    double_oracle = ROOT / "build/multi-result/targets/QKVDoubleGemm/oracle"
    isa = ROOT / "cases/double_gemm/QKVDoubleGemm.py"

    with tempfile.TemporaryDirectory(prefix="act-verify-") as temporary:
        directory = Path(temporary)

        def check(label, hlo, assembly, *extra, error=None):
            report = directory / f"{label}.json"
            if error is not None:
                report.write_text('{"passed": true}')
            argv = [sys.executable, ROOT / "scripts/verify.py", "--hlo", hlo, "--assembly", assembly,
                    "--report", report, *extra]
            result = subprocess.run([str(arg) for arg in argv], capture_output=True, text=True)
            output = result.stdout + result.stderr
            if error is None:
                assert result.returncode == 0, output
                data = json.loads(report.read_text())
                assert data["passed"] and data["datasets"] == 4, data
            else:
                assert result.returncode != 0 and error in output, output
                assert report.is_file(), "Failed run did not record its failure"
                assert not json.loads(report.read_text())["passed"], "Failed run left a PASS report"
                data = None
            print(f"PASS: {label}", flush=True)
            return data

        moved = directory / "moved.py"
        shutil.copy2(multi_qkv, moved)
        check("unknown-location", qkv_hlo, moved, error="pass --oracle")
        check("missing-input", directory / "missing.hlo", moved, "--oracle", qkv_oracle, error="Missing input")
        check("explicit-oracle", qkv_hlo, moved, "--oracle", qkv_oracle)
        data = check("qkv-allocation", qkv_hlo, multi_qkv)
        assert data["allocation"]["passed"]
        check("wrong-oracle", double_hlo, double, "--oracle", qkv_oracle, error="has no instruction")
        check("wrong-hlo", double_hlo, multi_qkv, error="input count differs")

        # Remove a real computation while keeping a runnable straight-line program.
        incorrect = directory / "incorrect.py"
        original = multi_qkv.read_text()
        assert "api.softmax(" in original
        incorrect.write_text("\n".join(line for line in original.splitlines() if "api.softmax(" not in line) + "\n")
        check("wrong-result", qkv_hlo, incorrect, "--oracle", qkv_oracle, error="Oracle/HLO results differ")

        # Equal result shapes must not hide a reversal of the two GEMM results.
        reversed_hlo = directory / "reversed_results.hlo"
        source = double_hlo.read_text()
        assert source.count("concatenate(%y0, %y1)") == 1
        reversed_hlo.write_text(source.replace("concatenate(%y0, %y1)", "concatenate(%y1, %y0)"))
        check("wrong-result-order", reversed_hlo, double, error="Oracle/HLO results differ")

        separate = next(path for path in sorted(double.parent.joinpath("compiler").glob("*.py"))
                        if "api.gemm_0(" in path.read_text() and "api.gemm_1(" in path.read_text()
                        and "api.double_gemm(" not in path.read_text())
        data = check("separate-gemms", double_hlo, separate)
        assert data["allocation"]["passed"] and not data["api_calls"].get("double_gemm")

        shutil.copy2(double, moved)
        sidecar = Path(str(moved) + ".allocation.json")
        allocation = json.loads(Path(str(double) + ".allocation.json").read_text())
        sidecar.write_text(json.dumps(allocation))
        selected = re.search(r"^# PII number: (\d+)$", double.read_text(), re.M)
        assert selected, "Assembly must identify its selected PII"
        pii = double.parent / "compiler" / f"{selected[1]}.pii"
        extra = ("--oracle", double_oracle, "--isa", isa, "--pii", pii)
        data = check("explicit-audit", double_hlo, moved, *extra)
        assert data["allocation"]["passed"]
        allocation["tensors"].pop(next(iter(allocation["tensors"])))
        sidecar.write_text(json.dumps(allocation))
        check("bad-allocation", double_hlo, moved, *extra, error="Lost/extra PII results")


if __name__ == "__main__":
    main()
