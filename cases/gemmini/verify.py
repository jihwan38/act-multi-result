"""Compare ACT addresses/Oracle with unchanged HLO and independent integer math.

This is functional validation only; RTL cycle results are a separate gate.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jax
from jaxlib import xla_client
import numpy as np

from assembly import read_program, run_address_model


def make_inputs(count, dataset):
    if dataset == "zero":
        return [np.zeros((16, 16), dtype=np.int8) for _ in range(count)]
    if dataset == "identity":
        return [np.eye(16, dtype=np.int8) * (i + 1) for i in range(count)]
    return [np.random.default_rng(dataset * 100 + i).integers(
        -128, 128, size=(16, 16), dtype=np.int16).astype(np.int8) for i in range(count)]


def reference(name, inputs):
    values = [value.astype(np.int64) for value in inputs]
    if name == "two_gemm":
        outputs = [values[0] @ values[1], values[2] @ values[3]]
    elif name == "shared_input":
        outputs = [values[0] @ values[1], values[0] @ values[2]]
    elif name == "four_gemm":
        outputs = [values[i] @ values[i + 1] for i in range(0, 8, 2)]
    elif name == "dependent_gemm":
        intermediate = values[0] @ values[1]
        quantized = np.clip(intermediate, -128, 127)
        outputs = [quantized @ values[2]]
    else:
        raise ValueError(f"No independent reference for {name}")
    if any(np.max(np.abs(value)) > np.iinfo(np.int32).max for value in outputs):
        raise ValueError("Reference would overflow INT32")
    return [value.astype(np.int32) for value in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--hlo", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Use a fresh report path")
    program = read_program(args.assembly)
    joint = sum(name == "double_gemm" for name, _ in program.calls)
    if program.name == "two_gemm" and joint != 1:
        raise AssertionError("The minimal positive control must select one real DoubleGEMM")
    if program.name == "dependent_gemm" and joint != 0:
        raise AssertionError("The dependency control must not select a joint GEMM")
    module = xla_client._xla.hlo_module_from_text(args.hlo.read_text())
    computation = xla_client.XlaComputation(module.as_serialized_hlo_module_proto())
    mlir = xla_client._xla.mlir.xla_computation_to_mlir_module(computation)
    executable = jax.devices("cpu")[0].client.compile(mlir, compile_options=xla_client.CompileOptions())
    sys.path.insert(0, str((args.target / "targets/GemminiDual").resolve()))
    from oracle import api
    from oracle.decorator import kernel, set_simulation_backend
    set_simulation_backend("CPU")
    spec = importlib.util.spec_from_file_location("generated_gemmini", args.assembly)
    assembly_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly_module)
    launch = getattr(assembly_module, program.name)(kernel, api)
    launch("fsim-compile")()
    datasets = []
    for dataset in (901, 902, 903, 904, "identity", "zero"):
        inputs = make_inputs(len(program.metadata["input"]), dataset)
        expected = reference(program.name, inputs)
        hlo_outputs = executable.execute([jax.device_put(value, executable.local_devices()[0]) for value in inputs])
        oracle_outputs, _ = launch("fsim")(*inputs)
        address_outputs = run_address_model(program, inputs)
        for label, outputs in (("raw HLO", hlo_outputs), ("Oracle", oracle_outputs),
                               ("address model", address_outputs)):
            if len(outputs) != len(expected):
                raise AssertionError(f"{label}: output count mismatch")
            for index, (actual, golden) in enumerate(zip(outputs, expected)):
                actual = np.asarray(actual)
                if actual.dtype != np.int32 or actual.shape != golden.shape:
                    raise AssertionError(f"{label}: output type/shape mismatch")
                np.testing.assert_array_equal(actual, golden, err_msg=f"{label}/{dataset}/output{index}")
        if len(expected) > 1 and isinstance(dataset, int) and np.array_equal(expected[0], expected[1]):
            raise AssertionError("Random references must distinguish the two results")
        datasets.append({"dataset": dataset,
                         "input_sha256": hashlib.sha256(b"".join(value.tobytes() for value in inputs)).hexdigest(),
                         "outputs_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in expected]})
    report = {"functional_passed": True, "rtl_verified": False,
              "hlo_sha256": hashlib.sha256(args.hlo.read_bytes()).hexdigest(),
              "assembly_sha256": hashlib.sha256(args.assembly.read_bytes()).hexdigest(),
              "calls": [name for name, _ in program.calls], "datasets": datasets}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{program.name}: raw HLO / Oracle / address model match integer reference on 6 datasets")


if __name__ == "__main__":
    main()
