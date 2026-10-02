#!/usr/bin/env python3
"""Execute an ACT assembly with its Oracle and compare it with the original HLO."""
import argparse
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import re
import runpy
import sys

from _internal.assembly import parse
from _internal.common import CASES, COMPONENTS, ROOT, TOOLCHAINS, digest, main_guard, write_json


def example_defaults(assembly):
    """Resolve only our documented output layout; never guess from op names."""
    for mode in TOOLCHAINS:
        base = ROOT / "build" / mode
        try:
            parts = assembly.relative_to(base / "runs").parts
        except ValueError:
            continue
        if len(parts) < 3 or parts[0] not in CASES:
            break
        case = parts[0]
        config = CASES[case]
        if mode not in config["toolchains"]:
            break
        oracle = base / "targets" / config["target"] / "oracle"
        isa = ROOT / "cases" / case / config["isa"] if mode == "multi-result" else None
        return oracle, isa
    return None, None


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlo", "--workload", dest="hlo", type=Path, required=True, help="Original workload HLO")
    parser.add_argument("--assembly", type=Path, required=True, help="Compiled Python assembly")
    parser.add_argument("--oracle", type=Path, help="Generated targets/<target>/oracle directory; inferred for the example run paths")
    parser.add_argument("--isa", type=Path, help="ISA Python declaration for an allocation audit; inferred for the multi-result run paths")
    parser.add_argument("--pii", type=Path, help="Selected PII for the allocation audit; otherwise found in compiler/")
    parser.add_argument("--report", type=Path, help="Report path; defaults to verification.json beside the assembly")
    args = parser.parse_args(argv)
    args.hlo, args.assembly = args.hlo.resolve(), args.assembly.resolve()
    default_oracle, default_isa = example_defaults(args.assembly)
    args.oracle = args.oracle or default_oracle
    if args.oracle:
        args.oracle = args.oracle.resolve()
    args.isa = args.isa or default_isa
    if args.isa:
        args.isa = args.isa.resolve()
    if args.pii:
        args.pii = args.pii.resolve()
    elif args.isa and args.assembly.is_file():
        try:
            args.pii = selected_pii(args)
        except (ValueError, OSError):
            # The audit will report this after the old report has been cleared.
            pass
    args.report = (args.report or args.assembly.parent / "verification.json").resolve()
    # Protect inputs before removing a previous report, including the audit sidecar.
    for path in verification_inputs(args):
        if path is not None and path.resolve() == args.report:
            raise ValueError("The verification report must not overwrite an input file")
    return args


def verification_inputs(args):
    paths = [args.hlo, args.assembly, args.isa, args.pii]
    if args.oracle:
        paths.extend((args.oracle / "api.py", args.oracle / "decorator.py"))
    if args.isa:
        paths.append(Path(str(args.assembly) + ".allocation.json"))
    return paths


def validate_arguments(args):
    if args.oracle is None:
        raise ValueError("Cannot infer the ISA from this assembly path; pass --oracle /path/to/targets/<target>/oracle")
    if args.pii and not args.isa:
        raise ValueError("--pii requires an ISA declaration; pass --isa /path/to/ISA.py")
    for path in verification_inputs(args):
        if path is not None and not path.is_file():
            raise ValueError(f"Missing input: {path}")
    if args.oracle.name != "oracle":
        raise ValueError("--oracle must name the generated oracle/ package directory")


def load_assembly(path):
    _, _, metadata, calls = parse(path.read_text())
    spec = importlib.util.spec_from_file_location("selected_assembly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, metadata["module_name"]), metadata, calls


def hlo_reference(path, inputs):
    import jax
    import jax.numpy as jnp
    from jaxlib import xla_client
    import numpy as np

    module = xla_client._xla.hlo_module_from_text(path.read_text())
    computation = xla_client.XlaComputation(module.as_serialized_hlo_module_proto())
    parameters = computation.program_shape().parameter_shapes()
    if len(parameters) != len(inputs):
        raise ValueError("Assembly input count differs from the HLO; this example expects flat tensor inputs")
    for shape, item in zip(parameters, inputs):
        if (shape.is_tuple() or tuple(shape.dimensions()) != tuple(item["shape"])
                or shape.numpy_dtype() != np.dtype(getattr(jnp, item["dtype"].removeprefix("jnp.")))):
            raise ValueError("Assembly input shape/dtype differs from the HLO")
    mlir = xla_client._xla.mlir.xla_computation_to_mlir_module(computation)
    return jax.devices("cpu")[0].client.compile(mlir, compile_options=xla_client.CompileOptions())


def input_sets(inputs):
    import jax.numpy as jnp
    import numpy as np

    for dataset in ("seed0", "seed1", "seed2", "zero"):
        values = []
        for index, item in enumerate(inputs):
            dtype = getattr(jnp, item["dtype"].removeprefix("jnp."))
            if dataset == "zero":
                value = jnp.zeros(item["shape"], dtype=dtype)
            else:
                rng = np.random.default_rng(int(dataset[4:]) * 1000 + index)
                if np.issubdtype(np.dtype(dtype), np.integer):
                    low = 0 if np.issubdtype(np.dtype(dtype), np.unsignedinteger) else -9
                    sample = rng.integers(low, 10, size=item["shape"])
                elif dtype == jnp.bool_:
                    sample = rng.integers(0, 2, size=item["shape"])
                else:
                    sample = rng.integers(-9, 10, size=item["shape"]) / 32
                value = jnp.asarray(sample, dtype=dtype)
            values.append(value)
        yield dataset, values


def compare_outputs(dataset, actual, expected):
    import numpy as np

    if len(actual) != len(expected):
        raise ValueError("Assembly output count differs from the HLO")
    outputs = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        a, b = np.asarray(a), np.asarray(b)
        same_shape = a.shape == b.shape
        exact = same_shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        finite = bool(np.isfinite(a.astype(np.float32)).all() and np.isfinite(b.astype(np.float32)).all())
        difference = (float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)), initial=0))
                      if same_shape and finite else None)
        outputs.append({"index": index, "actual_shape": list(a.shape), "expected_shape": list(b.shape),
                        "actual_dtype": str(a.dtype), "expected_dtype": str(b.dtype),
                        "exact_bytes": bool(exact), "finite": finite, "max_abs_diff": difference})
    return {"dataset": dataset, "outputs": outputs}


def selected_pii(args):
    if args.pii is not None:
        return args.pii
    match = re.search(r"^# PII number: (\d+)$", args.assembly.read_text(), re.M)
    if not match:
        raise ValueError("Assembly does not identify its selected PII; pass --pii")
    for path in (args.assembly.parent / "compiler" / f"{match[1]}.pii",
                 args.assembly.parent / f"{match[1]}.pii"):
        if path.is_file():
            return path
    raise ValueError("Cannot find the selected PII; pass --pii from the compiler's --log directory")


def save_report(args, calls, checks, allocation=None):
    passed = all(item["exact_bytes"] and item["finite"] for check in checks for item in check["outputs"])
    report = {"passed": passed, "datasets": len(checks), "checks": checks,
              "reference": "unchanged workload HLO, executed with XLA CPU",
              "assembly_sha256": digest(args.assembly), "hlo_sha256": digest(args.hlo),
              "api_calls": dict(Counter(call.op for call in calls)), "allocation": allocation,
              "oracle": str(args.oracle), "isa": str(args.isa) if args.isa else None}
    write_json(args.report, report)
    if not passed:
        raise ValueError(f"Oracle/HLO results differ; see {args.report}")
    print(f"PASS: {len(checks)} input sets matched exactly. Report: {args.report}")


def allocation_audit(args, target_name):
    if args.isa is None:
        return None
    # Read the declaration using the same ACT sources as this generated target.
    # run_path does not execute its `if __name__ == "__main__"` generators.
    sources = args.oracle.parents[2] / "src"
    if all((sources / name).is_dir() for name in COMPONENTS):
        sys.path[:0] = [str(sources / name) for name in COMPONENTS]
    from act.taidl import Accelerator
    from _internal.audit import audit
    from _internal.isa_ports import describe

    declared = runpy.run_path(str(args.isa))
    targets = {id(value): value for value in declared.values() if isinstance(value, Accelerator)}
    if len(targets) != 1:
        raise ValueError("--isa must declare exactly one Accelerator")
    target = next(iter(targets.values()))
    if target.name != target_name:
        raise ValueError(f"ISA {target.name} does not match Oracle {target_name}")
    return audit(selected_pii(args).read_text(),
                 json.loads(Path(str(args.assembly) + ".allocation.json").read_text()),
                 args.assembly.read_text(), describe(target))


def verify(args):
    import jax

    sys.path.insert(0, str(args.oracle.parent))
    from oracle import api
    from oracle.decorator import kernel, set_simulation_backend
    if not Path(api.__file__).resolve().is_relative_to(args.oracle):
        raise ValueError("Imported the wrong generated Oracle")
    print(f"Oracle: {args.oracle}", flush=True)
    set_simulation_backend("CPU")
    factory, metadata, calls = load_assembly(args.assembly)
    for call in calls:
        operation = getattr(api, call.op, None)
        if not callable(operation):
            raise ValueError(f"Oracle {api.TARGET_NAME} has no instruction {call.op}")
        try:
            inspect.signature(operation).bind(**call.params)
        except TypeError as error:
            raise ValueError(f"Assembly/Oracle arguments differ for {call.op}: {error}") from None
    reference = hlo_reference(args.hlo, metadata["input"])
    allocation = allocation_audit(args, api.TARGET_NAME)

    program = factory(kernel, api)
    program("fsim-compile")()
    checks = []
    for dataset, inputs in input_sets(metadata["input"]):
        expected = reference.execute([jax.device_put(value, reference.local_devices()[0]) for value in inputs])
        outputs, elapsed = program("fsim")(*inputs)
        checks.append(compare_outputs(dataset, outputs, expected))
    save_report(args, calls, checks, allocation)


def main():
    args = arguments()
    # A failed new attempt must not leave an old PASS report in place.
    args.report.unlink(missing_ok=True)
    try:
        validate_arguments(args)
        verify(args)
    except Exception as error:
        if not args.report.exists():
            write_json(args.report, {"passed": False, "error": str(error),
                                    "hlo": str(args.hlo), "assembly": str(args.assembly),
                                    "oracle": str(args.oracle) if args.oracle else None})
        raise


if __name__ == "__main__":
    main_guard(main)
