"""Generate the ordinary patched ACT Backend and functional Oracle for Gemmini."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from GemminiDual import make_target

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
BUILD = ROOT / "build/multi-result-legacy"
sys.path.insert(0, str(ROOT / "scripts"))
from _internal.common import fingerprint, source_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    build = args.build.resolve()
    if build != BUILD:
        raise ValueError(f"Generate this profile in {BUILD}")

    from act.generators import generate_oracle
    from act_backend import generate_backend
    import act_backend
    import taidl_to

    sources = BUILD / "src"
    provenance = json.loads((sources / "provenance.json").read_text())
    if provenance["fingerprint"] != fingerprint(source_plan("multi-result-legacy")):
        raise ValueError("Run prepare_sources.py multi-result-legacy first")
    for module in (act_backend, taidl_to):
        if not Path(module.__file__).resolve().is_relative_to(sources):
            raise ValueError(f"Wrong ACT profile: {module.__file__}")
    build.mkdir(parents=True, exist_ok=True)
    # A failed regeneration must not leave a success manifest from an older build.
    (build / "generation.json").unlink(missing_ok=True)
    target = make_target(build)
    try:
        generate_oracle(target)
        generate_backend(target)
    except subprocess.CalledProcessError as error:
        # The upstream facade captures compiler diagnostics. Preserve them on
        # failure rather than reporting only a nonzero Cargo/CMake status.
        for output in (error.stdout, error.stderr):
            if output:
                print(output.decode() if isinstance(output, bytes) else output, file=sys.stderr)
        raise

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "spec_sha256": digest(experiment / "GemminiDual.py"),
        "backend_binary_sha256": digest(build / "backends/GemminiDual"),
        "toolchain": "multi-result-legacy",
        "source_fingerprint": provenance["fingerprint"],
        "sources": {key: provenance[key] for key in ("upstreams", "patches")},
        "workloads": {
            path.name: digest(path) for path in sorted((experiment / "workloads").glob("*.hlo"))
        },
        "note": "Generation only. Neither RTL execution nor performance validation.",
    }
    snapshot = build / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "GemminiDual.py").write_bytes((experiment / "GemminiDual.py").read_bytes())
    (build / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated GemminiDual at {build}")


if __name__ == "__main__":
    main()
