# MultiPattern experiments

`ProbeISA.py` declares a small generated accelerator whose joint instructions
share real input variables. The same Python declaration, HLO inputs, Oracle
checks, and cost model are used for each selected toolchain.

| Instruction | Semantics |
| --- | --- |
| `shared_pair` | `add(x, a)`, `multiply(x, b)` |
| `shared_triple` | `add(x, a)`, `multiply(x, b)`, `dot(x, c)` |
| `reshape_pair` | `reshape(x, [2,16])`, `reshape(x, [1,32])`, with `x:[4,8]` |

Arithmetic tensors have shape `bf16[8,8]`. Each operation also has a scalar
instruction alternative. Loads and stores use the tutorial's byte-to-BF16
conversion convention with narrower rows. The three joint rules coexist, and
`mixed_pair_triple` requires both arithmetic joint instructions in its selected
assembly. The concat workloads have no explicit HLO tuple, so they exercise
automatic joint discovery. Other workloads cover ordered tuple outputs,
repeated inputs, different inputs, a result used as another result's external
operand, metadata-bearing operations, and 4/16 independent input groups.

## Build and check

Activate the project's `act` Conda environment first. Choose a fresh output
directory for a changed fixture or changed source plan:

```bash
conda activate act
python -B experiments/multipattern/run.py build \
  --output build/experiments/multipattern/run1 \
  --toolchains multi-result multipattern

python -B experiments/multipattern/run.py check \
  --output build/experiments/multipattern/run1 \
  --label correctness \
  --toolchains multi-result multipattern
```

The build phase copies the prepared sources and applicable patches before
generating each compiler and Oracle. It records the Git commit, source-plan
fingerprint, source hashes, generated-source hashes, runtime hashes, and binary
hashes. An existing unchanged build can be reused, and a new configured
toolchain can be added to the same experiment. A changed existing build is
rejected. Each check or measurement label must be new.

`check` executes all 14 HLO workloads by default. `--cases` selects named
workloads, for example `shared_pair_concat shared_triple_concat shared_reshape`.
Each produced assembly is compared with the unchanged HLO through the generated
Oracle on `seed0`, `seed1`, `seed2`, and zero inputs. The allocation audit checks
the selected PII, output ports, addresses, and lifetimes. Small positive cases
also require the intended joint instruction in the selected assembly. Negative
cases require correctness; the presence of a joint instruction alone does not
prove that unrelated outputs were incorrectly combined.

To examine fallback under a tight budget without treating a correct scalar
assembly as a failed correctness check:

```bash
python -B experiments/multipattern/run.py check \
  --output build/experiments/multipattern/run1 \
  --label zero-discovery --joint-limit 0 --allow-fallback
```

The standard limits are `--joint-limit 64`, `--raw-limit 4096`,
`--search-limit 4096`, and `--attempt-limit 4096`. They are passed explicitly as
`ACT_JOINT_LIMIT`, `ACT_JOINT_RAW_LIMIT`, `ACT_JOINT_SEARCH_LIMIT`, and
`ACT_JOINT_ATTEMPT_LIMIT`. The unmodified `multi-result` reference retains its
own cap-derived attempt policy and does not implement the three new controls.

## Compiler measurements

Run measurements serially after other builds and tests have finished:

```bash
python -B experiments/multipattern/run.py measure \
  --output build/experiments/multipattern/run1 \
  --label measured --repeats 5 --warmups 1
```

The runner rotates toolchain order each repetition and preserves warmups and
all failed samples. At least five measured repetitions and one warmup are
required. Compilation has a process-group timeout (`--timeout`, default 35 s);
verification has its own timeout and is outside the measured compiler interval.

Metrics have different meanings:

- `wall_seconds`: elapsed compiler process time, including process launch and
  excluding generation and Oracle verification. `communicate(timeout)` waits
  for output/exit without polling the child's status at fixed intervals.
- `gnu_time_max_rss_kib`: GNU time's compiler process maximum RSS.
- `model_cost`: the current ACT cost model's assembly line count. This is not
  simulated hardware latency. API calls and selected PII are recorded separately.
- `correctness_pass`: Oracle exactness, allocation audit, successful compiler
  completion, and unchanged runtime files.
- `selection_expectations_pass`: whether the small positive workload selected
  its expected joint instruction. Measurements report missed selection as a
  quality result even when scalar fallback is correct.
- `discovery_reports`: each `ACT_DISCOVERY` record emitted by a toolchain that
  supports it. `raw_matches` includes returned prefix replays;
  `replayed_matches` is recorded separately. These are work counts, not counts
  of distinct semantic matches. The reference has no such report.

`samples.json` retains individual runs; `summary.json` contains counts and
median/min/max metrics. Inspect correctness and selection rates before comparing
timings. A run that searches less or returns fewer candidates may be faster
without finding an equally good assembly.

## Controlled graph measurements

`IndependentISA.py` reuses the scalar/storage definitions and replaces the
joint rules with unconstrained pair/triple rules. It is kept separate so those
rules cannot obscure the shared-variable experiment. Build it into a separate
experiment directory:

```bash
python -B experiments/multipattern/run.py build \
  --output build/experiments/multipattern/independent1 \
  --fixture independent --toolchains multi-result multipattern

python -B experiments/multipattern/microbench.py build \
  --output build/experiments/multipattern/micro1 \
  --shared-build build/experiments/multipattern/run1 \
  --independent-build build/experiments/multipattern/independent1 \
  --toolchains multi-result multipattern

python -B experiments/multipattern/microbench.py check \
  --output build/experiments/multipattern/micro1 --label correctness

python -B experiments/multipattern/microbench.py measure \
  --output build/experiments/multipattern/micro1 \
  --label measured --repeats 5 --warmups 1
```

The Rust harness uses the actual generated patterns and rule implementations.
It measures one joint-candidate pass: discovery plus generated rule search/apply
for configurations with a discovery hook, or generated rule search/apply directly
for a direct-rule configuration. It records these components separately. It
does not run IR rewrite iterations, extraction, allocation, or the Oracle.

Shared and disjoint-input cases use 8/32/128 input groups. Additional graph
checks cover dependencies, invalid shapes, and reshape metadata. The independent
fixture has `N²` legal pairs and `N³` legal triples; the shared fixture has only
`N` of each. Assertions check result order and input-group consistency in the
registered tuples. Internal timings exclude graph construction; external wall
and RSS include graph construction and runtime overhead. Search report
completeness describes that bounded snapshot, not global compiler saturation.

All generated files and results belong under ignored `build/` directories.
Keep each experiment's frozen build directories in place: the compiled backend
references generated rewrite/cost files at those locations. Reuse checks verify
their hashes before running. Moving only an executable does not preserve its
runtime environment.
