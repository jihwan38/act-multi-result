# Discovery checkpoint

Historical checkpoint at `05692e4`, preserved by the tag
`comparison-20261026-multipattern`. The subsequent direct-rewrite implementation
and completed three-way comparison are recorded in [REPORT.md](REPORT.md).
The validation counts and workspace paths below describe this checkpoint.

The first implementation checkpoint keeps the generated tuple-based ISA rewrite
path and changes only the optional `multipattern` discovery configuration.
`main`'s baseline and multi-result patches remain byte-for-byte unchanged.

The new patch separates the cumulative tuple registration cap from per-refresh
raw-substitution, search-call and candidate-attempt budgets. Prefix replay counts
against the raw budget. Each ISA receives one search quantum in declaration
order, with anchors visited by increasing e-class ID. Search preparation stops
when enough feasible, distinct candidates are queued. Canonicalization may merge
previous tuples, but it cannot replenish the cumulative registration allowance.

The policy is deterministic for the same graph snapshot and IDs. It is not a
claim that all ACT phases or independently built graphs are deterministic.
An egg call cannot be interrupted mid-call; the new counters bound materialized
results and calls, not the VM's internal work. External experiment timeouts bound
whole compiler processes. Truncated searches are explicitly reported.

## Validation performed before branching

- Reference `multi-result`: all 10 existing test groups passed.
- Improved `multipattern`: all 10 existing test groups passed, including 51
  Python tests and Rust executions for QKV (48) and DoubleGEMM (39). The Rust
  totals include the same generic library tests compiled for both targets.
- Eleven new library tests cover independent budgets, pagination/replay,
  deadlines, deterministic selection, competing rules, three outputs and the
  cumulative cap after e-class unions.
- Both reference and improved compilers passed all 14 generated Probe ISA
  workloads. These include shared pair/triple instructions, concatenated outputs
  without an explicit workload tuple, repeated operands, direct dependencies,
  different-input negatives, shared reshape metadata, competing pair/triple
  instructions and workloads with 4/16 groups.
- Every successful Probe run was checked against HLO execution on seed0, seed1,
  seed2 and zero inputs, including the allocation audit. Positive small cases
  were required to select their intended multi-result instructions.
- Both implementations also passed 12 controlled-graph microbenchmark checks
  using the shared and unconstrained pair/triple ISAs. These check candidate
  semantics independently of extraction and allocation; timings are deferred
  to the repeated final measurements.

Evidence in the workspace:

- `build/goal-multipattern/reference/`: frozen reference provenance and full
  regression logs.
- `build/goal-multipattern/stage1/full-regression/`: improved full regression.
- `build/goal-multipattern/probe/builds.json`: frozen source, fixture, compiler
  and runtime hashes for the two generated Probe implementations.
- `build/goal-multipattern/probe/checks/baseline/`: 14/14 reference checks.
- `build/goal-multipattern/probe/checks/stage1/`: 14/14 improved checks.
- `build/goal-multipattern/micro/checks/main-smoke/` and `stage1-smoke/`:
  12/12 controlled-graph checks for each implementation.

These functional checks ran while development/build activity was ongoing; their
elapsed times are not used to claim a performance improvement. The final
comparison uses separate repeated measurements after all three implementations
are ready. Several scaled workloads reach ACT's existing ten-second compilation
budget while still returning correct assemblies, so final code cost and selected
joint instructions must be compared alongside elapsed time.
