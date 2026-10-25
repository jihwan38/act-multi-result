# Multi-result matching experiment

This repository experiment compares three implementations without changing the
fixed upstream revisions or the meaning of an ISA declaration.

## Branch sequence

1. `main` at `10f059e` remains the reference implementation. Its `multi-result`
   patch stack is not edited by this experiment.
2. Improve `experiment/multipattern-discovery` (starting at `474b40a`) and commit
   it only after its implementation and regression/integration checks pass.
3. Create `experiment/multipattern-rewrite` from that improved commit. Add a
   selectable direct-rewrite toolchain so all three implementations can be
   rebuilt and measured from the final checkout. Preserve the second branch.

## Stage 1: bounded and explicit discovery

- Separate registered-candidate limits from raw matching work and elapsed-time
  budgets. Avoid collecting thousands of combinations before the first batch.
- Bound raw substitution materialization and repeated searches; count rejected,
  duplicate and accepted matches separately. Record truncation and stopping
  reasons. Never describe a result-count bound as a hard execution-time bound.
- Select anchors and rules by an explicit, stable policy for a fixed e-graph.
  Test limited searches and multiple instructions. Document that IDs and other
  ACT search stages can still vary across independently constructed graphs.
- Preserve shared input equalities, metadata wildcards, identity results,
  ordered/repeated results, dependency rejection and refresh after unions.
- Use an additional patch in `patches/multipattern/`, keeping `main`'s patch
  files unchanged. Implement changes in scratch sources, then regenerate the
  patch and prepared sources rather than editing generated targets in place.

## Stage 1 validation

- Run the existing Python, Rust, QKV, DoubleGEMM, dependency and allocation
  regression suites.
- Add focused tests for independent work/registration limits, expiration,
  deterministic selection, repeated batches and explicit truncation.
- Generate a real ISA with shared inputs, at least three outputs and multiple
  multi-result instructions. Exercise nested metadata where supported.
- Compile positive and negative HLO cases. Compare every final tensor with the
  original HLO through the Oracle on the same seeded and zero input datasets.
- Check result ports and allocation records as well as numeric outputs. A
  passing generic fallback alone is not evidence that a new joint rule works:
  include a case/assertion that actually selects the intended instruction.

## Stage 2: direct generated rewrite

- Keep single-result rewrites on their established path.
- Generate a MultiPattern-based Searcher and a dedicated multi-result Applier.
  Retain bindings for semantic-node occurrences and ordered output roots so
  validation does not need to rediscover a speculative tuple pattern.
- Reuse generated shape, metadata and input-feasibility checks against the graph
  at application time. Do not carry an earlier eligibility verdict across unions.
- Validate before graph mutation. For a valid match, add one ISA producer,
  its ordered aggregate tuple and its output projections; union each output with
  the correct projection. Never union one tensor output with the whole producer.
- Remove the discovery hook from the direct toolchain. Preserve the aggregate
  representation needed by extraction, PII construction and allocation.
- Make matching and application budgets explicit. Account for egg's scheduler;
  moving work into a Rewrite does not automatically solve match explosion,
  starvation or interruption of a running egg search call.
- Test invalid matches leaving no speculative aggregate, repeat application
  reaching saturation, post-search unions, identity and repeated results, three
  outputs, multiple competing rules and extraction of multiple ports from one
  producer. Run the full existing regression and new integration suites.

## Comparative experiment

- Use identical ISA/HLO fixtures, inputs, compiler settings, hardware and cost
  model across `multi-result`, `multipattern`, and `multipattern-rewrite`.
- Record branch/commit, upstream revisions, patch fingerprints, fixture hashes,
  generated/runtime source hashes and compiler hashes. Keep historical runs.
- Measure compiler execution separately from generation/build and Oracle
  verification. Include a warm-up and at least five measured repetitions,
  rotating implementation order to reduce ordering bias.
- Record wall time, peak process RSS, selected instruction calls and final cost,
  numeric/allocation verification, raw/accepted/applied matching statistics when
  supported, and every timeout/failure. Do not silently omit failed samples.
- Include small and scaled independent outputs, shared-input workloads,
  invalid-type/shape/metadata/dependency cases, three outputs and multiple rules.
- Compare normal defaults and a documented common resource envelope. Candidate
  limits with different meanings are not automatically equal work budgets.
- Use a compiler-process timeout as the hard external wall-time bound. Internal
  deadlines checked between egg calls are cooperative, not preemptive.
- Evaluate reproducibility and output quality, not just compile time. Preserve
  raw measurements and reproducible commands; report tradeoffs even if direct
  rewriting loses a comparison.

## Completion criteria

All intended branch states are committed and available; tracked work is clean;
the new implementations pass the applicable regression and end-to-end checks;
three-way experiments are complete with raw evidence and source provenance;
the report distinguishes measured results, inferred benefits and limitations.
The final response identifies the three branch commits, reproduction commands,
test evidence and the conditions under which each approach helps or hurts.
