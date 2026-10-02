//! Generated typed joint rules, refreshed after real rewrites and e-class unions.
//! Pair registration is distinct from final physical instruction selection.
use backend::ir::{
    dtype::Dtype,
    egraph::{TensorInfo, TensorOp},
    metadata::OutputRoot,
};
use backend::isel::{discovery::Discovery, extractor, rewrites};
use egg::{Applier, EGraph, Id, Pattern, PatternAst, Rewrite, Runner, Subst, Symbol};
use std::collections::HashSet;
use std::time::{Duration, Instant};

type Graph = EGraph<TensorOp, TensorInfo>;

fn tensor(g: &mut Graph, op: TensorOp, shape: Vec<i32>) -> Id {
    let id = g.add(op);
    g.set_analysis_data(
        id,
        TensorInfo {
            dtype: Dtype::BF16,
            shape,
            is_const: false,
        },
    );
    id
}

fn matrix(g: &mut Graph, op: TensorOp) -> Id {
    tensor(g, op, vec![64, 64])
}

fn variable(g: &mut Graph, name: &str) -> Id {
    matrix(g, TensorOp::Var(name.to_owned()))
}

fn drain(discovery: &mut Discovery, g: &mut Graph) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut count = 0;
    while !discovery.exhausted() {
        assert!(
            Instant::now() < deadline,
            "discovery failed to finish its bounded snapshot"
        );
        count += discovery.batch(g, 64, deadline);
    }
    count
}

fn has_pair(g: &Graph, a: Id, b: Id) -> bool {
    g.lookup(TensorOp::OpTuple(vec![g.find(a), g.find(b)]))
        .is_some()
}

fn pair(dependent: bool, indirect: bool) -> (Graph, [Id; 4], [Id; 2]) {
    let mut g = Graph::default();
    let args = ["a", "b", "c", "d"].map(|name| variable(&mut g, name));
    let a = matrix(&mut g, TensorOp::OpDot([args[0], args[1]]));
    let input = if !dependent {
        args[2]
    } else if indirect {
        matrix(&mut g, TensorOp::OpCopy([a]))
    } else {
        a
    };
    let b = matrix(&mut g, TensorOp::OpDot([input, args[3]]));
    g.rebuild();
    (g, args, [a, b])
}

/// Deliberately omit Discovery: an explicit workload tuple must be checked by
/// the generated ISA rule itself, even with automatic discovery disabled.
fn rewrite_explicit_tuple(g: Graph) -> Graph {
    backend::PROCESSED.lock().unwrap().clear();
    let rules: Vec<_> = rewrites::get_rewrites()
        .into_iter()
        .filter(|r| r.name.to_string() == "double-gemm")
        .collect();
    assert_eq!(rules.len(), 1);
    Runner::default()
        .with_egraph(g)
        .with_iter_limit(4)
        .with_hook(|_| {
            backend::PROCESSED.lock().unwrap().clear();
            Ok(())
        })
        .run(&rules)
        .egraph
}

fn has_joint(g: &Graph, tuple: Id) -> bool {
    g[g.find(tuple)]
        .nodes
        .iter()
        .any(|op| matches!(op, TensorOp::DoubleGemm(_)))
}

#[test]
fn explicit_dependent_tuples_cannot_bypass_the_generated_rule() {
    for indirect in [false, true] {
        for reversed in [false, true] {
            let (mut g, _, [a, b]) = pair(true, indirect);
            let children = if reversed { vec![b, a] } else { vec![a, b] };
            let tuple = g.add(TensorOp::OpTuple(children));
            g.rebuild();
            let g = rewrite_explicit_tuple(g);
            assert!(
                !has_joint(&g, tuple),
                "indirect={indirect}, reversed={reversed}"
            );
            assert!(
                g[g.find(tuple)]
                    .nodes
                    .iter()
                    .any(|op| matches!(op, TensorOp::OpTuple(_))),
                "a dependent tuple is still a valid value"
            );
        }
    }
}

#[test]
fn explicit_independent_shared_input_and_repeated_result_tuples_still_match() {
    for mode in 0..3 {
        let (mut g, args, [a, b]) = pair(false, false);
        let second = match mode {
            0 => b,
            1 => matrix(&mut g, TensorOp::OpDot([args[0], args[3]])),
            _ => a,
        };
        let tuple = g.add(TensorOp::OpTuple(vec![a, second]));
        g.rebuild();
        let g = rewrite_explicit_tuple(g);
        assert!(has_joint(&g, tuple), "mode={mode}");
    }
}

#[test]
fn one_valid_match_does_not_authorize_dependent_alternatives_of_the_same_tuple() {
    let (mut g, args, [a, b]) = pair(false, false);
    let dependent = matrix(&mut g, TensorOp::OpDot([a, args[3]]));
    // Synthetic equality exposes two matches under the same output e-class.
    g.union(b, dependent);
    let tuple = g.add(TensorOp::OpTuple(vec![a, b]));
    g.rebuild();
    let g = rewrite_explicit_tuple(g);
    let joint_inputs: Vec<_> = g[g.find(tuple)]
        .nodes
        .iter()
        .filter_map(|op| {
            if let TensorOp::DoubleGemm(ids) = op {
                Some(*ids)
            } else {
                None
            }
        })
        .collect();
    assert!(joint_inputs
        .iter()
        .any(|ids| g.find(ids[2]) == g.find(args[2])));
    assert!(
        !joint_inputs.iter().any(|ids| g.find(ids[2]) == g.find(a)),
        "eligibility belongs to this semantic match, not the whole tuple e-class"
    );
}

#[test]
fn explicit_tuple_match_is_rechecked_after_an_independent_derivation_appears() {
    let (mut g, _, [a, b]) = pair(true, true);
    let tuple = g.add(TensorOp::OpTuple(vec![a, b]));
    g.rebuild();
    let mut g = rewrite_explicit_tuple(g);
    assert!(!has_joint(&g, tuple));
    let operand = g[g.find(b)]
        .nodes
        .iter()
        .find_map(|op| {
            if let TensorOp::OpDot([input, _]) = op {
                Some(*input)
            } else {
                None
            }
        })
        .unwrap();
    let independent = variable(&mut g, "independent_operand");
    g.union(operand, independent);
    g.rebuild();
    let g = rewrite_explicit_tuple(g);
    assert!(has_joint(&g, tuple));
}

#[test]
fn explicit_tuple_cannot_use_a_leafless_operand_cycle() {
    let (mut g, args, [a, b]) = pair(false, false);
    let copy = matrix(&mut g, TensorOp::OpCopy([args[0]]));
    g.union(copy, args[0]);
    g.rebuild();
    let id = g.find(args[0]);
    g[id].nodes.retain(|op| !matches!(op, TensorOp::Var(_)));
    let tuple = g.add(TensorOp::OpTuple(vec![a, b]));
    g.rebuild();
    let g = rewrite_explicit_tuple(g);
    assert!(!has_joint(&g, tuple));
}

/// Test-only row decomposition of a 128x64 product into two 64x64 products.
/// The generated DoubleGemm semantics cannot match either product before this.
#[derive(Clone)]
struct SplitRows;

impl Applier<TensorOp, TensorInfo> for SplitRows {
    fn apply_one(
        &self,
        g: &mut Graph,
        id: Id,
        subst: &Subst,
        _: Option<&PatternAst<TensorOp>>,
        _: Symbol,
    ) -> Vec<Id> {
        if g[id].data.shape != vec![128, 64] {
            return vec![];
        }
        let a = subst["?a".parse().unwrap()];
        let b = subst["?b".parse().unwrap()];
        let top = matrix(g, TensorOp::OpSlice("0:64,0:64".into(), [a]));
        let bottom = matrix(g, TensorOp::OpSlice("64:128,0:64".into(), [a]));
        let y0 = matrix(g, TensorOp::OpDot([top, b]));
        let y1 = matrix(g, TensorOp::OpDot([bottom, b]));
        let joined = tensor(g, TensorOp::OpConcat("0".into(), [y0, y1]), vec![128, 64]);
        if g.union(id, joined) {
            vec![joined]
        } else {
            vec![]
        }
    }
}

#[test]
fn hook_finds_a_joint_candidate_created_only_after_rewriting() {
    backend::PROCESSED.lock().unwrap().clear();
    let mut g = Graph::default();
    let a = tensor(&mut g, TensorOp::Var("a".into()), vec![128, 64]);
    let b = variable(&mut g, "b");
    tensor(&mut g, TensorOp::OpDot([a, b]), vec![128, 64]);
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    assert_eq!(drain(&mut discovery, &mut g), 0);
    let split = Rewrite::new(
        "test-split-rows",
        "(dot ?a ?b)".parse::<Pattern<TensorOp>>().unwrap(),
        SplitRows,
    )
    .unwrap();
    let mut rules: Vec<_> = rewrites::get_rewrites()
        .into_iter()
        .filter(|r| r.name.to_string() == "double-gemm")
        .collect();
    assert_eq!(rules.len(), 1);
    rules.push(split);
    let runner = Runner::default()
        .with_egraph(g)
        .with_iter_limit(8)
        .with_hook(move |runner| {
            backend::PROCESSED.lock().unwrap().clear();
            discovery.refresh(&runner.egraph); // Reproduction removes only this line.
            drain(&mut discovery, &mut runner.egraph);
            Ok(())
        })
        .run(&rules);
    assert!(matches!(
        runner.stop_reason,
        Some(egg::StopReason::Saturated)
    ));
    assert!(
        runner.egraph.classes().any(|class| {
            class.nodes.iter().any(|op|
        matches!(op, TensorOp::DoubleGemm(args) if args[0] != args[2] && args[1] == args[3]))
        }),
        "the two new row products must reach the real generated joint rewrite"
    );
}

#[test]
fn direct_and_transitive_dependencies_stay_rejected_after_refresh() {
    for indirect in [false, true] {
        let (mut g, _, [a, b]) = pair(true, indirect);
        let mut discovery = Discovery::new(&g, 64);
        drain(&mut discovery, &mut g);
        assert!(!has_pair(&g, a, b) && !has_pair(&g, b, a));
        discovery.refresh(&g);
        drain(&mut discovery, &mut g);
        assert!(!has_pair(&g, a, b) && !has_pair(&g, b, a));
    }
}

#[test]
fn shared_inputs_and_copy_projection_cycles_do_not_block_independent_results() {
    let (mut g, args, [a, _]) = pair(false, false);
    let b = matrix(&mut g, TensorOp::OpDot([args[0], args[3]]));
    let copy = matrix(&mut g, TensorOp::OpCopy([args[0]]));
    g.union(copy, args[0]);
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    drain(&mut discovery, &mut g);
    assert!(has_pair(&g, a, b));
    let size = g.total_size();
    discovery.refresh(&g);
    assert_eq!(drain(&mut discovery, &mut g), 0);
    assert_eq!(
        g.total_size(),
        size,
        "own tuple/GTE links must not create fresh candidates"
    );
}

#[test]
fn canonical_tuple_cache_survives_eclass_merges() {
    let (mut g, args, [a, b]) = pair(false, false);
    let copy = matrix(&mut g, TensorOp::OpCopy([args[0]]));
    let alias = matrix(&mut g, TensorOp::OpDot([copy, args[1]]));
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    drain(&mut discovery, &mut g);
    assert!(has_pair(&g, a, b) && has_pair(&g, alias, b));
    // copy(a) == a makes two previously distinct products congruent.
    g.union(copy, args[0]);
    g.rebuild();
    assert_eq!(g.find(a), g.find(alias));
    let size = g.total_size();
    discovery.refresh(&g);
    assert_eq!(drain(&mut discovery, &mut g), 0);
    assert_eq!(g.total_size(), size);
    assert!(has_pair(&g, a, b));
}

#[test]
fn rejected_match_is_reconsidered_when_an_operand_gets_an_independent_derivation() {
    let (mut g, _, [a, b]) = pair(true, true);
    let independent = variable(&mut g, "equivalent_input");
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    drain(&mut discovery, &mut g);
    assert!(!has_pair(&g, a, b));
    // Model a valid equality supplied by an IR rewrite. The old dependent
    // alternative remains in the class: union-of-ancestors would still reject.
    let operand = g[b]
        .nodes
        .iter()
        .find_map(|op| match op {
            TensorOp::OpDot([x, _]) => Some(*x),
            _ => None,
        })
        .unwrap();
    g.union(independent, operand);
    g.rebuild();
    discovery.refresh(&g);
    drain(&mut discovery, &mut g);
    assert!(has_pair(&g, a, b));
}

#[test]
fn a_cycle_without_a_leaf_cannot_supply_an_instruction_operand() {
    let (mut g, args, [a, b]) = pair(false, false);
    let copy = matrix(&mut g, TensorOp::OpCopy([args[0]]));
    g.union(copy, args[0]);
    g.rebuild();
    // Deliberately leave only x = copy(x), with no input/constant escape.
    // This synthetic unusable e-class must never certify independence.
    let id = g.find(args[0]);
    g[id].nodes.retain(|op| !matches!(op, TensorOp::Var(_)));
    let mut discovery = Discovery::new(&g, 64);
    drain(&mut discovery, &mut g);
    assert!(!has_pair(&g, a, b));
}

#[test]
fn projections_follow_only_the_selected_element_for_dependency_checks() {
    for index in [0, 1] {
        let (mut g, args, [a, _]) = pair(false, false);
        let tuple = g.add(TensorOp::OpTuple(vec![a, args[2]]));
        let input = matrix(&mut g, TensorOp::OpGetTupleElement(index, [tuple]));
        let b = matrix(&mut g, TensorOp::OpDot([input, args[3]]));
        g.rebuild();
        let mut discovery = Discovery::new(&g, 64);
        drain(&mut discovery, &mut g);
        assert_eq!(has_pair(&g, a, b), index == 1);
    }
}

#[test]
fn identity_result_equal_to_an_input_does_not_create_a_false_dependency() {
    let mut g = Graph::default();
    let a = variable(&mut g, "a");
    let b = variable(&mut g, "b");
    let identity = matrix(&mut g, TensorOp::OpEye("64,64,BF16".into()));
    let first = matrix(&mut g, TensorOp::OpDot([a, identity]));
    let second = matrix(&mut g, TensorOp::OpDot([a, b]));
    // Model the identity equality: A*I == A. A is already an input; the
    // second product need not wait for an instruction that reproduces A.
    g.union(a, first);
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    drain(&mut discovery, &mut g);
    assert!(has_pair(&g, first, second));
}

#[test]
fn repeated_refresh_respects_zero_and_global_candidate_limits_and_deadlines() {
    for limit in [0, 1] {
        let (mut g, _, _) = pair(false, false);
        let mut discovery = Discovery::new(&g, limit);
        let mut added = drain(&mut discovery, &mut g);
        for _ in 0..3 {
            discovery.refresh(&g);
            added += drain(&mut discovery, &mut g);
        }
        assert_eq!(added, limit);
    }
    let (mut g, _, _) = pair(false, false);
    let mut discovery = Discovery::new(&g, 64);
    let size = g.total_size();
    assert_eq!(discovery.batch(&mut g, 64, Instant::now()), 0);
    assert_eq!(g.total_size(), size);
}

#[test]
fn refreshed_candidate_reaches_fast_extraction_as_one_producer_with_two_ports() {
    let mut g = Graph::default();
    let mut inputs = HashSet::new();
    let mut offsets = vec![];
    let mut args = vec![];
    for i in 0..4 {
        let input = g.add(TensorOp::Var(format!("p{i}")));
        g[input].data = TensorInfo {
            dtype: Dtype::U8,
            shape: vec![8192],
            is_const: false,
        };
        inputs.insert(input);
        offsets.push((Some(input), i * 8192));
        args.push(matrix(
            &mut g,
            if i < 2 {
                TensorOp::LoadRm0("64".into(), [input])
            } else {
                TensorOp::LoadRm1("64".into(), [input])
            },
        ));
    }
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    assert_eq!(drain(&mut discovery, &mut g), 0);
    let a = matrix(&mut g, TensorOp::OpDot([args[0], args[1]]));
    let b = matrix(&mut g, TensorOp::OpDot([args[2], args[3]]));
    g.rebuild();
    discovery.refresh(&g);
    drain(&mut discovery, &mut g);
    backend::PROCESSED.lock().unwrap().clear();
    let rules: Vec<_> = rewrites::get_rewrites()
        .into_iter()
        .filter(|r| r.name.to_string() == "double-gemm")
        .collect();
    let mut g = Runner::default().with_egraph(g).run(&rules).egraph;
    let output = g.add(TensorOp::JoinStore([a, b]));
    g[output].data = TensorInfo {
        dtype: Dtype::U8,
        shape: vec![16384],
        is_const: false,
    };
    g.rebuild();
    let piis = extractor::extract_fast(
        &mut g,
        &[OutputRoot {
            eclass: output,
            hbm_offset: 32768,
        }],
        &inputs,
        &offsets,
    );
    assert_eq!(piis.len(), 1);
    let pii = &piis[0];
    assert!(pii.is_valid());
    let join = &pii.nodes[pii.outputs[0].node];
    assert_eq!(join.children[0].node, join.children[1].node);
    assert_eq!(
        join.children.iter().map(|v| v.result).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::DoubleGemm(_)))
            .count(),
        1
    );
}
