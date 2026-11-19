//! Exercise refreshed discovery with Gemmini's actual INT8 -> INT32 rules.
//! The row-split rule exists only in this test, not in the production Backend.
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

fn tensor(g: &mut Graph, op: TensorOp, dtype: Dtype, shape: Vec<i32>) -> Id {
    let id = g.add(op);
    g.set_analysis_data(
        id,
        TensorInfo {
            dtype,
            shape,
            is_const: false,
        },
    );
    id
}

fn input(g: &mut Graph, name: &str) -> Id {
    tensor(g, TensorOp::Var(name.into()), Dtype::I8, vec![16, 16])
}

fn dot(g: &mut Graph, a: Id, b: Id) -> Id {
    tensor(g, TensorOp::OpDot([a, b]), Dtype::I32, vec![16, 16])
}

fn drain(discovery: &mut Discovery, g: &mut Graph) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut count = 0;
    while !discovery.exhausted() {
        assert!(
            Instant::now() < deadline,
            "bounded discovery did not finish"
        );
        count += discovery.batch(g, 64, deadline);
    }
    count
}

fn has_pair(g: &Graph, a: Id, b: Id) -> bool {
    g.lookup(TensorOp::OpTuple(vec![g.find(a), g.find(b)]))
        .is_some()
}

fn joint_rules() -> Vec<Rewrite<TensorOp, TensorInfo>> {
    let rules: Vec<_> = rewrites::get_rewrites()
        .into_iter()
        .filter(|r| r.name.to_string() == "double-gemm")
        .collect();
    assert_eq!(rules.len(), 1, "use the real generated Gemmini joint rule");
    rules
}

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
        if g[id].data.dtype != Dtype::I32 || g[id].data.shape != vec![32, 16] {
            return vec![];
        }
        let a = subst["?a".parse().unwrap()];
        let b = subst["?b".parse().unwrap()];
        let top = tensor(
            g,
            TensorOp::OpSlice("0:16,0:16".into(), [a]),
            Dtype::I8,
            vec![16, 16],
        );
        let bottom = tensor(
            g,
            TensorOp::OpSlice("16:32,0:16".into(), [a]),
            Dtype::I8,
            vec![16, 16],
        );
        let y0 = dot(g, top, b);
        let y1 = dot(g, bottom, b);
        let joined = tensor(
            g,
            TensorOp::OpConcat("0".into(), [y0, y1]),
            Dtype::I32,
            vec![32, 16],
        );
        if g.union(id, joined) {
            vec![joined]
        } else {
            vec![]
        }
    }
}

#[test]
fn generated_integer_joint_rule_requires_refresh_after_row_split() {
    for refresh in [false, true] {
        backend::PROCESSED.lock().unwrap().clear();
        let mut g = Graph::default();
        let a = tensor(&mut g, TensorOp::Var("a".into()), Dtype::I8, vec![32, 16]);
        let b = input(&mut g, "b");
        tensor(&mut g, TensorOp::OpDot([a, b]), Dtype::I32, vec![32, 16]);
        g.rebuild();
        let mut discovery = Discovery::new(&g, 64);
        assert_eq!(drain(&mut discovery, &mut g), 0);
        let mut rules = joint_rules();
        rules.push(
            Rewrite::new(
                "test-only-split-rows",
                "(dot ?a ?b)".parse::<Pattern<TensorOp>>().unwrap(),
                SplitRows,
            )
            .unwrap(),
        );
        let runner = Runner::default()
            .with_egraph(g)
            .with_iter_limit(8)
            .with_hook(move |runner| {
                backend::PROCESSED.lock().unwrap().clear();
                runner.egraph.rebuild();
                if refresh {
                    discovery.refresh(&runner.egraph);
                }
                drain(&mut discovery, &mut runner.egraph);
                Ok(())
            })
            .run(&rules);
        assert!(matches!(
            runner.stop_reason,
            Some(egg::StopReason::Saturated)
        ));
        let present = runner.egraph.classes().any(|c| {
            c.nodes.iter().any(|op|
            matches!(op, TensorOp::DoubleGemm(args) if args[0] != args[2] && args[1] == args[3]))
        });
        assert_eq!(
            present, refresh,
            "frozen discovery must miss the newly exposed products"
        );
    }
}

#[test]
fn shared_reads_are_allowed_but_a_narrowed_previous_result_is_dependent() {
    for dependent in [false, true] {
        let mut g = Graph::default();
        let a = input(&mut g, "a");
        let b = input(&mut g, "b");
        let c = input(&mut g, "c");
        let y0 = dot(&mut g, a, b);
        let next = if dependent {
            tensor(
                &mut g,
                TensorOp::OpConvert("s8".into(), [y0]),
                Dtype::I8,
                vec![16, 16],
            )
        } else {
            a
        };
        let y1 = dot(&mut g, next, c);
        g.rebuild();
        let mut discovery = Discovery::new(&g, 64);
        for _ in 0..3 {
            discovery.refresh(&g);
            drain(&mut discovery, &mut g);
            assert_eq!(has_pair(&g, y0, y1), !dependent);
            assert_eq!(has_pair(&g, y1, y0), !dependent);
        }
    }
}

#[test]
fn refresh_preserves_the_global_candidate_limit() {
    for limit in [0, 1] {
        let mut g = Graph::default();
        let args = ["a", "b", "c", "d"].map(|name| input(&mut g, name));
        dot(&mut g, args[0], args[1]);
        dot(&mut g, args[2], args[3]);
        g.rebuild();
        let mut discovery = Discovery::new(&g, limit);
        let mut added = 0;
        for _ in 0..3 {
            discovery.refresh(&g);
            added += drain(&mut discovery, &mut g);
        }
        assert_eq!(added, limit);
    }
}

#[test]
fn refreshed_joint_producer_feeds_both_native_store_paths() {
    backend::PROCESSED.lock().unwrap().clear();
    let mut g = Graph::default();
    let mut inputs = HashSet::new();
    let mut offsets = vec![];
    let mut args = vec![];
    for i in 0..4 {
        let raw = tensor(&mut g, TensorOp::Var(format!("p{i}")), Dtype::U8, vec![256]);
        inputs.insert(raw);
        offsets.push((Some(raw), i * 256));
        let load = if i < 2 {
            TensorOp::Mvin0([raw])
        } else {
            TensorOp::Mvin1([raw])
        };
        args.push(tensor(&mut g, load, Dtype::I8, vec![16, 16]));
    }
    g.rebuild();
    let mut discovery = Discovery::new(&g, 64);
    assert_eq!(drain(&mut discovery, &mut g), 0);
    let y0 = dot(&mut g, args[0], args[1]);
    let y1 = dot(&mut g, args[2], args[3]);
    g.rebuild();
    discovery.refresh(&g);
    drain(&mut discovery, &mut g);
    let mut g = Runner::default().with_egraph(g).run(&joint_rules()).egraph;
    let o0 = tensor(&mut g, TensorOp::Mvout0([y0]), Dtype::U8, vec![1024]);
    let o1 = tensor(&mut g, TensorOp::Mvout1([y1]), Dtype::U8, vec![1024]);
    g.rebuild();
    let roots = [
        OutputRoot {
            eclass: o0,
            hbm_offset: 1024,
        },
        OutputRoot {
            eclass: o1,
            hbm_offset: 2048,
        },
    ];
    let piis = extractor::extract_fast(&mut g, &roots, &inputs, &offsets);
    assert_eq!(piis.len(), 1);
    let pii = &piis[0];
    assert!(pii.is_valid());
    let first = pii.nodes[pii.outputs[0].node].children[0];
    let second = pii.nodes[pii.outputs[1].node].children[0];
    assert_eq!(
        first.node, second.node,
        "both stores must share one producer"
    );
    assert_eq!((first.result, second.result), (0, 1));
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::DoubleGemm(_)))
            .count(),
        1
    );
}
