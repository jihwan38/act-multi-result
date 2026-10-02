//! Scalar regression: buffer versions of one value are not a dataflow cycle.
use backend::ir::buffer::Buffer;
use backend::ir::dtype::Dtype;
use backend::ir::egraph::{TensorInfo, TensorOp};
use backend::ir::metadata::OutputRoot;
use backend::isel::extractor;
use egg::{EGraph, Id};
use std::collections::HashSet;

fn add(graph: &mut EGraph<TensorOp, TensorInfo>, op: TensorOp, bytes: bool) -> Id {
    let id = graph.add(op);
    graph[id].data = TensorInfo {
        shape: if bytes { vec![8192] } else { vec![64, 64] },
        dtype: if bytes { Dtype::U8 } else { Dtype::BF16 },
        is_const: false,
    };
    id
}

fn graph(
    with_gemm: bool,
    failed_alternative: bool,
) -> (
    EGraph<TensorOp, TensorInfo>,
    HashSet<Id>,
    Vec<(Option<Id>, i32)>,
    OutputRoot,
) {
    let mut graph = EGraph::default();
    let a = add(&mut graph, TensorOp::Var("a".into()), true);
    let b = add(&mut graph, TensorOp::Var("b".into()), true);
    let bad_load = failed_alternative.then(|| {
        // Nonconstant but without an ISA implementation, not an undeclared
        // leaf (which ACT's separate constant detector would classify first).
        let missing = add(&mut graph, TensorOp::OpExp([a]), true);
        add(&mut graph, TensorOp::LoadRm("64".into(), [missing]), false)
    });
    let load_a = add(&mut graph, TensorOp::LoadRm("64".into(), [a]), false);
    let load_b = add(&mut graph, TensorOp::LoadRm("64".into(), [b]), false);
    let value = add(&mut graph, TensorOp::OpDot([load_a, load_b]), false);
    if let Some(bad_load) = bad_load {
        let bad = add(&mut graph, TensorOp::Gemm([load_a, bad_load]), false);
        graph.union(value, bad);
    }
    if with_gemm {
        let gemm = add(&mut graph, TensorOp::Gemm([load_a, load_b]), false);
        graph.union(value, gemm);
    }
    let mov = add(&mut graph, TensorOp::Mov("64".into(), [value]), false);
    graph.union(value, mov);
    let root = add(&mut graph, TensorOp::StoreRm("64".into(), [value]), true);
    graph.rebuild();
    (
        graph,
        HashSet::from([a, b]),
        vec![(Some(a), 0), (Some(b), 8192)],
        OutputRoot {
            eclass: root,
            hbm_offset: 16384,
        },
    )
}

#[test]
fn mov_retains_distinct_d2_and_d1_versions_of_the_same_eclass() {
    for failed_alternative in [false, true] {
        let (mut graph, inputs, offsets, output) = graph(true, failed_alternative);
        // Beyond the slow cutoff, with no tuple/GTE: exercise scalar fast only.
        let candidates = extractor::extract(&mut graph, &[output], &inputs, &offsets, 16);
        assert_eq!(candidates.len(), 1);
        let pii = &candidates[0];
        assert!(pii.is_valid());
        let mov = pii
            .nodes
            .iter()
            .find(|node| matches!(node.op, TensorOp::Mov(..)))
            .unwrap();
        assert_eq!(mov.results[0].buffer, Buffer::D1);
        assert_eq!(pii.result(mov.children[0]).buffer, Buffer::D2);
        assert!(matches!(
            pii.nodes[mov.children[0].node].op,
            TensorOp::Gemm(_)
        ));
        assert_eq!(
            pii.nodes
                .iter()
                .filter(|node| matches!(node.op, TensorOp::LoadRm(..)))
                .count(),
            2
        );
    }
}

#[test]
fn a_buffer_transfer_without_a_producer_is_not_a_complete_program() {
    let (mut graph, inputs, offsets, output) = graph(false, false);
    assert!(extractor::extract(&mut graph, &[output], &inputs, &offsets, 16).is_empty());
}
