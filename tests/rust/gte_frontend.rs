//! Prefix 0002: HLO/IR/projection checks without any multi-output extraction API.
mod common;

use backend::ir::dtype::Dtype;
use backend::ir::egraph::{tuple_elements, TensorOp};
use backend::isel::{initializer, rewrites};
use common::{artifact, gte_workload, scalar_gte_fixture, workload};
use egg::{Id, Language, Runner};
use std::collections::HashSet;

#[test]
fn tuple_frontend_keeps_both_dots_and_declared_output_order() {
    let (graph, _, root, inputs, metadata, ..) =
        initializer::parse_hlo_module_to_egraph(workload()).unwrap();
    let elements = tuple_elements(&graph, root).unwrap();
    assert_eq!(elements.len(), 2);
    assert_ne!(elements[0], elements[1]);
    assert_eq!(inputs.len(), 4);
    assert_eq!(
        metadata
            .output
            .iter()
            .map(|out| out.addr)
            .collect::<Vec<_>>(),
        vec![32768, 40960]
    );
    for (index, &element) in elements.iter().enumerate() {
        assert_eq!(graph[element].data.dtype, Dtype::BF16);
        assert_eq!(graph[element].data.shape, vec![64, 64]);
        let operands = graph[element]
            .nodes
            .iter()
            .find_map(|op| match op {
                TensorOp::OpDot(ids) => Some(*ids),
                _ => None,
            })
            .unwrap();
        for (id, name) in operands
            .iter()
            .zip([format!("%a{index}"), format!("%b{index}")])
        {
            // HBM input bytes pass through reshape/bitcast before the dot.
            let mut pending = vec![*id];
            let mut seen = HashSet::new();
            let mut leaves = HashSet::new();
            while let Some(value) = pending.pop() {
                if !seen.insert(value) {
                    continue;
                }
                for op in &graph[value].nodes {
                    if let TensorOp::Var(symbol) = op {
                        assert!(inputs.contains(&value));
                        leaves.insert(symbol.clone());
                    }
                    pending.extend(op.children());
                }
            }
            assert_eq!(leaves, HashSet::from([name]));
        }
    }
    assert!(!graph.classes().any(|class| class
        .nodes
        .iter()
        .any(|op| matches!(op, TensorOp::OpConcat(..)))));
}

#[test]
fn initializer_connects_intermediate_gtes_to_their_tuple() {
    let (graph, _, root, ..) = initializer::parse_hlo_module_to_egraph(gte_workload()).unwrap();
    let pairs = graph
        .classes()
        .filter(|class| tuple_elements(&graph, class.id).is_some())
        .map(|class| class.id)
        .collect::<Vec<_>>();
    assert_eq!(pairs.len(), 1);
    assert_ne!(pairs[0], root);
    for index in 0..2 {
        let gte = graph
            .classes()
            .find(|class| {
                class
                    .nodes
                    .contains(&TensorOp::OpGetTupleElement(index, [pairs[0]]))
            })
            .unwrap();
        assert_eq!(gte.data.dtype, Dtype::BF16);
        assert_eq!(gte.data.shape, vec![64, 64]);
    }
}

#[test]
fn gte_can_also_be_the_single_program_output() {
    for index in 0..2 {
        let path = scalar_gte_fixture(index);
        let (graph, _, root, _, metadata, ..) =
            initializer::parse_hlo_module_to_egraph(path).unwrap();
        assert_eq!(metadata.output.len(), 1);
        assert_eq!(graph[root].data.dtype, Dtype::U8);
        assert_eq!(graph[root].data.shape, vec![8192]);
        let bitcast = match &graph[root].nodes[0] {
            TensorOp::OpReshape(_, [id]) => *id,
            other => panic!("Expected output reshape, got {other:?}"),
        };
        let projected = match &graph[bitcast].nodes[0] {
            TensorOp::OpBitcvt([id]) => *id,
            other => panic!("Expected bitcast, got {other:?}"),
        };
        assert!(graph[projected]
            .nodes
            .iter()
            .any(|op| matches!(op, TensorOp::OpGetTupleElement(i, _) if *i == index)));
    }
}

#[test]
fn projection_rewrite_preserves_the_tuple_producer_edge() {
    let (mut graph, ..) = initializer::parse_hlo_module_to_egraph(gte_workload()).unwrap();
    let projections: Vec<_> = graph
        .classes()
        .flat_map(|class| {
            class.nodes.iter().filter_map(|op| {
                if let TensorOp::OpGetTupleElement(index, [tuple]) = op {
                    Some((class.id, *index, *tuple))
                } else {
                    None
                }
            })
        })
        .collect();
    assert_eq!(projections.len(), 2);
    let rules = rewrites::get_rewrites();
    let projection = rules
        .into_iter()
        .filter(|r| r.name.to_string() == "tuple-projection")
        .collect::<Vec<_>>();
    graph = Runner::default().with_egraph(graph).run(&projection).egraph;
    for (projected, index, tuple) in projections {
        let elements = backend::ir::egraph::tuple_elements(&graph, tuple).unwrap();
        assert_eq!(graph.find(projected), graph.find(elements[index]));
        assert!(graph[projected].nodes.iter().any(|op| matches!(op,
            TensorOp::OpGetTupleElement(i, [t]) if *i == index && graph.find(*t) == graph.find(tuple))));
    }
}

#[test]
fn gte_index_is_part_of_the_operator_identity() {
    use egg::FromOp;
    let zero = TensorOp::OpGetTupleElement(0, [Id::from(7)]);
    let one = TensorOp::OpGetTupleElement(1, [Id::from(7)]);
    assert!(!zero.matches(&one));
    assert_eq!(
        TensorOp::from_op(&one.to_string(), vec![Id::from(7)]).unwrap(),
        one
    );
    assert!(TensorOp::from_op("gte_bad", vec![Id::from(7)]).is_err());
}

#[test]
fn malformed_gte_is_rejected_before_extraction() {
    let original = std::fs::read_to_string(gte_workload()).unwrap();
    for (i, (before, after)) in [
        ("index=0", "index=2"),
        ("index=0", "index=-1"),
        ("index=0", ""),
        ("index=0", "index=0, index=1"),
        ("get-tuple-element(%pair)", "get-tuple-element(%missing)"),
        ("get-tuple-element(%pair)", "get-tuple-element(%a0)"),
        (
            "get-tuple-element(%pair)",
            "get-tuple-element(%pair, %pair)",
        ),
        ("%first = bf16[64,64]", "%first = bf16[32,64]"),
        ("%first = bf16[64,64]", "%first = f32[64,64]"),
    ]
    .iter()
    .enumerate()
    {
        assert!(
            original.contains(before),
            "Fixture no longer contains {before}"
        );
        let file = artifact(&format!("invalid_gte_{i}.hlo"));
        std::fs::write(&file, original.replace(before, after)).unwrap();
        assert!(
            std::panic::catch_unwind(|| initializer::parse_hlo_module_to_egraph(&file)).is_err(),
            "Accepted invalid GTE: {after}"
        );
    }
}
