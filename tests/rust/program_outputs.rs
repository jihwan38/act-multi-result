//! Prefix 0005: satisfy all program outputs through the existing fast/slow extractors.
//! Fixtures come from the repository's actual workloads, never copied HLO text.
use std::collections::HashSet;
use std::path::Path;

mod common;
use common::{artifact, repo, scalar_gte_fixture, variant, workload};

use backend::ir::buffer::Buffer;
use backend::ir::dtype::Dtype;
use backend::ir::egraph::{TensorInfo, TensorOp};
use backend::ir::metadata::OutputRoot;
use backend::ir::pii::{PiiGraph, ValueRef};
use backend::isel::{extractor, initializer, rewrites};
use egg::{EGraph, Id, Language, Runner};

type Prepared = (
    EGraph<TensorOp, TensorInfo>,
    Vec<(Option<Id>, i32)>,
    HashSet<Id>,
    Vec<OutputRoot>,
);

fn rewritten(path: &Path) -> Prepared {
    let (graph, offsets, _, inputs, _, outputs) =
        initializer::parse_hlo_module_to_egraph(path).unwrap();
    let runner = Runner::default()
        .with_egraph(graph)
        .with_iter_limit(8)
        .with_node_limit(5000)
        .with_hook(|_| {
            backend::PROCESSED.lock().unwrap().clear();
            Ok(())
        })
        .run(&rewrites::get_rewrites());
    (runner.egraph, offsets, inputs, outputs)
}

fn selected(path: &Path) -> PiiGraph {
    selected_with_limit(path, 10)
}

fn selected_with_limit(path: &Path, limit: usize) -> PiiGraph {
    let (mut graph, offsets, inputs, outputs) = rewritten(path);
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, limit);
    assert!(!piis.is_empty(), "No complete PII for {}", path.display());
    for pii in &piis {
        check(pii, &outputs);
    }
    piis.into_iter().min_by_key(|pii| pii.nodes.len()).unwrap()
}

fn check(pii: &PiiGraph, outputs: &[OutputRoot]) {
    assert!(pii.is_valid(), "Invalid buffer edge or detached PII node");
    assert_eq!(pii.outputs.len(), outputs.len());
    for (&id, output) in pii.outputs.iter().zip(outputs) {
        let node = pii.result(id);
        assert_eq!(node.buffer, Buffer::HBM);
        assert_eq!(node.hbm_offset, Some(output.hbm_offset));
        assert_eq!(node.info.dtype, Dtype::U8);
        assert_eq!(node.info.shape, vec![8192]);
    }
    for node in &pii.nodes {
        assert!(!matches!(
            node.op,
            TensorOp::OpTuple(_) | TensorOp::OpGetTupleElement(..) | TensorOp::OpDot(_)
        ));
    }
}

fn input_names(pii: &PiiGraph, root: ValueRef) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        let node = &pii.nodes[id.node];
        if let TensorOp::Var(name) = &node.op {
            names.insert(name.clone());
        }
        pending.extend(&node.children);
    }
    names
}

#[test]
fn logical_tuple_stays_typed_while_each_output_gets_a_byte_view() {
    let (graph, offsets, root, _, metadata, outputs) =
        initializer::parse_hlo_module_to_egraph(workload()).unwrap();
    let children = graph[root]
        .nodes
        .iter()
        .find_map(|op| match op {
            TensorOp::OpTuple(children) => Some(children),
            _ => None,
        })
        .unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(metadata.output.len(), 2);
    assert_eq!(
        outputs.iter().map(|o| o.hbm_offset).collect::<Vec<_>>(),
        vec![32768, 40960]
    );
    for (i, (&child, output)) in children.iter().zip(&outputs).enumerate() {
        assert_eq!(graph[child].data.dtype, Dtype::BF16);
        assert_eq!(graph[child].data.shape, vec![64, 64]);
        assert!(graph[child]
            .nodes
            .iter()
            .any(|op| matches!(op, TensorOp::OpDot(_))));
        let bitcast = match &graph[output.eclass].nodes[0] {
            TensorOp::OpReshape(_, ids) => ids[0],
            op => panic!("Expected byte reshape, got {op:?}"),
        };
        let projected = match &graph[bitcast].nodes[0] {
            TensorOp::OpBitcvt([id]) => *id,
            _ => panic!("Expected bitcast"),
        };
        assert!(graph[projected]
            .nodes
            .contains(&TensorOp::OpGetTupleElement(i, [root])));
        assert_eq!(graph[projected].data.dtype, graph[child].data.dtype);
        assert_eq!(graph[projected].data.shape, graph[child].data.shape);
        assert_eq!(graph[output.eclass].data.dtype, Dtype::U8);
        assert_eq!(graph[output.eclass].data.shape, vec![8192]);
        assert_eq!(offsets[4 + i], (Some(output.eclass), output.hbm_offset));
    }
}

#[test]
fn two_gemms_reach_both_ordered_hbm_outputs() {
    let pii = selected(&workload());
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Gemm(_)))
            .count(),
        2
    );
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Mov(..)))
            .count(),
        2
    );
    assert_eq!(
        input_names(&pii, pii.outputs[0]),
        HashSet::from(["%a0".into(), "%b0".into()])
    );
    assert_eq!(
        input_names(&pii, pii.outputs[1]),
        HashSet::from(["%a1".into(), "%b1".into()])
    );
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Var(_)))
            .count(),
        4
    );
    let text = pii.to_string();
    assert!(text.starts_with(&format!(
        "# outputs: t{}, t{}\n",
        pii.outputs[0].node, pii.outputs[1].node
    )));
    pii.save(&artifact("two_gemm_tuple.pii"));
}

#[test]
fn intermediate_tuple_is_consumed_by_a_later_gemm() {
    let path = repo().join("tests/fixtures/two_gemm_gte.hlo");
    let (graph, _, root, _, _, _) = initializer::parse_hlo_module_to_egraph(&path).unwrap();
    let tuples = graph
        .classes()
        .filter(|class| {
            class
                .nodes
                .iter()
                .any(|op| matches!(op, TensorOp::OpTuple(_)))
        })
        .collect::<Vec<_>>();
    assert_eq!(tuples.len(), 1);
    assert_ne!(tuples[0].id, root);
    // Three GEMMs need a 15-node scalar extraction tree (including inputs,
    // loads, moves and the store). The old 10-node budget is insufficient.
    let pii = selected_with_limit(&path, 15);
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Gemm(_)))
            .count(),
        3
    );
    assert_eq!(pii.outputs.len(), 1);
}

#[test]
fn unused_tuple_of_constants_is_not_folded_as_a_byte_tensor() {
    let (mut graph, offsets, inputs, outputs) = rewritten(&workload());
    let constant = graph.add(TensorOp::OpConstant("1".into()));
    graph.set_analysis_data(
        constant,
        TensorInfo {
            dtype: Dtype::BF16,
            shape: vec![1],
            is_const: true,
        },
    );
    let tuple = graph.add(TensorOp::OpTuple(vec![constant, constant]));
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, 10);
    assert!(!piis.is_empty());
    assert!(!graph[tuple]
        .nodes
        .iter()
        .any(|op| matches!(op, TensorOp::DetectedConst(_))));
}

#[test]
fn shared_input_is_not_duplicated_in_the_merged_pii() {
    let path = variant("shared_input.hlo", |text| {
        text.replace("dot(%a1, %b1)", "dot(%a0, %b1)")
    });
    let pii = selected(&path);
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|n| matches!(n.op, TensorOp::Var(_)))
            .count(),
        3
    );
    assert!(input_names(&pii, pii.outputs[0]).contains("%a0"));
    assert!(input_names(&pii, pii.outputs[1]).contains("%a0"));
}

#[test]
fn repeated_value_has_distinct_store_addresses() {
    let (mut graph, offsets, inputs, mut outputs) = rewritten(&workload());
    outputs[1].eclass = outputs[0].eclass;
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, 10);
    assert!(!piis.is_empty());
    for pii in &piis {
        check(pii, &outputs);
        assert_ne!(pii.outputs[0].node, pii.outputs[1].node);
        assert_eq!(
            pii.nodes
                .iter()
                .filter(|n| matches!(n.op, TensorOp::Var(_)))
                .count(),
            2
        );
    }
}

#[test]
fn output_order_comes_from_bindings_not_node_order_or_tuple_root() {
    let (mut graph, offsets, inputs, mut outputs) = rewritten(&workload());
    outputs.reverse();
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, 10);
    assert!(!piis.is_empty());
    for pii in &piis {
        check(pii, &outputs);
        assert_eq!(
            input_names(pii, pii.outputs[0]),
            HashSet::from(["%a1".into(), "%b1".into()])
        );
    }
}

#[test]
fn one_element_tuple_keeps_its_element_output_binding() {
    let path = variant("one_output_tuple.hlo", |text| {
        text.replace("(bf16[64,64]{1,0}, bf16[64,64]{1,0})", "(bf16[64,64]{1,0})")
            .replace("tuple(%y0, %y1)", "tuple(%y0)")
    });
    let pii = selected(&path);
    assert_eq!(pii.outputs.len(), 1);
    assert_eq!(
        input_names(&pii, pii.outputs[0]),
        HashSet::from(["%a0".into(), "%b0".into()])
    );
}

#[test]
fn three_outputs_are_not_special_cased_as_a_pair() {
    let path = variant("three_outputs.hlo", |text| {
        text.replace(
            "(bf16[64,64]{1,0}, bf16[64,64]{1,0})",
            "(bf16[64,64]{1,0}, bf16[64,64]{1,0}, bf16[64,64]{1,0})",
        )
        .replace("tuple(%y0, %y1)", "tuple(%y0, %y1, %y0)")
    });
    let pii = selected(&path);
    assert_eq!(pii.outputs.len(), 3);
    assert_eq!(
        pii.outputs
            .iter()
            .map(|&id| pii.result(id).hbm_offset.unwrap())
            .collect::<Vec<_>>(),
        vec![32768, 40960, 49152],
    );
    assert_eq!(
        input_names(&pii, pii.outputs[0]),
        input_names(&pii, pii.outputs[2])
    );
    assert_ne!(pii.outputs[0], pii.outputs[2]);
}

#[test]
fn equivalent_values_in_different_buffers_are_not_interchangeable() {
    let mut pii = selected(&workload());
    let root = pii.outputs[0];
    let mov = pii.nodes[root.node].children[0];
    assert!(matches!(pii.nodes[mov.node].op, TensorOp::Mov(..)));
    let gemm = pii.nodes[mov.node].children[0];
    assert_eq!(pii.result(gemm).buffer, Buffer::D2);
    // Bypass the D2 -> D1 move. StoreRm cannot read the resulting D2 edge.
    pii.nodes[root.node].children[0] = gemm;
    assert!(!pii.is_valid());
}

#[test]
fn no_partial_program_when_an_output_cannot_be_extracted() {
    let (mut graph, offsets, inputs, mut outputs) = rewritten(&workload());
    assert!(!extractor::extract(&mut graph, &outputs[..1], &inputs, &offsets, 10).is_empty());
    let input = *inputs.iter().next().unwrap();
    outputs[1].eclass = graph.add(TensorOp::OpDot([input, input]));
    assert!(extractor::extract(&mut graph, &outputs, &inputs, &offsets, 10).is_empty());
}

#[test]
fn fast_extractor_also_combines_all_output_paths() {
    // Rebuild a selected PII as an acyclic ISA-only e-graph, so this tests the
    // fast path without requiring it to unfold the upstream copy/mov cycles.
    let source = selected(&workload());
    let mut graph = EGraph::<TensorOp, TensorInfo>::default();
    let mut ids = vec![];
    let mut inputs = HashSet::new();
    let mut offsets = vec![];
    for node in &source.nodes {
        let mut op = node.op.clone();
        for (slot, &child) in op.children_mut().iter_mut().zip(&node.children) {
            *slot = ids[child.node];
        }
        let id = graph.add(op);
        graph.set_analysis_data(id, node.results[0].info.clone());
        if matches!(node.op, TensorOp::Var(_)) {
            inputs.insert(id);
            offsets.push((Some(id), node.results[0].hbm_offset.unwrap()));
        }
        ids.push(id);
    }
    let outputs: Vec<OutputRoot> = source
        .outputs
        .iter()
        .map(|&id| OutputRoot {
            eclass: ids[id.node],
            hbm_offset: source.result(id).hbm_offset.unwrap(),
        })
        .collect();
    let piis = extractor::extract(
        &mut graph,
        &outputs,
        &inputs,
        &offsets,
        backend::SLOW_LIMIT_CUTOFF + 1,
    );
    assert_eq!(piis.len(), 1, "Slow extraction is disabled here");
    check(&piis[0], &outputs);
    let mut incomplete = outputs.clone();
    let input = *inputs.iter().next().unwrap();
    incomplete[1].eclass = graph.add(TensorOp::OpDot([input, input]));
    assert!(extractor::extract(
        &mut graph,
        &incomplete,
        &inputs,
        &offsets,
        backend::SLOW_LIMIT_CUTOFF + 1,
    )
    .is_empty());
}

#[test]
fn input_alias_is_not_relabelled_as_an_output_copy() {
    let mut graph = EGraph::<TensorOp, TensorInfo>::default();
    let input = graph.add(TensorOp::Var("%input".into()));
    graph.set_analysis_data(
        input,
        TensorInfo {
            dtype: Dtype::U8,
            shape: vec![8192],
            is_const: false,
        },
    );
    let outputs = [OutputRoot {
        eclass: input,
        hbm_offset: 8192,
    }];
    assert!(extractor::extract(
        &mut graph,
        &outputs,
        &HashSet::from([input]),
        &vec![(Some(input), 0)],
        10,
    )
    .is_empty());
}

#[test]
fn stock_attention_still_extracts_one_complete_output() {
    let path = repo().join("cases/qkv/workloads/attention.hlo");
    let (mut graph, offsets, inputs, outputs) = rewritten(&path);
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, 15);
    assert!(!piis.is_empty());
    for pii in &piis {
        check(pii, &outputs);
        assert_eq!(pii.outputs.len(), 1);
        assert_eq!(
            input_names(pii, pii.outputs[0]),
            HashSet::from(["%q".into(), "%k".into(), "%v".into()])
        );
    }
}

#[test]
fn scalar_gte_output_uses_only_the_requested_element() {
    for index in 0..2 {
        let path = scalar_gte_fixture(index);
        let pii = selected(&path);
        assert_eq!(
            pii.nodes
                .iter()
                .filter(|n| matches!(n.op, TensorOp::Gemm(_)))
                .count(),
            1
        );
        assert_eq!(
            input_names(&pii, pii.outputs[0]),
            HashSet::from([format!("%a{index}"), format!("%b{index}")])
        );
    }
}

#[test]
fn empty_program_is_one_valid_candidate_not_an_extraction_failure() {
    let source = std::fs::read_to_string(
        repo().join("tests/fixtures/nested_two_gemm_tuple.hlo"),
    )
    .unwrap();
    let lines: Vec<_> = source
        .lines()
        .map(|line| {
            if line.starts_with("HloModule ") {
                format!("{}->()}}", line.split_once("->").unwrap().0)
            } else if line.trim_start().starts_with("ROOT ") {
                "  ROOT %nothing = () tuple()".to_string()
            } else {
                line.to_string()
            }
        })
        .collect();
    let path = artifact("empty_nested_root.hlo");
    std::fs::write(&path, lines.join("\n")).unwrap();
    let (mut graph, offsets, root, inputs, metadata, outputs) =
        initializer::parse_hlo_module_to_egraph(path).unwrap();
    assert!(backend::ir::egraph::tuple_elements(&graph, root).unwrap().is_empty());
    assert!(metadata.output.is_empty() && outputs.is_empty());
    assert_eq!(metadata.input.len(), 4); // Dead calculations do not change the argument ABI.
    let piis = extractor::extract(&mut graph, &outputs, &inputs, &offsets, 1);
    assert_eq!(piis.len(), 1);
    assert!(piis[0].is_valid());
    assert!(piis[0].nodes.is_empty() && piis[0].outputs.is_empty());
    assert!(PiiGraph::default().is_valid());
}
