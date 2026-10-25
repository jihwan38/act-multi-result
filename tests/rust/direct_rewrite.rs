//! Exercise generated joint Searchers/Appliers without Discovery or candidate tuples.
use backend::ir::{
    dtype::Dtype,
    egraph::{TensorInfo, TensorOp},
    metadata::OutputRoot,
};
use backend::isel::{
    bounded_search::{SearchLimits, StopReason},
    extractor, rewrites,
    rewrites::multi::DirectControl,
};
use egg::{EGraph, Id, Rewrite, Runner, Var};
use std::collections::HashSet;
use std::time::{Duration, Instant};

type Graph = EGraph<TensorOp, TensorInfo>;

fn tensor(graph: &mut Graph, op: TensorOp, shape: Vec<i32>) -> Id {
    let id = graph.add(op);
    graph.set_analysis_data(
        id,
        TensorInfo {
            dtype: Dtype::BF16,
            shape,
            is_const: false,
        },
    );
    id
}

fn matrix(graph: &mut Graph, op: TensorOp) -> Id {
    tensor(graph, op, vec![64, 64])
}

fn variable(graph: &mut Graph, name: &str) -> Id {
    matrix(graph, TensorOp::Var(name.into()))
}

fn control(limit: usize) -> DirectControl {
    DirectControl::new(SearchLimits::default(), limit, None)
}

fn joint_rule(control: &DirectControl) -> Rewrite<TensorOp, TensorInfo> {
    rewrites::get_rewrites_with_control(control.clone())
        .into_iter()
        .find(|rule| rule.name.to_string() == "double-gemm")
        .unwrap()
}

fn pair() -> (Graph, [Id; 4], [Id; 2]) {
    let mut graph = Graph::default();
    let args = ["a", "b", "c", "d"].map(|name| variable(&mut graph, name));
    let first = matrix(&mut graph, TensorOp::OpDot([args[0], args[1]]));
    let second = matrix(&mut graph, TensorOp::OpDot([args[2], args[3]]));
    graph.rebuild();
    (graph, args, [first, second])
}

fn tuple_count(graph: &Graph) -> usize {
    graph
        .classes()
        .flat_map(|class| &class.nodes)
        .filter(|node| matches!(node, TensorOp::OpTuple(_)))
        .count()
}

#[test]
fn search_is_read_only_and_apply_preserves_both_orders_and_repeated_result_ports() {
    let (mut graph, _, [a, b]) = pair();
    let control = control(64);
    let rule = joint_rule(&control);
    let size = graph.total_size();
    assert_eq!(tuple_count(&graph), 0);
    let matches = rule.search(&graph);
    assert_eq!(graph.total_size(), size);
    assert_eq!(tuple_count(&graph), 0);
    assert_eq!(
        matches
            .iter()
            .map(|found| found.substs.len())
            .sum::<usize>(),
        4
    );
    rule.apply(&mut graph, &matches);
    graph.rebuild();
    for first in [a, b] {
        for second in [a, b] {
            let tuple = graph
                .lookup(TensorOp::OpTuple(vec![
                    graph.find(first),
                    graph.find(second),
                ]))
                .expect("every valid ordered result assignment must be represented");
            assert!(graph[tuple]
                .nodes
                .iter()
                .any(|node| matches!(node, TensorOp::DoubleGemm(_))));
            for (port, result) in [first, second].into_iter().enumerate() {
                let projection = graph
                    .lookup(TensorOp::OpGetTupleElement(port, [tuple]))
                    .unwrap();
                assert_eq!(graph.find(projection), graph.find(result));
                assert_eq!(graph[result].data.dtype, Dtype::BF16);
                assert_eq!(graph[result].data.shape, [64, 64]);
            }
        }
    }
    assert_eq!(control.report().applications, 4);
}

#[test]
fn invalid_operand_contracts_do_not_spend_the_useful_application_cap() {
    let mut graph = Graph::default();
    for i in 0..8 {
        let left = tensor(
            &mut graph,
            TensorOp::Var(format!("wrong_left_{i}")),
            vec![64, 32],
        );
        let right = tensor(
            &mut graph,
            TensorOp::Var(format!("wrong_right_{i}")),
            vec![32, 64],
        );
        matrix(&mut graph, TensorOp::OpDot([left, right]));
    }
    let left = variable(&mut graph, "valid_left");
    let right = variable(&mut graph, "valid_right");
    let valid = matrix(&mut graph, TensorOp::OpDot([left, right]));
    graph.rebuild();
    let control = control(1);
    let rule = joint_rule(&control);
    let matches = rule.search(&graph);
    assert_eq!(tuple_count(&graph), 0);
    rule.apply(&mut graph, &matches);
    graph.rebuild();
    let tuple = graph
        .lookup(TensorOp::OpTuple(vec![graph.find(valid); 2]))
        .unwrap();
    assert!(graph[tuple]
        .nodes
        .iter()
        .any(|node| matches!(node, TensorOp::DoubleGemm(_))));
    assert_eq!(tuple_count(&graph), 1);
    assert_eq!(control.report().applications, 1);
    assert!(control.report().rejected_preconditions > 0);
}

#[test]
fn dependent_results_never_gain_a_joint_producer() {
    let (mut graph, args, [a, _]) = pair();
    let dependent = matrix(&mut graph, TensorOp::OpDot([a, args[3]]));
    graph.rebuild();
    let control = control(64);
    let rule = joint_rule(&control);
    let graph = Runner::default()
        .with_egraph(graph)
        .with_scheduler(control.scheduler())
        .with_iter_limit(4)
        .run(&[rule])
        .egraph;
    for values in [[a, dependent], [dependent, a]] {
        assert!(graph
            .lookup(TensorOp::OpTuple(values.map(|id| graph.find(id)).to_vec()))
            .is_none());
    }
}

#[test]
fn apply_rechecks_input_availability_after_a_union_changes_the_match() {
    let mut graph = Graph::default();
    let x = variable(&mut graph, "x");
    let input = matrix(&mut graph, TensorOp::OpCopy([x]));
    let b = variable(&mut graph, "b");
    let c = variable(&mut graph, "c");
    let d = variable(&mut graph, "d");
    let first = matrix(&mut graph, TensorOp::OpDot([input, b]));
    let second = matrix(&mut graph, TensorOp::OpDot([c, d]));
    graph.rebuild();
    let control = control(64);
    let rule = joint_rule(&control);
    let mut matches = rule.search(&graph);
    let roots: [Var; 2] = ["?__act_result_0", "?__act_result_1"].map(|s| s.parse().unwrap());
    for found in &mut matches {
        found.substs.retain(|subst| {
            graph.find(subst[roots[0]]) == graph.find(first)
                && graph.find(subst[roots[1]]) == graph.find(second)
        });
    }
    matches.retain(|found| !found.substs.is_empty());
    assert_eq!(
        matches
            .iter()
            .map(|found| found.substs.len())
            .sum::<usize>(),
        1
    );
    graph.union(input, second);
    graph.rebuild();
    let before = graph.total_size();
    assert!(rule.apply(&mut graph, &matches).is_empty());
    assert_eq!(graph.total_size(), before);
    assert_eq!(tuple_count(&graph), 0);
    assert_eq!(control.report().rejected_at_apply, 1);
}

#[test]
fn exhausted_joint_work_does_not_disable_scalar_isa_rewrites() {
    for mode in ["raw", "applications", "deadline"] {
        let (graph, _, [result, _]) = pair();
        let mut limits = SearchLimits::default();
        if mode == "raw" {
            limits.raw_matches = 0;
        }
        let control = DirectControl::new(
            limits,
            if mode == "applications" { 0 } else { 64 },
            (mode == "deadline").then(Instant::now),
        );
        let rules = rewrites::get_rewrites_with_control(control.clone());
        backend::PROCESSED.lock().unwrap().clear();
        let runner = Runner::default()
            .with_egraph(graph)
            .with_scheduler(control.scheduler())
            .with_iter_limit(3)
            .with_time_limit(Duration::from_secs(2))
            .with_hook(|_| {
                backend::PROCESSED.lock().unwrap().clear();
                Ok(())
            })
            .run(&rules);
        let graph = &runner.egraph;
        assert!(
            graph[graph.find(result)]
                .nodes
                .iter()
                .any(|node| matches!(node, TensorOp::Gemm0(_) | TensorOp::Gemm1(_))),
            "{mode}"
        );
        let report = control.report();
        assert!(report.disabled, "{mode}");
        assert!(!report.complete, "{mode}");
        assert_eq!(report.applications, 0, "{mode}");
        assert_eq!(
            report.search.stop_reason,
            match mode {
                "raw" => StopReason::RawMatchLimit,
                "applications" => StopReason::TupleLimit,
                _ => StopReason::Deadline,
            }
        );
    }
}

#[test]
fn direct_candidate_extracts_as_one_physical_producer_with_two_ports() {
    let mut graph = Graph::default();
    let mut inputs = HashSet::new();
    let mut offsets = Vec::new();
    let mut args = Vec::new();
    for i in 0..4 {
        let input = graph.add(TensorOp::Var(format!("p{i}")));
        graph[input].data = TensorInfo {
            dtype: Dtype::U8,
            shape: vec![8192],
            is_const: false,
        };
        inputs.insert(input);
        offsets.push((Some(input), i * 8192));
        args.push(matrix(
            &mut graph,
            if i < 2 {
                TensorOp::LoadRm0("64".into(), [input])
            } else {
                TensorOp::LoadRm1("64".into(), [input])
            },
        ));
    }
    let first = matrix(&mut graph, TensorOp::OpDot([args[0], args[1]]));
    let second = matrix(&mut graph, TensorOp::OpDot([args[2], args[3]]));
    graph.rebuild();
    let control = control(64);
    let rule = joint_rule(&control);
    let mut graph = Runner::default()
        .with_egraph(graph)
        .with_scheduler(control.scheduler())
        .run(&[rule])
        .egraph;
    let output = graph.add(TensorOp::JoinStore([first, second]));
    graph[output].data = TensorInfo {
        dtype: Dtype::U8,
        shape: vec![16384],
        is_const: false,
    };
    graph.rebuild();
    let piis = extractor::extract_fast(
        &mut graph,
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
        join.children
            .iter()
            .map(|value| value.result)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        pii.nodes
            .iter()
            .filter(|node| matches!(node.op, TensorOp::DoubleGemm(_)))
            .count(),
        1
    );
}
