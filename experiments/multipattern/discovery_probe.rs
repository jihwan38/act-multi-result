//! One actual generated joint-search/apply pass; no extraction or allocation.
#![allow(unexpected_cfgs)]
use backend::ir::dtype::Dtype;
use backend::ir::egraph::{TensorInfo, TensorOp};
use backend::isel::rewrites;
#[cfg(not(act_direct_rules))]
use backend::isel::discovery::Discovery;
#[cfg(any(act_discovery_report, act_direct_rules))]
use backend::isel::bounded_search::SearchLimits;
#[cfg(act_direct_rules)]
use backend::isel::rewrites::multi::DirectControl;
use egg::{EGraph, Id};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

type Graph = EGraph<TensorOp, TensorInfo>;

fn tensor(graph: &mut Graph, op: TensorOp, shape: &[i32]) -> Id {
    let id = graph.add(op);
    graph.set_analysis_data(id, TensorInfo {
        dtype: Dtype::BF16, shape: shape.to_vec(), is_const: false,
    });
    id
}

fn variable(graph: &mut Graph, name: String, shape: &[i32]) -> Id {
    tensor(graph, TensorOp::Var(name), shape)
}

fn fixture(scenario: &str, size: usize) -> (Graph, Vec<Vec<Id>>) {
    let mut graph = Graph::default();
    let mut results = vec![];
    for group in 0..size {
        if scenario == "metadata" {
            let x = variable(&mut graph, format!("x{group}"), &[4, 8]);
            let a = tensor(&mut graph, TensorOp::OpReshape("2,16".into(), [x]), &[2, 16]);
            let b = tensor(&mut graph, TensorOp::OpReshape("1,32".into(), [x]), &[1, 32]);
            results.push(vec![a, b]);
            continue;
        }
        let x = variable(&mut graph, format!("x{group}"), &[8, 8]);
        let a = variable(&mut graph, format!("a{group}"), &[8, 8]);
        let y = if matches!(scenario, "disjoint" | "independent") {
            variable(&mut graph, format!("y{group}"), &[8, 8])
        } else { x };
        let z = if matches!(scenario, "disjoint" | "independent") {
            variable(&mut graph, format!("z{group}"), &[8, 8])
        } else { x };
        let a_result = tensor(&mut graph, TensorOp::OpAdd([x, a]), &[8, 8]);
        let b = if scenario == "dependent" { a_result }
                else { variable(&mut graph, format!("b{group}"), &[8, 8]) };
        let c = variable(&mut graph, format!("c{group}"), &[8, 8]);
        let b_shape: &[i32] = if scenario == "wrong-shape" { &[4, 8] } else { &[8, 8] };
        let b_result = tensor(&mut graph, TensorOp::OpMultiply([y, b]), b_shape);
        let c_result = tensor(&mut graph, TensorOp::OpDot([z, c]), &[8, 8]);
        results.push(vec![a_result, b_result, c_result]);
    }
    graph.rebuild();
    (graph, results)
}

fn value(name: &str, default: usize) -> usize {
    std::env::var(name).ok().map(|text| text.parse().unwrap()).unwrap_or(default)
}

#[cfg(any(act_discovery_report, act_direct_rules))]
fn limits() -> SearchLimits {
    SearchLimits {
        raw_matches: value("ACT_JOINT_RAW_LIMIT", 4096),
        search_calls: value("ACT_JOINT_SEARCH_LIMIT", 4096),
        attempts: value("ACT_JOINT_ATTEMPT_LIMIT", 4096),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: discovery_probe SCENARIO SIZE");
    let scenario = args[1].as_str();
    assert!(matches!(scenario, "shared" | "disjoint" | "independent" | "dependent" | "wrong-shape" | "metadata"));
    let size: usize = args[2].parse().unwrap();
    assert!(size > 0);
    let cap = value("ACT_JOINT_LIMIT", 64);
    let (mut graph, expected) = fixture(scenario, size);
    let original_nodes = graph.total_size();
    backend::PROCESSED.lock().unwrap().clear();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(10);
    let discovery_ns;
    let discovery_report: Value;

    #[cfg(not(act_direct_rules))]
    {
        #[cfg(act_discovery_report)]
        let mut discovery = Discovery::with_limits(&graph, cap, limits(), Some(deadline));
        #[cfg(not(act_discovery_report))]
        let mut discovery = Discovery::new(&graph, cap);
        let initialized_ns = start.elapsed().as_nanos();
        while !discovery.exhausted() {
            assert!(Instant::now() < deadline, "discovery exceeded its per-process deadline");
            discovery.batch(&mut graph, cap, deadline);
        }
        discovery_ns = json!({"initialization": initialized_ns,
                              "batch": start.elapsed().as_nanos() - initialized_ns});
        #[cfg(act_discovery_report)]
        { discovery_report = serde_json::to_value(discovery.report()).unwrap(); }
        #[cfg(not(act_discovery_report))]
        { discovery_report = Value::Null; }
    }
    #[cfg(act_direct_rules)]
    {
        discovery_ns = Value::Null;
        discovery_report = Value::Null;
    }

    let rules_start = Instant::now();
    #[cfg(act_direct_rules)]
    let control = DirectControl::new(limits(), cap, Some(deadline));
    #[cfg(act_direct_rules)]
    let all_rules = rewrites::get_rewrites_with_control(control.clone());
    #[cfg(not(act_direct_rules))]
    let all_rules = rewrites::get_rewrites();
    let rules: Vec<_> = all_rules.into_iter().filter(|rule| {
        matches!(rule.name.as_str(), "shared-pair" | "shared-triple" | "reshape-pair"
                 | "independent-pair" | "independent-triple")
    }).collect();
    let rules_init_ns = rules_start.elapsed().as_nanos();
    assert!(!rules.is_empty());
    let search_start = Instant::now();
    let matches: Vec<_> = rules.iter().map(|rule| rule.search(&graph)).collect();
    let search_ns = search_start.elapsed().as_nanos();
    let search_substitutions: usize = matches.iter().flatten().map(|matched| matched.substs.len()).sum();
    let apply_start = Instant::now();
    for (rule, matches) in rules.iter().zip(&matches) {
        rule.apply(&mut graph, matches);
    }
    graph.rebuild();
    let apply_ns = apply_start.elapsed().as_nanos();
    let total_ns = start.elapsed().as_nanos();
    #[cfg(act_direct_rules)]
    let direct_report = serde_json::to_value(control.report()).unwrap();
    #[cfg(not(act_direct_rules))]
    let direct_report = Value::Null;

    let known: HashMap<Id, (usize, usize)> = expected.iter().enumerate().flat_map(|(group, results)| {
        results.iter().enumerate().map(move |(slot, id)| (*id, (slot, group)))
    }).map(|(id, position)| (graph.find(id), position)).collect();
    let mut tuples = HashSet::new();
    let mut isa_counts = BTreeMap::<String, usize>::new();
    let mut implemented_tuples = HashSet::new();
    for class in graph.classes() {
        let joint = class.nodes.iter().any(|op| matches!(op.to_string().as_str(),
            "shared_pair" | "shared_triple" | "reshape_pair" | "independent_pair" | "independent_triple"));
        for op in &class.nodes {
            let name = op.to_string();
            if matches!(name.as_str(), "shared_pair" | "shared_triple" | "reshape_pair" | "independent_pair" | "independent_triple") {
                *isa_counts.entry(name).or_default() += 1;
            }
            if let TensorOp::OpTuple(children) = op {
                let ids: Vec<_> = children.iter().map(|id| graph.find(*id)).collect();
                assert!(matches!(ids.len(), 2 | 3));
                let assignments: Vec<_> = ids.iter().map(|id| known.get(id).expect("tuple result must be a fixture result")).collect();
                for (slot, position) in assignments.iter().enumerate() {
                    assert_eq!(slot, position.0, "tuple result order changed");
                }
                if scenario != "independent" {
                    assert!(assignments.iter().all(|item| item.1 == assignments[0].1), "mixed shared-input groups were combined");
                }
                assert!(!matches!(scenario, "disjoint" | "dependent" | "wrong-shape"), "ineligible joint tuple was registered");
                if joint { implemented_tuples.insert(ids.clone()); }
                tuples.insert(ids);
            }
        }
    }
    let theoretical_pairs = match scenario {
        "shared" | "metadata" => size as u128,
        "independent" => (size as u128).pow(2),
        _ => 0,
    };
    let theoretical_triples = match scenario {
        "shared" => size as u128,
        "independent" => (size as u128).pow(3),
        _ => 0,
    };
    println!("{}", json!({
        "scenario": scenario, "size": size, "tuple_or_application_cap": cap,
        "original_nodes": original_nodes, "final_nodes": graph.total_size(),
        "discovery_ns": discovery_ns, "rules_init_ns": rules_init_ns,
        "rule_search_ns": search_ns, "rule_apply_rebuild_ns": apply_ns,
        "joint_candidate_pass_ns": total_ns, "rule_search_substitutions": search_substitutions,
        "registered_tuple_count": tuples.len(), "implemented_tuple_count": implemented_tuples.len(),
        "joint_isa_nodes": isa_counts, "discovery_report": discovery_report, "direct_report": direct_report,
        "theoretical_valid_pairs": theoretical_pairs, "theoretical_valid_triples": theoretical_triples,
        "unconstrained_pair_space": (size as u128).pow(2), "unconstrained_triple_space": (size as u128).pow(3),
        "semantic_assertions_passed": true,
        "scope": "One generated joint candidate search/apply pass; excludes IR rewrites, extraction, allocator, Oracle, and graph construction"
    }));
}
