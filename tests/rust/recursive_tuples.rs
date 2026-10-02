//! Recursive logical values; only tensor leaves belong to the physical ABI.
mod common;

use backend::ir::dtype::Dtype;
use backend::ir::egraph::{tuple_elements, TensorOp};
use backend::ir::value::{TensorType, ValueType};
use backend::isel::initializer;
use common::{artifact, repo};
use std::fs;

fn workload() -> std::path::PathBuf {
    repo().join("tests/fixtures/nested_two_gemm_tuple.hlo")
}

#[test]
fn recursive_types_preserve_empty_branches_and_scalar_leaves() {
    let scalar = ValueType::Tensor(TensorType {
        dtype: Dtype::BF16,
        shape: vec![],
        tile: "".into(),
    });
    let matrix = ValueType::Tensor(TensorType {
        dtype: Dtype::FP32,
        shape: vec![2, 3],
        tile: "".into(),
    });
    let empty = ValueType::Tuple(vec![]);
    let ty = ValueType::Tuple(vec![
        scalar.clone(),
        ValueType::Tuple(vec![empty.clone(), matrix.clone()]),
        empty.clone(),
    ]);
    assert_eq!(
        ty.leaves()
            .iter()
            .map(|leaf| leaf.bytes())
            .collect::<Vec<_>>(),
        vec![2, 24]
    );
    assert_eq!(
        serde_json::to_value(ty.tree(&mut 0)).unwrap(),
        serde_json::json!([0, [[], 1], []])
    );
    let different = ValueType::Tuple(vec![
        ValueType::Tuple(vec![scalar, empty.clone()]),
        matrix,
        empty.clone(),
    ]);
    assert!(!ty.same_type(&different));
    assert!(ty.same_type(&ty.clone()));
    assert!(empty.leaves().is_empty());
}

#[test]
fn nested_root_binds_only_tensor_leaves_in_depth_first_order() {
    let (graph, offsets, root, inputs, metadata, outputs) =
        initializer::parse_hlo_module_to_egraph(workload()).unwrap();
    assert_eq!(inputs.len(), 4);
    assert_eq!(offsets.len(), 6);
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs.iter().map(|out| out.hbm_offset).collect::<Vec<_>>(),
        vec![32768, 40960]
    );
    assert_eq!(
        serde_json::to_value(&metadata.output_tree).unwrap(),
        serde_json::json!([0, [[], 1]])
    );
    let outer = tuple_elements(&graph, root).unwrap();
    let inner = tuple_elements(&graph, outer[1]).unwrap();
    assert!(tuple_elements(&graph, inner[0]).unwrap().is_empty());
    for leaf in [outer[0], inner[1]] {
        assert!(graph[leaf]
            .nodes
            .iter()
            .any(|node| matches!(node, TensorOp::OpDot(_))));
    }
    assert!(offsets.iter().all(|(id, _)| id.is_some()));
}

#[test]
fn scalar_parameter_byte_shape_has_no_empty_dimension() {
    let source = fs::read_to_string(workload())
        .unwrap()
        .replace("bf16[64,64]", "bf16[]")
        .replace("{1,0}", "")
        .replace(" dot(", " add(")
        .replace(", lhs_contracting_dims={1}, rhs_contracting_dims={0}", "");
    let path = artifact("nested_scalar_add.hlo");
    fs::write(&path, source).unwrap();
    let (graph, _, _, inputs, metadata, outputs) =
        initializer::parse_hlo_module_to_egraph(path).unwrap();
    assert_eq!(inputs.len(), 4);
    assert!(graph.classes().any(|class| class
        .nodes
        .iter()
        .any(|node| matches!(node, TensorOp::OpReshape(shape, _) if shape == "2"))));
    assert!(metadata
        .input
        .iter()
        .chain(&metadata.output)
        .all(|info| info.shape.is_empty()));
    assert_eq!(
        outputs.iter().map(|out| out.hbm_offset).collect::<Vec<_>>(),
        vec![8, 10]
    );
}

#[test]
fn equal_leaf_counts_do_not_make_different_tuple_shapes_compatible() {
    let source = fs::read_to_string(workload()).unwrap();
    let (header, body) = source.split_once('\n').unwrap();
    let header = header.replace(
        "(bf16[64,64]{1,0}, ((), bf16[64,64]{1,0}))",
        "((bf16[64,64]{1,0}, ()), bf16[64,64]{1,0})",
    );
    assert_ne!(header, source.lines().next().unwrap());
    let path = artifact("wrong_nested_header.hlo");
    fs::write(&path, format!("{header}\n{body}")).unwrap();
    assert!(std::panic::catch_unwind(|| initializer::parse_hlo_module_to_egraph(path)).is_err());
}
