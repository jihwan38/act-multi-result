//! Guard native INT8 saturation against an incorrectly matched clamp bound.
use backend::ir::egraph::TensorOp;
use backend::isel::{initializer, rewrites};
use egg::Runner;
use std::fs;
use std::path::PathBuf;

#[test]
fn requantization_requires_exact_integer_bounds() {
    let root = PathBuf::from(std::env::var("ACT_TEST_EXPERIMENT").unwrap());
    let source = fs::read_to_string(root.join("workloads/dependent_gemm.hlo")).unwrap();
    let bad_low = source.replacen("-128", "-127", 1);
    let bad_high = source.replacen("127", "126", 1);
    for (name, text, expected) in [("valid", source, true), ("low", bad_low, false),
                                  ("high", bad_high, false)] {
        let build = PathBuf::from(std::env::var("ACT_BUILD_DIR").unwrap());
        let path = build.join(format!("saturation_match_{name}.hlo"));
        fs::write(&path, text).unwrap();
        backend::PROCESSED.lock().unwrap().clear();
        let (graph, ..) = initializer::parse_hlo_module_to_egraph(path).unwrap();
        let runner = Runner::default().with_egraph(graph).with_iter_limit(8)
            .run(&rewrites::get_rewrites());
        let present = runner.egraph.classes().any(|class| class.nodes.iter()
            .any(|node| matches!(node, TensorOp::RequantizeStore0(..) | TensorOp::RequantizeStore1(..))));
        assert_eq!(present, expected, "{name}");
    }
}
