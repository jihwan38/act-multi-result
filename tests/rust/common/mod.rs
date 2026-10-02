//! All HLO fixtures are read from checked-in workloads or derived from them.
#![allow(dead_code)]

use std::path::PathBuf;

pub fn repo() -> PathBuf {
    PathBuf::from(std::env::var("ACT_TEST_REPO_ROOT").expect("Set ACT_TEST_REPO_ROOT"))
}

pub fn workload() -> PathBuf {
    repo().join("tests/fixtures/two_gemm_tuple.hlo")
}

pub fn gte_workload() -> PathBuf {
    repo().join("tests/fixtures/two_gemm_gte.hlo")
}

pub fn artifact(name: &str) -> PathBuf {
    let dir = PathBuf::from(std::env::var("ACT_TEST_BUILD").expect("Set ACT_TEST_BUILD"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

pub fn variant(name: &str, transform: impl FnOnce(String) -> String) -> PathBuf {
    let path = artifact(name);
    std::fs::write(
        &path,
        transform(std::fs::read_to_string(workload()).unwrap()),
    )
    .unwrap();
    path
}

pub fn scalar_gte_fixture(index: usize) -> PathBuf {
    assert!(index < 2);
    let source = std::fs::read_to_string(gte_workload()).unwrap();
    let before = "dot(%first, %second), lhs_contracting_dims={1}, rhs_contracting_dims={0}";
    assert_eq!(source.matches(before).count(), 1);
    let path = artifact(&format!("gte_scalar_output_{index}.hlo"));
    std::fs::write(
        &path,
        source.replace(before, &format!("get-tuple-element(%pair), index={index}")),
    )
    .unwrap();
    path
}
