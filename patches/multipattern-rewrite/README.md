# multipattern-rewrite: 공동 검색과 적용을 rewrite로 연결

[예제로 보는 두 방식의 차이](../../handoff/README.md#3-검색-실험-multipattern과-multipattern-rewrite) · [비교 보고서](../../experiments/multipattern/REPORT.md) · [적용 목록](series)

2026-10-26의 [커밋 3b85d5d](https://github.com/jihwan38/act-multi-result/commit/3b85d5dad68867fb6768d355ecba85876fcd5685)에서
추가한 후속 구현이다. `multi-result`의 기반과 `multipattern`의 공동 검색을 발전시켜,
목표였던 **공동 검색 → 조건 검사 → 명령 적용을 한 rewrite에서 처리하는 구조**로 통합했다.
중간 tuple 후보를 등록한 뒤 다시 검색하던 연결을 정리한 성과가 있다.
성능 비교에서 전반적인 우위는 확인하지 못했으며, 기본 실행 구성은 `multi-result`로 유지한다.

적용 순서는 **고정 upstream → baseline → multi-result → multipattern → multipattern-rewrite**다.
앞 묶음의 검색 예산 코드 등을 사용하므로 `multipattern` 패치도 적용한다.
다만 최종 컴파일 경로에서는 별도 `Discovery` hook을 제거한다.

## 개별 패치와 실행 흐름

파일은 [Backend 0001 — generated-rewrites](act-backend/0001-generated-rewrites.patch) 하나다.
아래 경로는 `act_backend/`를 기준으로 한다.

| 수정 위치 | 바꾸는 동작 |
| --- | --- |
| `generic/src/isel/rewrites/lib.rs`, `applier.rs` | `get_rewrites_with_control()` → `get_applier()`에서 다중 결과 규칙을 `JointRule`로 연결한다. 단일 결과 규칙은 기존 적용 경로를 유지한다. |
| `isel_ir2isa_rewrites_rs_generator.py` | 공동 검색이 반환하는 바인딩 순서에 맞춰 의미식 노드의 타입·shape·metadata·입력 조건을 검사하는 코드를 생성한다. |
| `generic/src/isel/rewrites/multi.rs` | `MatchPlan`으로 여러 출력의 패턴과 바인딩을 구성한다. `JointRule::search_anchors()`가 MultiPattern 검색·조건 검사를 하고, `apply_one()`이 저장된 바인딩으로 ISA 생산자와 각 결과의 GTE를 연결한다. |
| 같은 `multi.rs`의 `DirectControl`, `DirectScheduler` | 컴파일 전체의 검색량·유효 적용 수를 제어하고 ISA별 몫을 나눈다. 첫 검색에서 기존 후보만 나왔다고 다음 후보를 보기 전에 saturation으로 끝내지 않도록 제어한다. |
| `generic/src/pipeline.rs` | discovery hook 대신 생성된 공동 rewrite와 scheduler를 Runner에 연결하고 `ACT_DIRECT_REWRITE` 통계를 기록한다. 추출·할당·최종 평가는 기존 경로를 사용한다. |

검색은 그래프를 수정하지 않는다. 앞선 rewrite가 e-class를 합치면 조건이 달라질 수 있어
**적용 직전에 다시 검사**한다. 성공한 적용에서만 tuple·ISA 생산자·결과 참조를 만든다.
tuple 표현은 추출기가 명령의 출력 번호를 구분하는 데 필요하므로 계속 사용한다.

## 한도와 비교 결과

검색량과 유효 적용 수는 컴파일 전체의 누적 한도이며 ISA별로 고정 배분한다.
한 ISA가 앞에서 예산을 독점하지 못하지만, 매칭되지 않는 ISA의 남은 몫을 다른 ISA에 넘기지 않는다.
다중 결과 예산이 소진돼도 단일 결과·IR rewrite는 계속 실행한다.

검색·검사·적용을 한 경로에 모으는 구조는 구현했지만, 비교 입력에서 전반적인 컴파일 속도나
최종 코드 비용의 우위는 확인하지 못했다. 이 때문에 실험 구성으로 남겨뒀다.
측정 당시 `3b85d5d`와 현재 실행 구성의 차이·상세 수치는 [보고서](../../experiments/multipattern/REPORT.md)에 있다.

## 적용된 소스와 확인 경로

`python scripts/prepare_sources.py multipattern-rewrite`로 준비하면
`build/multipattern-rewrite/src/act-backend/act_backend/`에서 위 파일을 읽을 수 있다.
변경은 [소스 수정 안내](../../README.md#패치가-적용된-소스-읽기와-수정)에 따라 패치에 보존한다.

검색·적용을 고칠 때는 거절된 후보가 그래프나 유효 적용 예산을 바꾸지 않는지, 검색 후 union이
발생해도 다시 검사하는지, 한 생산자의 두 결과가 올바르게 추출되는지를 확인한다.
[direct_rewrite.rs](../../tests/rust/direct_rewrite.rs)가 생성된 ISA와 연결한 검사를 제공하며,
[실험 실행 안내](../../experiments/multipattern/README.md)에서 세 구성을 비교할 수 있다.
