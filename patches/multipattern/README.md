# multipattern: 여러 결과의 후보를 함께 검색

[인수인계](../../handoff/README.md) · [비교 보고서](../../experiments/multipattern/REPORT.md) · [적용 목록](series)

현재 다중 결과 구현의 **공동 후보 발견 부분**을 바꾸는 Backend 패치 2개다.
**고정 upstream → baseline → multi-result → multipattern** 순서로 적용한다.
결과별 검색 후 조합을 열거하던 대신, 여러 결과 패턴과 같은 입력을 가리켜야 한다는 조건을
`egg::MultiPattern`으로 함께 검색한다. 찾은 tuple 후보를 등록하고 ISA rewrite가 다시
검색·적용하는 구조는 유지한다. 추출·주소 할당·Oracle은 `multi-result`의 경로를 사용한다.

## 개별 패치

| 패치 | 바꾸는 동작과 실제 수정 위치 |
| --- | --- |
| [Backend 0001 — direct-discovery](act-backend/0001-direct-discovery.patch) | `act_backend/generic/src/isel/discovery.rs`. `append_pattern()`, `joint_query()`, `joint_matches()`로 실제 e-graph를 공동 검색한다. 연산자별 child 변수와 입력 동등성 조건을 사용해 공유·반복 입력 및 shape·metadata wildcard를 보존한다. 검색 결과는 기존 후보 검사·등록 경로에 넘긴다. |
| [Backend 0002 — bounded-discovery](act-backend/0002-bounded-discovery.patch) | 같은 `discovery.rs`를 `JointQuery`, `InstructionStream`, `Discovery::refresh_until()`로 재구성하고 `isel/bounded_search.rs`와 `pipeline.rs`를 연결한다. 원시 매칭 결과 수·검색 호출 수·조합 검사 시도 수에 한도를 두며, ISA를 번갈아 검색하고 등록 목표에 도달하면 중단한다. |

0001은 공동 검색 방식의 도입이고 0002는 그 검색량과 후보 배분을 제어하는 후속 수정이다.
따라서 최종 소스에서는 0001의 일부 함수가 0002의 구조로 바뀌어 있다.
10월 26일 비교 보고서의 `multipattern`은 **두 패치를 모두 적용한 구성**이다.

## 검색 한도의 의미와 남은 한계

`SearchBudget`은 원시 결과를 다시 읽는 비용과 실패한 검색도 센다.
이 작업량 한도는 refresh마다 새로 시작하지만, 등록 가능한 tuple 수는 누적 한도다.
그래프가 합쳐져 후보 수가 줄어도 이미 쓴 등록 예산을 되돌려 주지 않는다.
마감 시간은 검색 호출 사이에서 확인하므로 한 번의 egg 검색을 도중에 강제로 끊지는 못한다.

공유 입력·세 결과·여러 ISA의 경쟁에서 후보 배분은 달라졌지만, 전체 컴파일과 최종 코드 비용이
전반적으로 좋아지지는 않았다. 큰 triple 입력의 메모리도 늘었다.
상세 수치와 한도별 비교는 [보고서](../../experiments/multipattern/REPORT.md)에 있다.

## 적용된 소스와 확인 경로

`python scripts/prepare_sources.py multipattern`으로 준비하면
`build/multipattern/src/act-backend/act_backend/generic/src/isel/`에서 최종 코드를 읽을 수 있다.
실제 변경 보존은 [소스 수정 안내](../../README.md#패치가-적용된-소스-읽기와-수정)를 따른다.

검색기를 수정하면 공유·반복 입력의 동일성, 결과 순서, 타입·shape 조건, 한도 소진 후의
대체 경로를 확인한다. [ProbeISA.py](../../experiments/multipattern/ProbeISA.py)와
[고정 그래프 검사](../../experiments/multipattern/discovery_probe.rs)가 해당 비교를 제공한다.
전체 생성·실행 비교는 [실험 README](../../experiments/multipattern/README.md)에 있다.

검색과 적용까지 하나의 rewrite로 합친 후속 비교는
[multipattern-rewrite](../multipattern-rewrite/README.md)다.
