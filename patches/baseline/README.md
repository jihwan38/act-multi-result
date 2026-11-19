# baseline: 공통 실행·정확성 수정

[인수인계](../../handoff/README.md) · [전체 구성](../../handoff/README.md#5-패치-폴더는-각각-무엇인가) · [적용 목록](series)

고정 upstream에 가장 먼저 적용하는 Backend 3개·Oracle 1개다.
`baseline`만 선택하면 QKV를 실행할 수 있고, 나머지 구성도 이 수정 위에서 시작한다.
다중 결과를 추가하는 패치는 아니다.

## 개별 패치

| 패치 | 바꾸는 동작과 실제 수정 위치 |
| --- | --- |
| [Backend 0001 — use-configurable-native-dependency-roots](act-backend/0001-use-configurable-native-dependency-roots.patch) | `act_backend/generic/build.rs`, `cpp/malloc/CMakeLists.txt`. OR-Tools·nlohmann JSON의 설치 위치를 `ORTOOLS_ROOT`, `NLOHMANN_JSON_ROOT`로 받는다. Rust 링크 경로·실행 시 라이브러리 경로와 C++ 빌드 경로를 맞추고 필요한 파일이 없으면 실패시킨다. |
| [Backend 0002 — honor-explicit-hlo-root](act-backend/0002-honor-explicit-hlo-root.patch) | `generic/src/isel/initializer/parser.rs`, `lib.rs`. 마지막 줄을 반환값으로 간주하던 처리를 명시적 `ROOT` 기준으로 바꾼다. `HloInstruction.is_root`를 파서에서 e-graph 초기화까지 전달하고, 출력 저장을 만드는 과정에서 원래 논리 값을 덮어쓰지 않는다. |
| [Backend 0003 — enforce-declared-address-constraints](act-backend/0003-enforce-declared-address-constraints.patch) | `act_backend/address_constraints.py`의 `ConstraintLowerer`, `malloc_instructions_h_generator.py`, C++ `address_constraints.h`와 명령 템플릿. 명세의 주소 조건을 OR-Tools 제약으로 생성하고 주소 변수의 범위를 저장 공간에 맞춘다. 선언한 정렬·나머지·주소 관계가 실제 할당에 반영되게 한다. |
| [Oracle 0001 — preserve-broadcast-dimensions](act-oracle/0001-preserve-broadcast-dimensions.patch) | `taidl_to/oracle_visitor.py`, `templates/templates.txt`. broadcast의 명시적 `dimensions`를 생성된 HLO까지 전달한다. 출력 shape로 축을 추측하지 않고, 입력 개수나 축 속성이 잘못되면 생성 단계에서 거절한다. |

Backend 0003은 정수 산술·비교·논리식과 양수 modulus를 지원한다. 알 수 없는 변수나
지원하지 않는 식은 조용히 버리지 않고 오류로 처리한다. 새 제약 문법을 추가할 때는
파싱뿐 아니라 solver의 정수 범위·음수 나머지 의미도 확인해야 한다.
Backend 0002는 ENTRY에 명시적 ROOT가 정확히 하나 있어야 한다.

## 적용된 소스에서 보기

[소스 준비 안내](../../README.md#패치가-적용된-소스-읽기와-수정)에 따라
`python scripts/prepare_sources.py baseline`을 실행한다.
위 경로는 `build/baseline/src/act-backend/` 또는 `build/baseline/src/act-oracle/` 아래에서 찾는다.
다른 구성은 이후 패치가 같은 파일을 더 바꿀 수 있으므로 그 구성의 적용 결과도 함께 확인한다.

공통 패치를 수정하면 모든 구성에 영향을 준다. 대상 동작의 기존 회귀 검사와 영향을 받는
실행 구성을 [검사 안내](../../README.md#선택-사항-명령-묶어서-실행하기와-회귀-테스트)에 따라 확인한다.
