# Gemmini용 이전 다중 결과 구현

[인수인계](../../handoff/README.md) · [전체 구성](../../handoff/README.md#5-패치-폴더는-각각-무엇인가) · [적용 목록](series)

`act-parallelism`의 [실험 05, commit 6164b44](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/patches)에서
가져온 **Backend 9개 + Oracle 3개**다. 패치 내용은 원본과 같다.

적용 순서는 **고정 upstream → `baseline` → `multi-result-legacy`**다.
현재 `multi-result` 패치 위에 추가하는 구성이 아니다. 네 ACT upstream의 고정 커밋과
`baseline` 패치는 이전 저장소와 같아서 중복 복사하지 않는다.

## 개별 패치

아래 Backend 경로는 `act_backend/`, Oracle 경로는 `taidl_to/`를 기준으로 한다.
현재 `multi-result`의 파일 번호와 일대일로 대응하는 목록은 아니다.

| 패치 | 바꾸는 동작과 실제 수정 위치 |
| --- | --- |
| [Backend 0001 — support-tuple-root-frontend](act-backend/0001-support-tuple-root-frontend.patch) | `generic/src/ir/egraph.rs`, `generic/src/isel/initializer/{parser,lib}.rs`. 입력 HLO의 tuple 반환을 읽고 e-graph의 `OpTuple`과 복수 출력 저장 위치를 만든다. ISA 명세의 ROOT 문법을 바꾸는 패치와 구분한다. |
| [Backend 0002 — support-get-tuple-element](act-backend/0002-support-get-tuple-element.patch) | e-graph·initializer·`generic/src/isel/rewrites/tuple.rs`. `get-tuple-element`와 `tuple_elements()`, `projection_rule()`로 tuple의 특정 결과를 참조하고 그 결과 값에 연결한다. |
| [Backend 0003 — model-multi-result-instructions](act-backend/0003-model-multi-result-instructions.patch) | `generic/src/ir/{buffer,pii}.rs`, `ir_buffer_rs_generator.py`, extractor·pipeline. `BufferSignature`, `ValueRef`, `PiiResult`로 한 명령의 여러 결과를 나타낸다. 기존 추출기를 새 구조에 연결하되 이 단계만으로 다중 결과 할당을 완성하지는 않는다. |
| [Backend 0004 — complete-typed-tuple-semantics](act-backend/0004-complete-typed-tuple-semantics.patch) | `semantics.py`, rewrite 생성기, `generic/src/ir/{value,metadata}.rs`, initializer·applier. 의미식을 결과별 패턴으로 만들고 타입·shape 검사를 생성한다. HLO의 중첩 입출력 구조와 출력 metadata도 연결한다. |
| [Backend 0005 — discover-multi-result-candidates](act-backend/0005-discover-multi-result-candidates.patch) | `generic/src/isel/discovery.rs`, `generic/src/pipeline.rs`. 결과별 매칭의 조합을 검사해 tuple/GTE 후보를 등록하고 e-graph 반복 중 다시 찾는다. `Derivations`로 출력에 의존하지 않는 입력 계산 경로가 있는지 검사하며, ISA 노드는 생성된 rewrite가 추가한다. |
| [Backend 0006 — support-multi-result-fast-slow](act-backend/0006-support-multi-result-fast-slow.patch) | `generic/src/isel/extractor/{fast,slow,utils,lib}.rs`. 여러 프로그램 출력에서 필요한 생산자·결과를 선택한다. `Selection`이 이미 선택한 생산자의 결과 재사용, 출력 주소 조건, 실패한 선택의 복구를 처리한다. fast 탐색은 처음 성공한 입력 경로를 유지하므로 완전 탐색은 아니다. |
| [Backend 0007 — allocate-multiple-results-in-act-cpp](act-backend/0007-allocate-multiple-results-in-act-cpp.patch) | `generic/cpp/malloc/`, 관련 C++ 생성기·템플릿, `generic/src/pipeline.rs`. 다중 결과 PII를 읽고 결과마다 Tensor·수명·주소를 만든 뒤 모든 출력 인자를 호출문에 쓴다. 여러 할당 후보 간 상태 정리와 미평가 후보가 있을 때의 조기 중단 보류도 포함한다. |
| [Backend 0008 — correct-signed-integer-dtype-names](act-backend/0008-correct-signed-integer-dtype-names.patch) | `isel_ir2isa_rewrites_rs_generator.py`. HLO의 signed 정수 타입을 존재하지 않는 `S8/S32` 대신 Rust IR의 `I8/I32`로 생성한다. |
| [Backend 0009 — support-integer-saturation-semantics](act-backend/0009-support-integer-saturation-semantics.patch) | `semantics.py`, rewrite 생성기, e-graph·initializer. 정수 상수와 `minimum`·`maximum`을 의미식·검색 패턴에 연결한다. Gemmini의 INT32 → INT8 포화 저장과 다음 GEMM 입력을 표현하는 데 사용한다. |
| [Oracle 0001 — support-two-input-concatenate](act-oracle/0001-support-two-input-concatenate.patch) | `oracle_visitor.py`, `templates/templates.txt`. 두 입력 concatenate와 명시한 축을 생성된 HLO에 전달한다. 지원하는 입력 개수는 둘이다. |
| [Oracle 0002 — preserve-structured-kernel-io](act-oracle/0002-preserve-structured-kernel-io.patch) | `generic/decorator.py`. kernel의 tuple 입력을 tensor 목록으로 펼치고 반환값을 원래 구조로 복원한다. tree의 tensor 참조가 유효한지도 검사한다. |
| [Oracle 0003 — forward-constant-literals](act-oracle/0003-forward-constant-literals.patch) | `oracle_visitor.py`, `templates/templates.txt`. 상수 literal을 주소·매개변수로 해석하지 않고 생성된 HLO의 `constant(...)`에 넘긴다. Backend 0009와 함께 포화 의미식의 기능 실행을 가능하게 한다. |

## 현재 구현과의 경계

이 버전의 `GemminiDual.py`는 두 `ROOT`를 쓰는 옛 문법을 유지한다.
현재 `multi-result`의 `ROOT tuple(...)` 문법과 섞지 않는다.
이전 실험 보존을 위한 구성이며, 현재 다중 결과 구현으로 Gemmini를 이식한 것은 아니다.

현재 구현으로 옮기려면 필요한 정수 의미 지원을 현재 생성기·공유 입출력 해석에 맞게 이식한다.
legacy 전체를 현재 `multi-result` 뒤에 적용하면 같은 파일의 서로 다른 구현이 충돌한다.
또한 이 묶음은 기능·주소 처리를 위한 것으로, 두 엔진의 병렬 발행 자체는
[native 함수](../../cases/gemmini/native/gemmini_dual.h)의 `act_double_gemm()`에서 수행한다.

## 적용된 소스와 실행

`python scripts/prepare_sources.py multi-result-legacy`로 준비하면
`build/multi-result-legacy/src/{act-backend,act-oracle}/`에서 최종 구현을 읽을 수 있다.
[소스 수정 안내](../../README.md#패치가-적용된-소스-읽기와-수정)에 따라 변경을 보존한다.

실행은 [cases/gemmini](../../cases/gemmini/README.md), 하드웨어 관측 패치는
[hardware/patches](../../cases/gemmini/hardware/patches)에 있다.
기능·계산 겹침·cycle 측정의 범위는 [결과 보고서](../../cases/gemmini/RESULTS.md)에 있다.
