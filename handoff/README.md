# ACT 연구 인수인계

2026-11-20 정리. 이 문서는 연구 목표·수정 흐름·실험에서 확인한 것과 남은 한계를 설명한다.
실행법은 [저장소 README](../README.md), 개별 수정은 아래의 [패치 폴더 안내](#5-패치-폴더는-각각-무엇인가)에서 찾아볼 수 있다.

## 1. 무엇을 하려 했고 어디까지 됐나

목표는 **독립적인 계산 여러 개를 한 명령으로 묶어 ACT가 선택하게 하는 것**이었다.
대표 사례는 `C0 = A0 @ B0`, `C1 = A1 @ B1`을 함께 수행하고 결과 두 개를 돌려주는
`double_gemm`이다. 이런 명령으로 두 계산 유닛을 함께 사용할 수 있을지 살펴봤다.
이를 위해 구현한 것이 **한 명령의 여러 결과를 끝까지 구분해 처리하는 multi-result 지원**이다.

| 연구 질문 | 현재까지 확인한 것 |
| --- | --- |
| 보통의 HLO 계산 두 개를 DoubleGEMM으로 선택할 수 있는가? | BF16 64×64 예제에서 후보 발견 → 명령 선택 → 결과별 주소 할당 → 코드 생성 → 기능 검증까지 연결했다. |
| 공동 후보를 더 직접적으로 찾으면 좋아지는가? | `multi-result`, `multipattern`, `multipattern-rewrite`를 비교했다. 검색 제어와 후보 배분은 바꿨지만, 전반적인 컴파일 성능·최종 코드 비용의 우위는 확인하지 못했다. |
| 선언한 공동 명령이 실제 가속기 명령으로 실행 가능한가? | Dual Gemmini RTL 시뮬레이션에서 두 엔진의 계산 겹침을 확인했다. 다만 이전 구현을 보존한 `multi-result-legacy`의 결과다. |

현재 코드와 비교 자료는 `main`에 모여 있다. QKV는 `baseline`, BF16 DoubleGEMM은
`multi-result`, Gemmini는 `multi-result-legacy`로 실행한다. **현재 다중 결과 구현과 Gemmini용
이전 구현은 별도 구성**이며, 현재 구현을 Gemmini까지 이식한 상태는 아니다.

## 2. DoubleGEMM을 위해 무엇을 바꿨나

초기에는 “ISA와 전용 버퍼를 두 벌 선언하면 ACT가 계산을 두 유닛에 나눠 줄까?”를 확인했다.
당시 attention 입력에서는 한쪽 유닛만 선택됐다. 다음에는 두 유닛의 결과를 함께 받아 저장하는
`join_store`를 넣어 두 유닛을 쓰는 명령열을 만들었다. 이 단계는 명세의 저장 경로로 두 유닛 사용을
유도한 것이며, ACT가 실행시간을 비교해 최적 배치를 찾았다는 뜻은 아니다.

그다음 질문이 **“두 GEMM을 함께 수행하는 DoubleGEMM 자체를 명령 선택 대상으로 넣을 수 있는가?”**였다.
기존의 단일 결과 중심 처리에서는 한 명령과 그 결과 값을 거의 같은 단위로 다뤘다.
DoubleGEMM은 한 번 실행하지만 `C0`, `C1`은 다른 버퍼에 놓이고, 서로 다른 후속 명령이
서로 다른 시점까지 사용할 수 있다. 두 값을 단순히 하나의 큰 tensor로 합치는 것만으로는
이 관계를 표현할 수 없어 명령 선택 이후의 처리까지 바꿨다.

현재 [DoubleGEMM 명세](../cases/double_gemm/QKVDoubleGemm.py)와
[입력 HLO](../cases/double_gemm/workloads/two_gemm_concat.hlo)의 흐름은 다음과 같다.

```text
입력 HLO: 일반 dot 두 개 → 두 결과를 concatenate
    ↓ 명세의 두 결과 의미식에 맞는 계산 조합을 찾음
double_gemm 후보: 명령 하나, 결과 0과 결과 1
    ↓ 다른 단일 GEMM 후보들과 함께 명령열을 선택
결과별 버퍼·주소·마지막 사용 시점 결정
    ↓
double_gemm(..., addr_out0, addr_out1) → join_store(...)
    ↓
생성된 호출문을 Oracle로 실행하고 원래 HLO의 실행 결과와 비교
```

입력 HLO에 `double_gemm` 호출을 미리 넣지 않는다. 떨어져 있는 두 `dot`을 명세의 공동 계산으로
찾아내야 한다. `double_gemm`의 결과는 끝까지 두 개이며, 마지막 `join_store`가 이를 합쳐 저장한다.

| 수정한 부분 | 필요한 이유 |
| --- | --- |
| 명세의 입출력 해석 | `parameter(index)`와 `ROOT tuple(...)`로 입력·결과의 순서를 정한다. Backend와 Oracle이 같은 순서를 써야 결과가 뒤바뀌지 않는다. |
| 공동 후보 발견과 검사 | 결과별 계산을 찾아 조합하고, 타입·shape·입력 관계가 명세에 맞는지 검사한다. 서로의 결과를 입력으로 쓰는 계산을 독립 DoubleGEMM으로 잘못 묶지 않아야 한다. |
| 명령과 결과 값의 분리 | 중간 표현에서 `ValueRef(node, result)`로 “어느 명령의 몇 번째 결과인지”를 기록한다. 두 결과를 쓴다고 생산 명령을 두 번 실행해서는 안 된다. |
| 추출·할당·코드 생성 | 한 생산자의 여러 결과를 재사용하고, 각 결과의 주소와 수명을 따로 계산해 출력 인자에 전달한다. 선택이나 할당에 실패하면 다른 후보를 탐색할 수 있어야 한다. |
| 기능 검증 | 생성한 호출문의 결과를 원래 HLO의 XLA 실행 결과와 비교한다. 프로그램 입출력의 tuple 구조와 ISA 명령 내부의 여러 결과도 구분해 보존한다. |

이 결과별 참조·추출·할당 경로는 다른 다중 결과 명령에도 사용할 수 있게 구성했다.
비교 실험에는 두 결과·세 결과, 서로 다른 산술 연산, 공유·반복 입력도 포함했다.
다만 BF16 예제의 버퍼와 명령은 실험용 명세이며, CPU Oracle의 정답 일치만으로 실제 장치의
동시 실행이나 속도 향상을 알 수는 없다. 그래서 검색 방식과 Gemmini 실행을 따로 살펴봤다.

## 3. 검색 실험: MultiPattern과 multipattern-rewrite

여러 결과를 묶으려면 “첫 번째 결과가 될 수 있는 계산”과 “두 번째 결과가 될 수 있는 계산”을
함께 찾아야 한다. 각각 찾고 조합을 열거하면 중간 조합이 커지거나 앞쪽 명령이 검색 기회를
많이 차지할 수 있다. 이 부분을 바꾸기 위해 세 구현을 비교했다.

| 구성 | 후보를 찾고 ISA로 연결하는 방식 |
| --- | --- |
| `multi-result` | 결과별로 매칭한 뒤 조합을 검사한다. 적합한 tuple 후보를 등록하고 ISA rewrite가 이를 다시 찾아 적용한다. |
| `multipattern` | 여러 결과의 패턴과 공유 입력 조건을 MultiPattern으로 함께 검색한다. tuple 후보 등록 후 ISA rewrite를 적용하는 단계는 유지한다. 검색량에 한도를 두고 ISA를 번갈아 검색한다. |
| `multipattern-rewrite` | 생성된 rewrite 내부에서 공동 검색·조건 검사·ISA 적용을 처리한다. 찾은 바인딩을 적용에 사용하고, 성공할 때 tuple과 결과 참조를 만든다. 별도 discovery hook을 거치지 않는다. |

**`multipattern-rewrite`는 2026-10-26의 [커밋 3b85d5d](https://github.com/jihwan38/act-multi-result/commit/3b85d5dad68867fb6768d355ecba85876fcd5685)에서
추가한 세 번째 비교 구현**이다. MultiPattern discovery는 10월 5일부터 있었고,
10월 26일에 검색 한도·후보 배분을 보강한 뒤 직접 rewrite 방식까지 비교했다.
현재 모두 `--toolchain`으로 선택할 수 있으며, 기본 `multi-result`를 대체하지 않았다.

관찰은 두 가지를 함께 봐야 한다.

- **검색 방식의 변화는 있었다.** 여러 ISA에 검색 기회를 나누고 검색량을 제한했으며,
  직접 rewrite에서는 후보를 등록한 뒤 다시 검색하는 단계를 합쳤다. 고정 그래프 검사에서
  기존 방식이 pair 후보에만 예산을 쓰던 경우, 새 방식들은 triple 후보도 찾았다.
- **그 변화가 전반적인 성능 향상으로 이어지지는 않았다.** 기본 HLO 입력 14개의 최종 코드 비용은
  세 방식이 같았다. `shared_triples_16`의 최대 메모리 중앙값은 210.68 MiB에서
  286.09 / 287.95 MiB로 약 36% 늘었다. `shared_pairs_16`에서는 새 두 방식이 공동 후보/적용
  16개를 준비했지만 최종 코드에는 공동 명령 두 개만 선택됐다.

여기서 비용은 **assembly 파일의 줄 수**이며 장치 실행시간이 아니다.
컴파일 시간에는 공동 검색뿐 아니라 명령열 추출·주소 할당·비용 평가도 포함된다.
따라서 후보를 많이 찾았는데 선택이 적다는 사실만으로 검색기나 할당기 하나를 원인으로 단정할 수 없다.
어떤 후보를 탐색했는지, 어디서 거절했는지, 비용 비교에서 밀렸는지를 이어서 추적할 필요가 있다.
직접 rewrite의 ISA별 고정 예산도 일치하지 않는 ISA의 남은 몫을 다른 ISA에 넘기지 못하는 한계가 있다.

세 방식의 차이·측정 조건·상세 결과는 [MultiPattern 보고서](../experiments/multipattern/REPORT.md)에 있다.

## 4. 실제 가속기에서도 쓸 수 있는가: Dual Gemmini

이 실험에서는 **DoubleGEMM이라는 공동 명령을 실제 가속기의 명령 인터페이스로 구현하고,
ACT가 고른 호출문을 그대로 실행할 수 있는지**를 확인했다.
구성은 Rocket CPU 하나와 Gemmini 두 개, 엔진별 scratchpad·accumulator, 공유 DDR3다.
입력은 16×16 INT8, GEMM 결과는 INT32이며, Verilator로 SoC RTL을 실행했다.

여기서 `double_gemm`은 새 하드웨어 opcode가 아니다.
[native 함수](../cases/gemmini/native/gemmini_dual.h)의 `act_double_gemm()`이
두 엔진에 각각 preload·compute를 보낸 뒤 함께 완료를 기다리는 소프트웨어 명령 묶음이다.
반면 개별 `act_gemm()`은 한 엔진에 보낸 뒤 바로 기다린다. 공동 명령 안에서 두 계산 사이의
대기를 없애 실제 겹침이 생기는지 확인하는 구조다.
[변환기](../cases/gemmini/emit_native.py)는 ACT가 정한 명령 종류·엔진·주소를 native 호출로 옮기며,
개별 GEMM을 사후에 다시 묶어서 병렬화하지 않는다.

| 입력 | ACT가 선택한 것 | 이 결과에서 알 수 있는 것 |
| --- | --- | --- |
| 독립 두 GEMM | DoubleGEMM 한 개 | dense 입력에서 두 엔진의 PE(0,0) 곱셈 활동이 같은 cycle에 나타났다. 호출만 두 개 발행한 수준을 넘어 계산 겹침을 관측했다. |
| 입력 X를 공유하는 두 GEMM | 개별 GEMM 두 개 | 입력을 공유해도 두 계산은 독립적이며 수동 병렬 실행이 가능했다. ACT는 이 기회를 공동 명령 선택으로 연결하지 못했다. |
| 앞 결과를 다음 입력으로 쓰는 GEMM | 개별 GEMM | 계산 의존성이 있으므로 동시에 묶지 않는 것이 맞다. INT32 결과를 INT8로 포화 변환해 다음 계산에 사용한다. |
| 독립 네 GEMM | 한 쌍만 DoubleGEMM | 수동 구현은 두 쌍을 묶지만 ACT는 한 쌍만 묶었다. 여러 계산에 대한 공동 명령 선택을 충분히 활용한 상태는 아니다. |

2026-11-20 재실행에서 독립 두 GEMM의 재호출 cycle 중앙값은 순차 592, 수동 병렬 511,
ACT 512였다. 같은 조건에서 독립 네 GEMM은 수동 병렬 1027, ACT 1110이었다.
이는 입력 준비·정답 검사를 제외하고 **load·compute·store·완료 대기**를 포함한 값이다.
한 사례에서 수동 병렬과 비슷한 결과가 나왔어도, 계산을 더 많이 주면 자동으로 같은 수준의
병렬 명령열을 찾는다고 볼 수는 없다. 첫 호출과 재호출의 수치는 [결과 보고서](../cases/gemmini/RESULTS.md)에 구분했다.

이 실험은 원래 `act-parallelism/experiments/05_dual_gemmini`에서 진행했다.
현재 저장소에는 그 패치·명세·native 실행 경로를 `multi-result-legacy`로 가져왔고,
compiler·Oracle·native 프로그램을 다시 생성해 4개 workload × 입력 6종 × 실행 방식 3종,
총 72회 RTL 실행·144회 결과 검사를 통과했다. 기존 RTL 실행 파일은 재사용했다.
**현재 `multi-result`의 하드웨어 검증이나, 빈 환경에서의 RTL 재빌드, FPGA·실칩 실행을 완료한 것은 아니다.**

## 5. 패치 폴더는 각각 무엇인가

각 패치 폴더는 **선택한 실행 구성을 만드는 변경 묶음**이다.
폴더 이름을 누르면 개별 패치의 역할·수정 파일·읽을 순서를 볼 수 있다.

| 폴더 | 역할 | 적용하는 바탕 |
| --- | --- | --- |
| [baseline](../patches/baseline/README.md) | 공통 실행·정확성 수정: native 의존성 경로, 명시적 HLO ROOT, 주소 제약, Oracle broadcast | 고정 upstream. 모든 구성에 공통 적용 |
| [multi-result](../patches/multi-result/README.md) | 현재 다중 결과 구현: 명세 해석부터 후보 발견·추출·주소 할당·Oracle까지 | `baseline` |
| [multi-result-legacy](../patches/multi-result-legacy/README.md) | 이전 Dual Gemmini 구현 보존. 옛 다중 결과 처리와 정수 포화 의미 지원 | `baseline`만 적용. 현재 `multi-result`와 별도 |
| [multipattern](../patches/multipattern/README.md) | 공동 후보 검색을 MultiPattern으로 바꾸고 검색 한도·ISA별 기회 제어 | `baseline` → `multi-result` |
| [multipattern-rewrite](../patches/multipattern-rewrite/README.md) | 공동 검색과 적용을 생성된 rewrite 안으로 옮긴 비교 구현 | `baseline` → `multi-result` → `multipattern` |

실제 조합은 [TOOLCHAINS](../scripts/_internal/common.py), 각 묶음의 파일 목록은 해당 폴더의 `series`에 있다.
`act-backend/`, `act-oracle/`, `taidl/` 하위 폴더는 **수정 대상 upstream 저장소**다.
`third_party/`는 고정 원본, `patches/`는 보존하는 변경, `build/<구성>/src/`는 적용된 소스다.
코드를 이어서 고칠 때의 준비·보존 방법은 [소스 수정 안내](../README.md#패치가-적용된-소스-읽기와-수정)에 있다.

## 6. 남은 한계와 이어서 볼 질문

**다중 결과의 기능 지원과 실행 성능 최적화는 아직 간격이 있다.**
명령 하나의 결과 둘을 각각 할당·사용할 수 있게 만든 부분은 재사용할 수 있다.
반면 어떤 계산을 묶어야 이득인지, 몇 쌍을 어느 엔진에 놓을지, 언제 데이터를 옮길지를
함께 최적화하는 문제는 남아 있다.

- **공동 명령의 실행 방법은 명세 작성자가 제공해야 한다.** DoubleGEMM의 수학적 의미를 선언해도
  하드웨어가 저절로 병렬 실행하지는 않는다. Gemmini에서는 기존 명령을 묶는 native 함수를 만들었다.
  다른 가속기에는 그 장치의 명령 발행·완료 대기·버퍼 규칙에 맞는 구현이 필요하다.
- **현재 스케줄러는 시간표를 만들지 않는다.** 의존 관계와 중간값의 메모리 점유를 기준으로 순서를
  정하지만 엔진 점유·명령 지연시간·DMA 동작을 모델링하지 않는다. 예를 들어 현재 타일을 계산하는 동안
  다음 타일을 미리 읽으려면 비동기 이동과 대기, 버퍼 재사용 시점까지 표현해야 한다.
  Gemmini 경로도 이동 명령마다 완료를 기다리므로 일반적인 이동·계산 중첩을 검증하지 않았다.
- **타일 안의 명령 선택과 여러 커널의 배치는 다른 문제다.** 이번 실험은 작은 고정 타일의 계산을
  대상으로 했다. 모델을 여러 커널로 나누고 장치에 배치하며 커널 사이의 메모리·동기화를 관리하는
  전체 모델 실행은 구현 범위에 없다. PE 일부의 겹침을 전체 array 활용률이나 모델 속도 향상으로 확대할 수 없다.
- **현재 구현으로 Gemmini를 돌리는 작업은 남아 있다.** legacy 명세의 두 `ROOT`를 현재의
  `ROOT tuple(...)`로 바꾸고, 정수 상수·`minimum`·`maximum`을 이용한 포화 변환을 현재 의미 변환기에
  맞춰야 한다. 이전 패치를 그대로 현재 패치 뒤에 붙이는 방식으로 합칠 수는 없다.

이어간다면 우선 연구 질문을 고르면 된다. 다중 결과 자체의 효용이 관심이라면 최대값과 그 위치를
함께 반환하는 연산처럼 **장치가 실제로 여러 결과를 제공하는 사례**를 찾아볼 수 있다.
이것은 다음 사례의 제안이며 구현된 max·argmax 지원을 뜻하지 않는다.
병렬 실행이 관심이라면 공유 입력·독립 네 GEMM에서 공동 명령이 덜 선택된 이유를 추적하고,
명령 수 대신 실행시간과 자원을 반영하는 선택 방법을 살펴볼 수 있다.

별도의 연결 아이디어로는 [MATCHA](https://arxiv.org/abs/2604.09124v1)가 있다.
MATCHA는 여러 이기종 가속기가 있는 SoC에서 DNN의 배치·메모리 할당·병렬 실행 순서를 다룬다.
연결한다면 ACT가 각 계산 구간의 명령열을 만들고 MATCHA가 구간들의 배치와 시작·완료를 관리하는
역할 분담을 생각할 수 있다. 입출력·실행 인터페이스 연결이 추가로 필요하며,
**이 저장소의 구현이나 실험 성과에는 포함되지 않는다.**

## 7. 필요한 자료만 찾아보기

| 필요한 것 | 자료 |
| --- | --- |
| 설치·실행·회귀 검사 | [저장소 README](../README.md) |
| 검색 실험의 상세 결과 / 재실행 | [MultiPattern 보고서](../experiments/multipattern/REPORT.md) / [실행 안내](../experiments/multipattern/README.md) |
| Gemmini 결과 / 실행 / native 연결 | [결과 보고서](../cases/gemmini/RESULTS.md) / [실행 안내](../cases/gemmini/README.md) / [native 설명](../cases/gemmini/docs/native-interface.md) |
| ISA 복제·join·DoubleGEMM으로 이어진 이전 실험 | [act-parallelism의 실험들](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments) |

배경은 [ACT](https://arxiv.org/abs/2510.09932v2)와 [TAIDL](https://doi.org/10.1145/3725843.3756075),
검색을 이어서 볼 때는 [egg](https://arxiv.org/abs/2004.03082)와
[Relational E-Matching](https://arxiv.org/abs/2108.02290v2)이 연결된다.
위 MATCHA는 여러 커널·장치 수준의 문제를 별도로 살펴볼 때의 참고 자료다.

`handoff-20261118` 태그는 기존 다중 결과·검색 비교의 코드 기준이며 이후 가져온 Gemmini legacy는
포함하지 않는다. MultiPattern 보고서의 `main`도 측정 당시 구현 `10f059e`를 뜻한다.
세 측정 버전은 보고서의 `comparison-20261026-*` 태그로 보존했다.
