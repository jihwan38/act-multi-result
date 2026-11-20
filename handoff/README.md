# ACT 연구 인수인계

2026-11-20 정리. 연구 목표·결과·한계는 이 문서에, 실행법은 [저장소 README](../README.md)에 있다.
개별 수정은 [패치 폴더 안내](#5-패치-폴더는-각각-무엇인가)에서 찾아본다.

## 1. 무엇을 하려 했고 어디까지 됐나

목표는 **독립 계산 여러 개를 한 명령으로 묶어 ACT가 선택하게 하는 것**이었다.
대표 사례인 `double_gemm`은 `C0 = A0 @ B0`, `C1 = A1 @ B1`을 함께 수행해 결과 두 개를 낸다.
이를 위해 한 명령의 여러 결과를 구분하는 **multi-result 지원**을 구현했다.

| 작업 | 현재 상태 |
| --- | --- |
| 현재 다중 결과 구현 | BF16 64×64 DoubleGEMM에서 후보 발견 → 명령 선택 → 결과별 주소 할당 → 코드 생성 → 기능 검증까지 연결 |
| 검색 방식 비교 | `multi-result`, `multipattern`, `multipattern-rewrite` 비교 완료. 전반적인 성능 우위는 확인하지 못함 |
| Dual Gemmini | 이전 구현인 `multi-result-legacy`로 RTL 기능과 두 엔진의 동시 계산 확인. 현재 `multi-result`로의 이식은 남음 |

코드와 비교 자료는 `main`에 있다. 기본 실행은 QKV=`baseline`, BF16 DoubleGEMM=`multi-result`,
Gemmini=`multi-result-legacy`다.

## 2. DoubleGEMM을 위해 무엇을 바꿨나

초기 attention 실험에서는 ISA와 전용 버퍼를 두 벌 선언해도 한쪽 유닛만 선택됐다.
두 유닛의 결과를 요구하는 `join_store`로 두 유닛 사용을 유도한 뒤,
**두 GEMM을 묶은 명령 자체를 선택하게 만드는 것**으로 확장했다.
`join_store` 때문에 두 유닛을 쓴다는 사실만으로 최적 배치를 찾았다고 볼 수는 없다.

DoubleGEMM은 한 번 실행하지만 `C0`, `C1`은 다른 버퍼에 놓이고 서로 다른 시점까지 사용될 수 있다.
따라서 명령 생산자와 결과 값을 분리해야 했다. 두 결과를 쓴다고 생산 명령을 두 번 실행하거나,
한쪽 결과의 마지막 사용 시점에 다른 결과까지 해제해서는 안 된다.

| 수정 | 역할 |
| --- | --- |
| 명세 입출력 | `parameter(index)`와 `ROOT tuple(...)`로 순서를 정하고 Backend·Oracle이 함께 사용 |
| 공동 후보 발견 | 결과별 계산을 찾아 타입·shape·입력 관계를 검사. 서로 의존하는 계산을 독립 DoubleGEMM으로 잘못 묶지 않음 |
| 결과 표현·추출 | `ValueRef(node, result)`로 생산자와 결과 번호를 기록하고 한 생산자의 여러 결과를 재사용 |
| 할당·코드 생성 | 결과별 버퍼·주소·마지막 사용 시점을 계산해 각각의 출력 인자로 전달 |
| 기능 검증 | 생성된 호출문의 Oracle 결과를 원래 HLO의 XLA 실행 결과와 비교 |

[대표 HLO](../cases/double_gemm/workloads/two_gemm_concat.hlo)는 일반 `dot` 두 개와 concatenate다.
미리 DoubleGEMM 호출을 넣지 않으며, [명세](../cases/double_gemm/QKVDoubleGemm.py)에 맞는 조합을 찾아
`double_gemm(..., addr_out0, addr_out1)`과 `join_store(...)`를 생성한다.
비교 실험에는 두 결과·세 결과, 서로 다른 산술 연산, 공유·반복 입력도 포함했다.
다만 CPU Oracle의 정답 일치로는 장치의 동시 계산 여부나 실행속도를 알 수 없다.

## 3. 검색 실험: MultiPattern과 multipattern-rewrite

진행 흐름은 **다중 결과 기반(`multi-result`) → 공동 검색 도입(`multipattern`) →
rewrite 통합(`multipattern-rewrite`)**이다. 최종적으로 원했던 것은 공동 검색과 조건 검사·명령 적용을
한 rewrite 안에서 연결하는 구조였고, 그 형태까지 구현했다.

예를 들어 HLO에 아래 계산이 있고, 명세에는 두 계산을 함께 수행하는 DoubleGEMM이 있다고 하자.
여기서 rewrite는 찾은 계산에 같은 의미의 ISA 구현을 추가하는 규칙이다.

```text
G0 = dot(A, B)
G1 = dot(C, D)

multipattern
  공동 검색으로 G0·G1을 찾음
  → e-graph에 tuple(G0, G1)을 후보로 등록
  → 별도 ISA rewrite가 이 tuple 패턴을 다시 검색·검사
  → double_gemm(A, B, C, D) 구현을 후보에 추가

multipattern-rewrite
  double_gemm의 rewrite 자체가 G0·G1을 공동 검색
  → 찾은 A·B·C·D의 대응을 적용 단계로 전달하고 재검사
  → double_gemm 구현과 결과를 나타내는 tuple을 함께 추가
```

`multipattern`은 기존 ISA 변환 규칙을 재사용하기 위해 먼저 tuple 후보를 만들던 중간 단계다.
`multipattern-rewrite`에서는 **중간 후보 등록·재검색을 없애고 검색 → 검사 → 적용의 연결을 정리했다.**
결과를 표현하는 tuple은 적용할 때 만들며, 최종 명령 선택과 주소 할당은 이후에 진행한다.
이 통합은 [2026-10-26에 추가](https://github.com/jihwan38/act-multi-result/commit/3b85d5dad68867fb6768d355ecba85876fcd5685)했다.
기본 실행 구성은 `multi-result`로 유지하고 두 실험 구성은 `--toolchain`으로 선택한다.

**구조를 정리한 성과와 성능 측정 결과는 구분한다.** 비교에서 전반적인 성능 우위는 확인하지 못했다.

- ISA별 검색 기회를 나눠 기존 방식이 놓치던 종류의 후보도 찾았지만, **기본 입력 14개의 최종 코드 비용은 같았다.**
- 큰 triple 입력의 최대 메모리 중앙값은 210.68 MiB에서 286.09 / 287.95 MiB로 약 36% 늘었다.
- `shared_pairs_16`에서는 새 두 방식이 공동 후보/적용 16개를 준비해도 최종 코드에는 공동 명령 두 개만 남았다.

비용은 **assembly 파일의 줄 수**이며 장치 실행시간이 아니다. 후보 발견이 최종 선택으로 이어지지 않는
이유는 추출·주소 할당·비용 평가까지 추적해야 한다. 직접 rewrite도 ISA별로 고정 배분한 검색 예산의
남은 몫을 다른 ISA에 넘기지 못한다. 상세 조건과 결과는 [비교 보고서](../experiments/multipattern/REPORT.md)에 있다.

## 4. 실제 가속기에서도 쓸 수 있는가: Dual Gemmini

Rocket CPU 하나와 Gemmini 두 개, 엔진별 scratchpad·accumulator, 공유 DDR3를 둔 SoC를
Verilator로 실행했다. 16×16 INT8 입력·INT32 결과에서 **ACT가 선택한 명령열이 실행 가능하고
두 엔진의 계산도 겹치는지** 확인했다.

`double_gemm`은 새 하드웨어 opcode가 아니라 기존 명령을 묶은 소프트웨어 함수다.
[`act_double_gemm()`](../cases/gemmini/native/gemmini_dual.h)은 두 엔진에 preload·compute를 보낸 뒤
함께 기다린다. 개별 `act_gemm()`은 한 엔진에 보낸 뒤 바로 기다린다.
[native 변환기](../cases/gemmini/emit_native.py)는 ACT의 명령·엔진·주소를 유지하며 GEMM을 사후에 다시 묶지 않는다.

| 입력 | ACT의 선택 | 관찰 |
| --- | --- | --- |
| 독립 두 GEMM | DoubleGEMM 한 개 | dense 입력에서 두 엔진의 PE(0,0) 곱셈 활동이 같은 cycle에 나타남 |
| 입력 X를 공유하는 두 GEMM | 개별 GEMM 두 개 | 두 계산은 독립적이고 수동 병렬 실행도 가능했지만 ACT는 공동 명령을 선택하지 않음 |
| 앞 결과를 다음 입력으로 쓰는 GEMM | 개별 GEMM | 의존성 때문에 동시 묶음은 불가. INT32 결과를 INT8로 포화 변환해 다음 계산에 사용 |
| 독립 네 GEMM | 한 쌍만 DoubleGEMM | 수동 구현은 두 쌍을 묶음. ACT가 모든 병렬 기회를 활용하지 못함 |

2026-11-20 측정의 재호출 cycle 중앙값은 독립 두 GEMM에서 순차 592, 수동 병렬 511, ACT 512였다.
독립 네 GEMM은 수동 병렬 1027, ACT 1110이었다. load·compute·store·완료 대기를 포함한 값이며,
첫 호출과 재호출은 [결과 보고서](../cases/gemmini/RESULTS.md)에 구분했다.

원래 `act-parallelism/experiments/05_dual_gemmini`의 실험을 `multi-result-legacy`로 가져온 결과다.
현재 저장소에서 compiler·Oracle·native 프로그램을 다시 만들어 72회 RTL 실행·144회 결과 검사를 통과했다.
기존 RTL 실행 파일은 재사용했다. **현재 `multi-result`의 Gemmini 이식, 빈 환경의 RTL 재빌드,
FPGA·실칩 검증까지 마친 것은 아니다.**

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

## 6. 남은 한계

| 한계 | 구체적으로 남은 것 |
| --- | --- |
| 공동 명령의 실제 구현 | 의미식 외에 장치별 명령 발행·완료 대기·버퍼 규칙이 필요하다. Gemmini에서는 native 함수로 제공했다. |
| 성능을 고려한 선택 | 현재 비용은 코드 줄 수다. 공유 입력·독립 네 GEMM에서 덜 선택된 공동 명령이 어디서 탈락했는지 추적해야 한다. |
| 이동·계산 중첩 | 스케줄러는 의존 관계와 중간값의 메모리 점유를 기준으로 순서를 정한다. 명령 지연시간·엔진 점유·비동기 이동을 모델링하지 않아 다음 타일을 미리 읽는 시간표를 찾지 않는다. Gemmini 경로도 이동마다 기다린다. |
| 여러 커널·모델 전체 | 고정 타일 실험을 넘어 커널 분할·장치 배치·커널 간 이동과 동기화를 관리하는 실행 경로는 없다. 일부 PE의 겹침은 전체 모델 성능의 증거가 아니다. |
| 현재 구현으로 Gemmini 이식 | 옛 두 ROOT를 `ROOT tuple(...)`로 바꾸고, 정수 상수·`minimum`·`maximum`을 쓰는 포화 의미를 현재 변환기에 맞춰야 한다. |

**병렬 구조에 따른 DoubleGEMM의 역할**
[14번 슬라이드](https://docs.google.com/presentation/d/1xrX7PP3hDbG9fPs4vbIdwh3tAeJvsIbW8EetvnBUnIM/edit#slide=id.v2s31)에서
정리한 핵심은 **병렬 실행을 위해 GEMM 두 개를 반드시 한 명령으로 묶을 필요는 없다**는 것이다.

| 병렬 구조 | 필요한 지원 | DoubleGEMM과의 관계 |
| --- | --- | --- |
| 멀티코어형: 독립 코어·가속기 | 작업을 나눠 각각 발행하고 완료를 동기화 | 코어마다 개별 GEMM을 실행할 수 있다. 고정 묶음 명령은 필수가 아니다. |
| 분리 엔진: 이동·계산 등의 명령 큐가 따로 있음 ([Gemmini](https://github.com/ucb-bar/gemmini#decoupled-accessexecute)) | 엔진 간 의존성을 지키며 명령을 발행하고 필요한 때 기다림 | 독립적인 엔진 명령은 공동 명령 없이도 겹쳐 실행할 수 있다. |
| VLIW: 한 명령 묶음에 여러 실행 슬롯 ([명령 제약 예시](https://www.ti.com/lit/ug/spru732j/spru732j.pdf)) | 지연시간·슬롯·자원 제약에 맞춰 연산 조합을 구성 | 묶음 스케줄링이 필요하다. 고정된 GEMM 두 개를 고르는 우리 구현은 이런 슬롯 스케줄링을 하지 않는다. |

우리 2레인 설계에서 일반적인 병렬화의 핵심은 **작업 분할·레인 배치·동기화**이며,
현재 ACT 위의 타일링·배치 단계와 런타임을 연결해 다룰 문제다. Gemmini의 `double_gemm`은
두 엔진에 먼저 발행한 뒤 기다리는 방식을 한 함수에 담은 구현 선택이었다.
별도로 재사용할 성과는 **한 명령의 여러 결과를 추적·할당하는 기반**이다. 최대값과 위치를 함께 반환하는
실제 명령을 다음 사례로 검토할 수 있으며, 이 사례는 아직 구현하지 않았다.

## 7. 후속 연구로 검토했던 논문

[분석 슬라이드 16–18쪽](https://docs.google.com/presentation/d/1xrX7PP3hDbG9fPs4vbIdwh3tAeJvsIbW8EetvnBUnIM/edit#slide=id.v2s33)에서
검토한 후보 중 ACT의 명세 작성과 매핑 탐색에 직접 연결되는 두 연구다.
마지막 열은 **연결해 볼 연구 방향**이며, 이 저장소에 구현된 기능은 아니다.

| 논문 | 핵심 아이디어 | 우리 연구에서 볼 지점 |
| --- | --- | --- |
| [TensorLift](https://arxiv.org/abs/2604.13523v3) | RTL에서 추출한 비트 수준 동작을 MLIR로 분석해 텐서 연산·메모리 의미를 복원하고 TAIDL 명세를 만든다. Gemmini·VTA에 적용했다. | 손으로 쓰던 ACT 입력 명세를 자동화하는 방향. 추출된 의미를 현재 다중 결과·정수 연산 지원 범위에 맞춰 받을 수 있는지 검토한다. |
| [Zomboss](https://arxiv.org/abs/2608.00894v3) / [Contour](https://arxiv.org/abs/2608.00894v4) | 에이전트가 타일 크기·메모리 배치 등의 선택을 제안하면 컴파일러가 제약을 만족하는 코드로 완성한다. Contour는 탐색 선택지 자체를 확장하는 기능도 다룬다. | ACT의 명령·배치 선택 일부를 외부 탐색에 노출하고, 추출·할당·정답 검증과 연결하는 방향. |

슬라이드의 Zomboss는 같은 arXiv 논문의 **2026-11-18 v4에서 Contour로 이름과 구성이 바뀌었다.**
위 링크로 당시 버전과 개정판을 구분해 읽을 수 있다.

별도의 연결 후보인 [MATCHA](https://arxiv.org/abs/2604.09124v1)는 여러 이기종 가속기의 DNN 배치·메모리·
병렬 실행을 다룬다. ACT가 구간별 명령열을, MATCHA가 구간 간 배치·시작·완료를 맡는 연결을 생각할 수 있다.
이 저장소에는 통합하지 않았다.

## 8. 자료 위치

| 필요한 것 | 자료 |
| --- | --- |
| 설치·실행·회귀 검사 | [저장소 README](../README.md) |
| 검색 실험의 결과 / 재실행 | [MultiPattern 보고서](../experiments/multipattern/REPORT.md) / [실행 안내](../experiments/multipattern/README.md) |
| Gemmini 결과 / 실행 / native 연결 | [결과 보고서](../cases/gemmini/RESULTS.md) / [실행 안내](../cases/gemmini/README.md) / [native 설명](../cases/gemmini/docs/native-interface.md) |
| ISA 복제·join·DoubleGEMM의 이전 실험 | [act-parallelism](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments) |

기본 배경은 [ACT](https://arxiv.org/abs/2510.09932v2)·[TAIDL](https://doi.org/10.1145/3725843.3756075),
패턴 검색은 [egg](https://arxiv.org/abs/2004.03082)·[Relational E-Matching](https://arxiv.org/abs/2108.02290v2)을 참고한다.

`handoff-20261118`은 다중 결과·검색 비교의 코드 기준이며 이후 가져온 Gemmini legacy는 포함하지 않는다.
MultiPattern의 측정 버전은 보고서의 `comparison-20261026-*` 태그로 보존했다.
