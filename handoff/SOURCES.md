# ACT 인계 자료와 근거

핵심 내용은 [인계 본문](README.md)에 있다. 이 파일은 수치·구현·공개 자료를 확인할 때 사용한다.
자료 정리일은 2026-11-20이며, 아래 기록은 기존 측정 결과를 보존한 것이다.

## 1. 현재 코드와 MultiPattern

코드 기준은 `handoff-20261118`, commit `129d5998bf47ce25f6c77a16caa3e4d3f698e214`다.
인계 문서 추가는 이 코드의 기능과 측정값을 바꾸지 않는다.

| 자료 | 무엇을 확인할 수 있나 |
| --- | --- |
| [README](../README.md) | 설치·실행·검증 방법, 실행 구성, 패치별 역할 |
| [DoubleGEMM 명세](../cases/double_gemm/QKVDoubleGemm.py)와 [입력 HLO](../cases/double_gemm/workloads/two_gemm_concat.hlo) | 일반 dot 두 개와 concatenate를 다중 결과 명령으로 구현하는 대표 경로 |
| [세 구성의 회귀 검사 기록](evidence/regression.json) | 구성별 10개 그룹 통과와 Python 52개 검사 로그. 기능 검증 기록이며 성능 측정값은 아님 |
| [최종 비교 보고서](../experiments/multipattern/REPORT.md) | 2026-10-26 비교의 조건·결론·한계 |
| [컴파일 측정 CSV](../experiments/multipattern/results/compiler-samples.csv) | 반복 실행의 시간·메모리·코드 비용 |
| [고정 그래프 측정 CSV](../experiments/multipattern/results/micro-samples.csv) | 검색 단계만 분리한 측정 |
| [원자료 압축 JSON](../experiments/multipattern/results/raw-results.json.gz) | 보고서의 전체 원자료 |
| [측정 소스 대응](../experiments/multipattern/results/source-provenance.json) | 측정 커밋·upstream·patch·실행 파일의 대응 |

보고서에서 `main`은 측정 당시 기준 구현 `10f059e`를 뜻한다. 현재 `main`에는 세 구성이 모두 있다.
측정 시점은 보존 태그 `comparison-20261026-multi-result`, `comparison-20261026-multipattern`,
`comparison-20261026-multipattern-rewrite`로 구분한다.
10월 보고서의 Python 검사 수 51개와 인계 코드의 52개는 시점이 다르다.
정수 타입 생성 오류 수정과 검사가 나중에 추가됐으며, 과거 수치를 덮어쓰지 않았다.

### 구현을 확인할 위치

| 확인할 내용 | 유지하는 소스와 주요 이름 |
| --- | --- |
| 명세의 결과 순서 | [TAIDL 패치](../patches/multi-result/taidl/0001-semantic-interface.patch): `semantic_interface()` |
| 의미식과 ISA 규칙 생성 | [Backend 의미 변환 패치](../patches/multi-result/act-backend/0001-isa-rewrites.patch): `lower_semantics()` |
| 명령과 결과 값의 구분 | [PII 패치](../patches/multi-result/act-backend/0004-pii-model.patch): `ValueRef`, `PiiResult`, `PiiGraph` |
| 명령 선택 이후의 처리 | [추출 패치](../patches/multi-result/act-backend/0005-pii-extraction.patch), [주소·수명·출력 패치](../patches/multi-result/act-backend/0006-pii-to-assembly.patch) |
| 기능 검증의 기준 | [verify.py](../scripts/verify.py): HLO reference와 Oracle 결과 비교 |
| 빈 출력의 실패 처리 | [run.py](../scripts/_internal/run.py): assembly 존재·크기, 선택된 PII, 검증 성공 확인 |

현재 고정 버전에서 확인한 구현 제약은 다음과 같다. 과거 실패 사례 전체가 현재도 실패한다고 가정하지 않는다.

- `lower_semantics()`는 상수·minimum·maximum 의미식을 아직 처리하지 않는다. 표준 행렬곱 이외의 dot 속성도 거절한다.
- 입력 HLO의 `initializer/parser.rs`는 dot을 일반 이항 연산으로 읽으며 contracting/batch 속성을 보존하지 않는다. 비표준 dot을 일반 지원 범위로 간주하면 안 된다.
- `emitter.cc`의 상수 metadata는 특정 16×16 eye 초기화식에 고정돼 있다. 일반 상수 지원을 추가할 때는 의미 변환과 실제 초기화 경로를 함께 확인해야 한다.
- `scheduler::topological()`은 메모리 점유를 기준으로 의존성 순서를 고른다. 기본 `python/cost/model.py`는 assembly 줄 수를 반환한다.
- 과거 Backend의 “종료 코드 0인데 출력이 없음”과 현재 `run.sh`의 성공 판정은 다르다. 현재 실행 래퍼는 빈 출력을 실패로 처리한다.

## 2. 이전 저장소 act-parallelism의 Dual Gemmini 결과

**이 절의 결과는 `act-parallelism/experiments/05_dual_gemmini`에서 얻었다.**
`act-multi-result`에서 Gemmini를 실행해 얻은 결과가 아니다. 현재 저장소로의 이식·재검증은 미완료다.

별도 저장소의 기준은 [`act-parallelism` commit 6164b44](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini)다.
그 안의 [결과 보고서](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/docs/results.md),
[실행과 측정](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/docs/reproduction.md),
[native 연결](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/docs/native-interface.md)이 상세 자료다.

여기서는 **2026-09-18 재검증**의 기록을 사용한다. 첫 호출 / 결과 검사 후 재호출의
6개 입력 중앙값이며 단위는 cycle이다. 입력 초기화와 검사는 제외하고 load·compute·store·완료 fence를 포함한다.

| 입력 | 순차 기준선 | 수동 병렬 | ACT |
| --- | ---: | ---: | ---: |
| 독립 두 GEMM | 1089 / 592 | 1005 / 511 | 1005 / 511 |
| X 공유 두 GEMM | 1007 / 590 | 877 / 509 | 959 / 546 |
| 의존 GEMM | 879 / 572 | 881 / 571 | 878 / 574 |
| 독립 네 GEMM | 1916 / 1192 | 1807 / 1027 | 1734 / 1110 |

[실행 요약](evidence/gemmini-summary.json)과 [분석 JSON](evidence/gemmini-analysis.json)은
이전 저장소의 기록을 보존한 사본이다. 그 실험의 72개 실행과 144개 커널 검사, cycle 및 계산 겹침 기록이 있다.
독립 두 GEMM의 seed 901에서 ACT의 PE(0,0) nonzero 곱셈 겹침은 두 호출 합계 30 cycle이다.
PE 하나의 관측으로 전체 array utilization을 측정했다고 할 수는 없다.

이 실험은 한 SoC 안의 두 Gemmini, 16×16 INT8 입력·INT32 결과, 공유 DDR3를 사용한다.
`double_gemm`은 preload·compute·fence로 내려가는 소프트웨어 명령 묶음이다.
DMA는 완료를 기다리므로 일반적인 DMA/compute 중첩이나 전체 모델 병렬화를 검증하지 않는다.
9월 16일과 18일의 ACT cycle 차이는 일부 row 주소 변경과 함께 발생했다. 공동 명령 수는 같았으므로
그 차이를 공동 명령 선택 개선의 성능 효과로 설명하지 않는다.

현재 저장소의 `cases/`에는 Gemmini가 없다. 이식할 때는 기존 05의 상수·포화 의미 처리와
native 연결을 현재 다중 결과 경로에 맞춰 가져와야 한다.

## 3. 공개 ACT에서 확인한 범위

분석 기준은 [ACT 논문 v2](https://arxiv.org/abs/2510.09932v2)와
[OOPSLA artifact commit a304892](https://github.com/act-compiler/act-artifact-oopsla26/tree/a30489226fe60fe281d855ff0b2a718817523dec)다.
아래는 논문과 공개 코드의 대조이며, 우리가 모든 장치의 성능을 재측정한 결과는 아니다.

**Gemmini 명세에는 이동도 들어 있다.**
[gemmini.py](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/accelerators/gemmini.py)의
TAIDL 연산은 이동 10종과 계산 4종이다. 계산은 matmul8·matmul32·mac8·mac32다.
TAIDL 연산 하나가 실제 opcode 하나와 일치하지 않는다. 예를 들어 matmul은
[RoCC 변환기](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/evaluation/gemmini/codegen/gemmini_rocc.py)에서
preload와 compute로 펼쳐진다. 실행 설정은
[C 템플릿 예](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/evaluation/gemmini/templates/swlib/AxB_plus_C.c)도 함께 봐야 한다.

**Trainium의 2.3배는 컴파일·정확성 성공 개수의 비율이다.**
논문 §8.4는 준비한 1,055개 커널에서 ACT 1,055개, neuronx-cc 459개 성공을 보고한다.
모델 실행속도가 2.3배라는 뜻이 아니다. 공개
[NKI 평가 코드](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/evaluation/nki/eval_nki.py)는
기능 시뮬레이터로 결과를 비교한다. 논문의 실장치 평가 보고와 공개 코드의 재현 범위를 구분한다.
코퍼스 생성도 지원 연산 필터와 형상 재구성을 거친다. TPUGraphs 전체 원본 프로그램 지원으로 확대하지 않는다.

논문이 설명한 neuronx-cc 실패의 “미지원 연산”은 shift-left처럼 NKI가 직접 제공하지 않는 연산을
다른 연산으로 바꾸어 구현하지 못하는 경우다. “빈 출력”은 일부 단일 노드 타일 커널에서 NKI 코드가
비어 있는 경우, “상수 초기화”는 필요한 상수 텐서를 초기화하지 않아 정확성 검사에 실패하는 경우다.
각 실패 유형을 모든 입력에 적용되는 일반 결함이나 현재 최신 버전의 상태로 단정하지 않는다.

**AMX의 평가는 선정한 타일 커널에 대한 것이다.**
논문 §8.5와 부록 H는 커널 5개, Xeon Gold 5415+, oneDNN 대비 성능 1.0배를 보고한다.
CNN·RNN 이름이 전체 모델 추론을 뜻하지 않는다. 해당 OOPSLA artifact의
[full.sh](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/scripts/full.sh)는
저장된 AMX 성능 CSV를 사용하므로 그래프를 다시 그리는 것과 실장치 성능을 다시 측정하는 것을 구분한다.

**공식 예제 데이터도 의미식과 대조해야 했다.** 이전 QKV 실험에서는 Oracle이 broadcast 차원을
잘못 재사용했고, 제공된 attention 데이터도 그 과거 동작과 일치했다.
수정 후에는 입력 HLO를 XLA로 실행한 결과와 비교했다.
[00 실험의 검증 설명](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/00_stock_qkv/docs/06-functional-verification.md)을 참고한다.

## 4. 후속 방향의 출발점

논문 링크와 질문별 구분은 [인계 본문의 읽을거리](README.md#5-필요한-자료만-찾아보기)에 있다.

MATCHA 연동에서 검토한 역할 분담은 다음과 같다. ACT는 각 장치에서 실행할 계산 구간의 명령열을 만들고,
MATCHA는 구간의 장치 배치·전역 메모리·시작과 완료를 관리한다. 구간을 HLO로 바꾸고 ACT의 입출력 규약과
연결하는 어댑터는 새로 구현해야 한다. ACT와 MATCHA를 연결한 실행은 아직 없다.

검토한 MATCH commit은 `b499427e9bb9c757e81c1b7c63e3634b003c7973`이다.
연결 지점은 [ExecModule](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/target/exec_module.py),
[MATCHA transform](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/matcha/transform.py),
[비동기 실행 템플릿](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/libs/c/mako/matcha/graph.c)이다.
두 Oracle로 연결의 기능을 검증하더라도 그것만으로 실제 장치의 병렬 성능이 검증되지는 않는다.

max·argmax는 자연스러운 다중 결과 사례의 후보이고, Contour·TensorLift는 각각 매핑 탐색과 명세 생성의
후속 읽을거리다. 이 프로젝트에서 이미 구현하거나 성능을 검증한 기능으로 포함하지 않는다.

## 5. 함께 보존한 파일

`evidence/regression.json`은 현재 저장소의 회귀 검사 기록이다.
`gemmini-summary.json`과 `gemmini-analysis.json`은 이전 `act-parallelism` 실험의 참고 사본이다.
[출처와 SHA-256](evidence/provenance.json)에 원래 위치와 파일 hash를 기록했다.
Gemmini JSON 안의 과거 경로와 `05_for_next` 이름은 원본 기록 그대로이며 현재 실행 경로가 아니다.
이 요약에는 RTL 실행 파일·전체 trace·빌드 환경이 포함되지 않는다. 재실행에는 별도 저장소의 준비 절차가 필요하다.

MultiPattern 원자료는 이미 `experiments/multipattern/results/`에 있어 복제하지 않았다.
공개 분석의 핵심 근거는 위의 고정 버전 링크로 연결했다.
로컬 `docs/`의 과거 초안·그림과 `build/` 전체를 함께 읽을 필요는 없다.
