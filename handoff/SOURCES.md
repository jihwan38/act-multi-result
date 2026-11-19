# 필요한 자료와 읽을 때의 주의점

연구 흐름은 [인계 본문](README.md)에 있다. 아래는 특정 질문을 더 살펴볼 때 쓰는 안내다.

## 코드와 실험

| 궁금한 것 | 볼 곳 |
| --- | --- |
| 어떻게 실행하고 검사하는가? | [저장소 README](../README.md), [Gemmini README](../cases/gemmini/README.md) |
| 어떤 코드를 고쳤는가? | [README의 패치 안내](../README.md#폴더와-패치의-역할) |
| MultiPattern 비교에서 무엇이 나왔는가? | [보고서](../experiments/multipattern/REPORT.md). 측정값과 당시 소스 버전도 여기서 연결한다. |
| Gemmini에서 무엇을 확인했는가? | [결과 보고서](../cases/gemmini/RESULTS.md). 이전 실험과 현재 경로의 측정을 구분한다. |
| 이전 실험은 어디에 있는가? | [act-parallelism의 실험들](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments) |

`handoff-20261118`은 다중 결과 지원과 검색 비교의 기존 코드 기준이다. 이후 추가한 Gemmini legacy
구성은 포함하지 않는다. MultiPattern 보고서의 `main`도 **측정 당시 기준 구현 `10f059e`**를 뜻한다.
세 구현의 측정 버전은 보고서에 적힌 `comparison-20261026-*` 태그로 보존했다.

현재 `multi-result`를 확장할 때는 다음 제약을 먼저 확인한다.

- `lower_semantics()`는 상수·minimum·maximum과 비표준 dot 의미를 일반 지원하지 않는다. Gemmini legacy에는 별도 상수·포화 처리가 있다.
- 입력 HLO의 dot contracting/batch 속성 보존과 일반 상수 초기화도 함께 확인해야 한다. 일부 지원만 추가해 전체 연산을 지원한다고 간주하지 않는다.
- signed 정수의 Rust 이름을 `S8/S32`로 생성하던 오류는 `I8/I32`로 고쳤다. `set_inplace`·scale·가변 크기·2차원 주소 등 공개 artifact의 인터페이스 호환은 남아 있다.

## 공개 ACT를 읽을 때

분석 기준은 [ACT 논문 v2](https://arxiv.org/abs/2510.09932v2)와
[OOPSLA artifact commit a304892](https://github.com/act-compiler/act-artifact-oopsla26/tree/a30489226fe60fe281d855ff0b2a718817523dec)다.
최신 upstream이나 우리가 고정한 코드와 같은 버전으로 취급하지 않는다.

**Gemmini 명세에는 이동도 포함된다.** 공개 [gemmini.py](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/accelerators/gemmini.py)는
이동 10종과 계산 4종, 총 14개 TAIDL 연산을 선언한다. 계산은 `matmul8`, `matmul32`, `mac8`, `mac32`다.
TAIDL 연산과 하드웨어 opcode는 일대일이 아니다. matmul은
[RoCC 변환기](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/evaluation/gemmini/codegen/gemmini_rocc.py)에서
preload·compute로 펼쳐지고, 실행 설정은 주변 템플릿도 함께 제공한다.

**Trainium의 2.3배는 성공 개수의 비율이다.** 논문 §8.4의 ACT 1,055개와 neuronx-cc 459개는
준비한 커널의 컴파일·정확성 성공 수다. 모델 실행속도가 2.3배라는 뜻은 아니다.
실패 설명의 미지원 연산은 shift-left 등의 대체 구현 실패, 빈 출력은 일부 단일 노드 커널의 빈 코드,
상수 초기화는 필요한 상수를 초기화하지 않은 경우다. 당시 사례를 현재 버전 전체의 결함으로 확대하지 않는다.
공개 [NKI 평가 코드](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/evaluation/nki/eval_nki.py)는
기능 시뮬레이터를 사용하므로 논문의 실장치 평가와 재현 범위를 구분한다.

**AMX의 평가는 선정한 타일 커널 5개에 대한 것이다.** CNN·RNN 이름이 전체 모델 추론을 뜻하지 않는다.
공개 artifact의 [full.sh](https://github.com/act-compiler/act-artifact-oopsla26/blob/a30489226fe60fe281d855ff0b2a718817523dec/scripts/full.sh)는
저장된 AMX 성능 CSV를 사용한다. 그래프를 다시 그리는 것과 실장치 측정은 구분한다.

**예제 정답도 의미식과 대조해야 했다.** 이전 QKV에서 Oracle의 broadcast 처리와 제공된 데이터가
함께 잘못된 동작에 맞춰져 있었다. 수정 후에는 원래 HLO를 XLA로 실행한 결과와 비교했다.
[당시 분석](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/00_stock_qkv/docs/06-functional-verification.md)을 참고한다.

## MATCHA를 이어서 살펴본다면

검토한 역할 분담은 ACT가 각 계산 구간의 명령열을 만들고, MATCHA가 구간의 장치 배치·전역 메모리·
시작과 완료를 관리하는 것이다. 구간을 HLO로 바꾸고 ACT 입출력과 연결하는 어댑터는 구현하지 않았다.

검토 버전은 MATCH commit `b499427e9bb9c757e81c1b7c63e3634b003c7973`이다.
연결 지점은 [ExecModule](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/target/exec_module.py),
[MATCHA transform](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/matcha/transform.py),
[비동기 실행 템플릿](https://github.com/eml-eda/match/blob/b499427e9bb9c757e81c1b7c63e3634b003c7973/match/libs/c/mako/matcha/graph.c)이다.
읽을 논문의 전체 목록은 [인계 본문](README.md#5-필요한-자료만-찾아보기)에 있다.
