# Dual Gemmini에서 확인한 것

**독립 두 GEMM을 두 엔진에서 겹쳐 실행할 수 있었다.** 공유 입력 두 GEMM은 ACT가 개별 명령으로
선택했고, 독립 네 GEMM도 한 쌍만 공동 명령으로 선택했다. 여러 결과를 내는 명령의 실행 가능성과
선택의 한계를 보여주는 실험이다. 일반적인 비동기 스케줄러나 전체 모델의 성능을 검증한 것은 아니다.

## 구성과 검증 범위

- Rocket CPU 하나 + Gemmini 두 개 + 공유 DDR3, 16×16 INT8 입력·INT32 결과.
- `double_gemm`은 기존 preload·compute·fence 명령을 묶은 소프트웨어 명령이다.
- 네 workload에 dense random 4종·identity·zero 입력을 사용하고, 순차·수동 병렬·ACT 생성을 비교했다.
- 독립 두 GEMM의 dense 입력에서는 두 엔진의 PE(0,0) nonzero 곱셈 활동이 겹쳤다. 전체 array 활용률은 측정하지 않았다.
- 이동 명령마다 완료를 기다린다. 일반적인 DMA/compute 중첩은 이 실행 경로에 없다.

2026-09-18 기록은 `act-parallelism`의 이전 실험이다. 2026-11-20에는 이를 `multi-result-legacy`로
가져와 compiler·Oracle·native 프로그램을 다시 만들고 RTL 72회·결과 검사 144회를 통과했다.
기존 RTL 실행 파일과 도구 환경은 독립 복사본으로 재사용했으며, 빈 환경에서 하드웨어를 재빌드하지는 않았다.
현재 `multi-result` 구현을 Gemmini에 이식한 결과와 구분한다.

## 현재 경로에서 측정한 cycle

아래는 **2026-11-20 측정**이다. 각 칸은 입력 6종의 중앙값으로, 첫 호출 / 결과 검사 후 재호출이다.
입력 준비와 수치 검사는 제외하고 load·compute·store·완료 fence를 포함한다.

| 입력 | 순차 | 수동 병렬 | ACT | ACT의 공동 명령 수 |
| --- | ---: | ---: | ---: | ---: |
| 독립 두 GEMM | 1089 / 592 | 1005 / 511 | 1006 / 512 | 1 |
| X 공유 두 GEMM | 1007 / 590 | 877 / 509 | 1004 / 542 | 0 |
| 의존 GEMM | 879 / 572 | 881 / 571 | 878 / 574 | 0 |
| 독립 네 GEMM | 1916 / 1192 | 1807 / 1027 | 1737 / 1110 | 1 |

공동 명령이 실제로 선택되는지, 실제 계산이 겹치는지, 커널 전체가 빨라지는지는 각각 확인해야 한다.
특히 첫 호출과 재호출의 시간을 섞지 않는다. 이식 전후의 cycle 차이를 컴파일러 개선 효과로 해석하지 않는다.

## 측정 자료와 재실행

[rtl-measurements.json](results/rtl-measurements.json)에 9월과 11월 측정의 cycle·계산 겹침·
실행 파일 hash·소스 식별 정보를 날짜별로 보존했다. 두 분석 결과의 수치는 원래 기록과 같다.
상세 회귀 로그는 이 자료에 포함하지 않는다. 전체 RTL executable과 trace도 저장소에는 포함하지 않는다.

이전 실험의 맥락은 [act-parallelism의 결과 보고서](https://github.com/jihwan38/act-parallelism/blob/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/docs/results.md)에,
현재 실행 명령은 [README](README.md)에 있다. 하드웨어와 native 명령의 관계는
[native 연결 설명](docs/native-interface.md)을 참고한다.
