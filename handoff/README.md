# ACT 연구 인수인계

2026-11-20 정리. 연구 흐름과 현재 상태는 이 문서에, 실행법은 각 README에 있다.
공유용 [PDF](ACT-handoff.pdf)도 같은 내용이다.

## 1. 현재 어디까지 됐나

**다중 결과 지원을 구현했고, 검색 방식 비교를 마쳤다. 옛 Gemmini 실험도 별도 구성으로 가져왔다.**

| 작업 | 현재 상태 | 위치 |
| --- | --- | --- |
| 다중 결과 지원 | 명령 선택·주소 할당·코드 생성·Oracle 기능 검증까지 연결 | `multi-result` |
| MultiPattern 비교 | 세 구현의 비교 완료. 전반적인 성능 우위는 확인하지 못함 | `experiments/multipattern/` |
| Dual Gemmini | 옛 구현으로 RTL 기능과 계산 겹침 재확인 | `cases/gemmini/`, `multi-result-legacy` |

코드와 비교 자료는 현재 저장소에 모았다. QKV는 `baseline`, BF16 DoubleGEMM은 `multi-result`,
Gemmini는 `multi-result-legacy`를 사용한다. **현재 다중 결과 구현과 옛 Gemmini 구현은 별도다.**
MATCHA와의 연결은 검토만 했고, 통합 구현·실행은 하지 않았다.

## 2. 왜 시작했고 무엇을 바꿨나

시작 질문은 “ACT에 계산 유닛을 여러 개 기술하면 독립 계산을 여러 유닛에 배치할 수 있는가?”였다.
초기 attention 실험에서는 ISA를 두 벌 선언해도 한쪽 유닛만 사용했다.
두 유닛의 결과를 요구하는 `join_store`로 두 유닛을 사용하는 명령열을 만든 뒤,
한 명령이 여러 결과를 내는 경우를 ACT 전체 처리 과정에 연결했다.

수정의 중심은 **명령 하나와 그 명령이 만드는 여러 값을 구분하는 것**이다.
명세에서 결과 순서를 읽고, 여러 계산을 공동 명령 후보로 찾고, 결과마다 주소와 마지막 사용 시점을
관리하도록 바꿨다. 생성된 호출문의 출력은 입력 HLO의 실행 결과와 비교한다.

대표 예제는 일반 `dot` 두 개와 concatenate를 입력받아 `double_gemm`을 선택하는
BF16 64×64 사례다. 한 명령의 여러 결과를 처리하는 기반을 남겼으며,
엔진 배치와 실행시간을 자동으로 최적화하는 기능은 별도 과제다.

## 3. 실험에서 얻은 결론

**검색 방식을 바꾼 것만으로 전반적인 개선은 나오지 않았다.** 결과별 검색 후 조합하는 방식과
두 MultiPattern 방식을 비교했다. 기본 입력 14개에서 최종 코드 비용은 같았고,
큰 triple 입력에서는 새 두 방식의 최대 메모리가 약 36% 늘었다.
공동 후보 16개 중 최종 코드에는 공동 명령 두 개만 선택된 사례도 있었다.
후보를 찾은 뒤의 추출·할당·비용 평가까지 살펴볼 필요가 있다.
여기서 코드 비용은 assembly 줄 수이며 장치 실행시간이 아니다.

**Gemmini에서는 두 엔진의 실제 계산 겹침을 확인했다.** 16×16 정수 사례에서
`double_gemm`은 두 엔진에 preload·compute를 보내는 소프트웨어 명령 묶음이다.
공유 입력의 두 GEMM은 개별 명령으로 남고, 독립 네 GEMM도 한 쌍만 공동 명령으로 선택됐다.
이전 ACT 패치와 명세를 보존한 `multi-result-legacy`의 결과다.
현재 저장소에서 프로그램을 다시 생성해 실행했으며, 기존 RTL 실행 파일은 재사용했다.

## 4. 한계와 이어갈 선택

현재 스케줄러는 의존 관계와 중간값의 메모리 점유를 기준으로 명령 순서를 고른다.
엔진 점유·지연시간을 모델링해 “계산 중 다음 입력을 미리 읽는 순서”를 직접 찾지는 않는다.
또한 한 타일 커널의 명령 선택과 여러 커널의 장치 배치·비동기 실행은 다른 최적화 단위다.
모델 전체의 병렬 실행은 이번 구현 범위에 들어 있지 않다.

다음은 이어받는 사람이 연구 목적에 따라 고를 수 있는 방향이다.

| 관심 방향 | 남은 판단과 작업 |
| --- | --- |
| 다중 결과 명령의 효용 | max·argmax처럼 원래 결과가 여러 개인 실제 명령 사례를 찾고 효과 확인 |
| 현재 구현으로 Gemmini 이식 | ROOT tuple 문법과 상수·minimum·maximum 의미 처리를 맞춤. 공개 artifact 전체 호환은 추가 작업 |
| 실행 성능 최적화 | 후보가 최종 코드에서 탈락하는 이유, 실행시간 비용모델·자원 제약·커널 간 스케줄링 검토 |

MATCHA는 마지막 방향의 연결 후보이며, 구현된 기능은 아니다.
남은 기술 제약과 공개 ACT를 읽을 때의 주의점은 [자료 안내](SOURCES.md)에 모았다.

## 5. 필요한 자료만 찾아보기

| 필요한 것 | 자료 |
| --- | --- |
| 설치·실행·수정 파일 위치 | [저장소 README](../README.md) |
| 검색 방식 비교의 결론과 측정값 | [MultiPattern 보고서](../experiments/multipattern/REPORT.md) |
| Gemmini 실행과 실험 결과 | [실행 안내](../cases/gemmini/README.md) · [결과 보고서](../cases/gemmini/RESULTS.md) |
| 공개 ACT 분석과 참고 자료 | [자료 안내](SOURCES.md) · [분석 슬라이드](https://docs.google.com/presentation/d/1xrX7PP3hDbG9fPs4vbIdwh3tAeJvsIbW8EetvnBUnIM/edit) |

배경을 다시 볼 때는 [ACT](https://arxiv.org/abs/2510.09932v2)와
[TAIDL](https://doi.org/10.1145/3725843.3756075)부터 읽으면 된다. 이후에는 관심 질문에 따라 고른다.

- 패턴 검색: [egg](https://arxiv.org/abs/2004.03082), [Relational E-Matching](https://arxiv.org/abs/2108.02290v2).
- 여러 장치와 커널의 스케줄링: [MATCHA](https://arxiv.org/abs/2604.09124v1).
- 매핑 탐색과 명세 생성: [Contour](https://arxiv.org/abs/2608.00894v4), [TensorLift](https://arxiv.org/abs/2604.13523v3).

수정할 소스는 `patches/`와 `cases/`, 고정 원본은 `third_party/`, 생성물은 `build/`에 있다.
실행할 필요가 생기면 README의 명령과 기존 검사 스크립트를 사용한다.
