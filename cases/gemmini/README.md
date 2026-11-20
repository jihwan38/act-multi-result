# Dual Gemmini · 이전 구현 재현

`act-parallelism`의 [실험 05](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini)를
`multi-result-legacy` 구성으로 가져왔다. **16×16 INT8 입력·INT32 결과**를 다루며,
Rocket CPU 하나에 붙은 Gemmini 두 개를 Verilator에서 실행한다.
현재 BF16 예제인 `cases/double_gemm`과는 별도 구성이다.

**먼저 볼 것:** ISA는 아래 표, 실행은 그다음 두 단계, 상세 구조는 [native 연결](docs/native-interface.md).
결론과 측정값은 [결과 보고서](RESULTS.md)에 있다.

2026-11-20 현재 경로에서 compiler·Oracle과 RTL 기능·계산 겹침을 재확인했다.
기존 RTL 실행 파일을 재사용했으며, 빈 환경에서 하드웨어를 재빌드하는 과정은 다시 실행하지 않았다.

## ISA 구성

[GemminiDual.py](GemminiDual.py)의 `make_target()`이 두 엔진의 버퍼와 명령을 선언한다.
`k=0,1`은 Gemmini 엔진 번호이며, 각 엔진의 scratchpad·accumulator는 서로 독립이다.

| 버퍼 | 역할·형태 | 용량 | 주소 단위 |
| --- | --- | --- | --- |
| `d0` | 공유 외부 메모리, TAIDL이 자동으로 제공하는 `u8` 메모리. RTL에서는 DDR3 | 명세에서 고정하지 않음 | 1 byte |
| `sp_0`, `sp_1` | 엔진별 scratchpad, `s8[16384,16]` | 각각 256 KiB | 1행 = 16 bytes |
| `acc_0`, `acc_1` | 엔진별 accumulator, `s32[1024,16]` | 각각 64 KiB | 1행 = 64 bytes |

이 실험의 TAIDL 명령은 **엔진별 4개 × 2엔진 + `double_gemm` 1개 = 9개**다.
이동·저장이 6개, 계산이 3개다.
모든 행렬은 16×16이며, `_k`는 실제 명령의 `_0`·`_1`을 묶어 표기했다.

| 명령 (종수) | 입력 → 출력 | 의미 | native 구현 |
| --- | --- | --- | --- |
| `mvin_k` (2) | `d0` 256 bytes → `sp_k` 16행 | INT8 입력 타일 읽기 | `mvin` |
| `gemm_k` (2) | `sp_k`의 타일 2개 → `acc_k` 16행 | INT8 행렬곱을 INT32로 저장. 기존 결과는 덮어씀 | `preload → compute_preloaded` |
| `mvout_k` (2) | `acc_k` 16행 → `d0` 1,024 bytes | INT32 결과를 그대로 저장 | INT32 `mvout` |
| `requantize_store_k` (2) | `acc_k` 16행 → `d0` 256 bytes | 결과를 `[-128,127]`로 제한해 INT8로 저장 | 저장 설정 변경 + INT8 포화 `mvout` |
| `double_gemm` (1) | `sp_0`의 타일 2개 + `sp_1`의 타일 2개 → `acc_0`, `acc_1` | 독립 GEMM 두 개, INT32 결과 2개 | 두 엔진에 `preload`·`compute_preloaded`를 보낸 뒤 한 번에 완료 대기 |

`i0`, `i1`, …은 입력 주소, `o0`, `o1`은 출력 주소이며 위 표의 주소 단위를 따른다.
`double_gemm`은 옛 문법인 두 `ROOT`로 결과를 선언한다. 첫 결과는 `acc_0[o0]`,
두 번째는 `acc_1[o1]`에 대응한다. 현재 BF16 명세의 `ROOT tuple(...)`과 구분해야 한다.

이 명령 수는 하드웨어 opcode 수가 아니다. [emit_native.py](emit_native.py)의 `lower_calls()`가
ACT 명령열을 [gemmini_dual.h](native/gemmini_dual.h)의 호출로 옮긴다.
WS(weight-stationary)·scale 1 등의 설정도 이 header가 제공하며, `double_gemm`은 새 opcode 없이
두 엔진의 계산을 묶는다. 개별 이동·계산 명령은 각각 완료를 기다린다.

## 1. compiler·Oracle 확인

저장소 루트에서 실행한다. 공통 [설치](../../README.md#1-설치와-act-소스-준비)가 끝난 상태를 전제로 한다.

```bash
conda activate act
./scripts/run.sh gemmini
```

이 명령은 `baseline + multi-result-legacy` 패치 적용 → compiler·Oracle 생성 →
`two_gemm.hlo` 컴파일 → 입력 6종의 기능 검증을 수행한다. 이 단계에는 Chipyard 설치가 필요 없다.
원래 HLO의 XLA 결과, CPU Oracle, 독립 주소 모델을 INT64 행렬곱 기준값과 비교한다.

생성과 검증을 따로 실행하려면 다음 순서를 쓴다.

```bash
python scripts/prepare_sources.py multi-result-legacy
source scripts/env.sh multi-result-legacy
python cases/gemmini/generate.py
./scripts/run.sh gemmini --skip-build --workload cases/gemmini/workloads/dependent_gemm.hlo
```

생성물은 전부 `build/multi-result-legacy/` 아래에 있다.
`targets/GemminiDual/`은 생성 소스, `backends/GemminiDual`은 compiler,
`runs/gemmini/<입력 이름>/`은 assembly·기능 검증 기록이다.
Gemmini는 전용 `verify.py`를 사용하며, `run.sh`가 자동으로 선택한다.

## 2. RTL에서 실행

```bash
source scripts/env.sh multi-result-legacy
bash cases/gemmini/hardware/setup.sh
bash cases/gemmini/hardware/build.sh dual

# 아직 존재하지 않는 출력 폴더를 지정한다.
python cases/gemmini/run_suite.py --output "$ACT_BUILD_DIR/runs/gemmini/rtl-01"
python cases/gemmini/analyze_results.py \
  --suite "$ACT_BUILD_DIR/runs/gemmini/rtl-01" \
  --output "$ACT_BUILD_DIR/runs/gemmini/rtl-01/analysis.json"
```

`setup.sh`는 고정 Chipyard를 초기화하고 독립 작업본·RTL Conda 환경을 준비한다.
`build.sh dual`은 관측 패치와 두 엔진 설정을 적용해 시뮬레이터를 만든다.
공통 `scripts/setup.sh`는 ACT 원본 4개만 초기화한다. RTL 의존성은 이 단계에서만 준비한다.

`run_suite.py`는 **입력 그래프 4개 × 데이터 6종 × 실행 방식 3종 = RTL 72회**를 실행한다.
각 프로세스에서 두 번 호출하므로 결과 검사는 144회다. 실행 방식은 순차·수동 병렬·ACT 생성이며,
각 실행의 binary·trace hash, 수치 결과, 실제 수락된 명령과 PE(0,0) 활동을 검사한다.
`summary.json`과 `analysis.json`을 함께 확인한다. 기존 결과 폴더에는 덮어쓰지 않는다.

| 위치 | 내용 |
| --- | --- |
| `cases/gemmini/` | 명세, 입력 HLO, native 변환·검증, 하드웨어 설정 |
| `patches/multi-result-legacy/` | 이전 ACT 패치 12개와 적용 순서 |
| `third_party/chipyard/` | 고정 원본. 직접 수정하지 않음 |
| `build/multi-result-legacy/hardware/chipyard/` | 하드웨어 패치·빌드용 작업본 |
| `build/multi-result-legacy/deps/` | 독립 RTL 도구 환경 |

## 범위와 추가 검사

`double_gemm`은 두 엔진에 preload·compute를 보내는 **소프트웨어 명령 묶음**이다.
새 하드웨어 opcode를 추가하지 않았다. 이동 명령은 완료를 기다리므로 일반적인
DMA/compute 동시 실행이나 모델 전체 성능을 검증하는 실험은 아니다.
공유 입력 두 GEMM, 의존 GEMM, 독립 네 GEMM도 함께 검사하여 선택 범위와 한계를 드러낸다.

옛 명세·ACT 패치·native 계산·입력 HLO는 유지하고, 경로·소스 준비·실행 기록을 현재 저장소에 맞췄다.
현재 `multi-result`로 옮기려면 별도로 ROOT tuple 문법과 상수·포화 의미 처리를 맞춰야 한다.
공개 ACT artifact 전체의 호환 작업과도 구분한다.

추가 회귀 검사는 legacy 환경을 선택한 셸에서 실행한다.

```bash
python -m unittest discover -s cases/gemmini/tests -p 'test_*.py' -v

mkdir -p "$ACT_BUILD_DIR/targets/GemminiDual/backend/tests"
cp cases/gemmini/tests/*.rs "$ACT_BUILD_DIR/targets/GemminiDual/backend/tests/"
ACT_TEST_EXPERIMENT="$PWD/cases/gemmini" cargo test --release \
  --manifest-path "$ACT_BUILD_DIR/targets/GemminiDual/backend/Cargo.toml" \
  --test saturation_matching --test iterative_discovery -- --test-threads=1

python cases/gemmini/tests/rtl_mutations.py \
  --source "$ACT_BUILD_DIR/runs/gemmini/rtl-01/native/two_gemm_901_act.c" \
  --output "$ACT_BUILD_DIR/runs/gemmini/rtl-01-mutations"
```

마지막 검사는 결과 lane 복제·완료 fence 제거·잘못된 누산을 주입한다.
정답 배열은 유지하며, 실제 수치 비교가 실패해야 검출 성공이다.
