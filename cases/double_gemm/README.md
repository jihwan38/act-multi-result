# DoubleGEMM · 다중 결과 ISA

[QKVDoubleGemm.py](QKVDoubleGemm.py)는 [QKV ISA](../qkv/README.md)를 두 유닛으로 늘리고,
한 명령에서 두 GEMM 결과를 내는 `double_gemm`과 두 결과를 합쳐 저장하는 `join_store`를 추가한다.
BF16 64×64 예제이며 기본 실행 구성은 `multi-result`다.

## ISA 구성

`k=0,1`은 유닛 번호다. 각 유닛은 자기 `d1_k`·`d2_k`를 쓰고, 외부 메모리 `d0`만 공유한다.
아래 로컬 버퍼의 용량은 **유닛 하나당** 값이다.

| 버퍼 | 역할·형태 | 용량 | 주소 단위 |
| --- | --- | --- | --- |
| `d0` | 공유 HBM, TAIDL이 자동으로 제공하는 `u8` 메모리 | 명세에서 고정하지 않음 | 1 byte |
| `d1_0`, `d1_1` | 각 유닛의 GEMM 입력, `bf16[128,64]` | 각각 16 KiB, 타일 2개 | 1행 = 128 bytes |
| `d2_0`, `d2_1` | 각 유닛의 계산 결과, `bf16[64,64]` | 각각 8 KiB, 타일 1개 | 1행 = 128 bytes |

TAIDL 명령은 **유닛별 7개 × 2유닛 + 공동 명령 2개 = 16개**다.
표의 `_k`는 실제 이름의 `_0`·`_1`을 묶어 표기한 것이다.

| 명령 | 명령 수 | 입력 → 출력 | 동작 |
| --- | --- | --- | --- |
| `load_rm_k` | 2 | `d0` → `d1_k` | `n × 128` bytes를 BF16 `n`행으로 읽기 |
| `load_cm_k` | 2 | `d0` → `d1_k` | 읽은 BF16 행렬을 전치 |
| `store_rm_k` | 2 | `d1_k` → `d0` | BF16 `n`행을 `n × 128` bytes로 저장 |
| `store_cm_k` | 2 | `d1_k` → `d0` | BF16 행렬을 전치한 뒤 저장 |
| `mov_k` | 2 | `d2_k` → `d1_k` | 결과 `n`행을 같은 유닛의 입력 버퍼로 복사 |
| `gemm_k` | 2 | `d1_k`의 행렬 2개 → `d2_k` | BF16 64×64 행렬곱, 결과 1개 |
| `softmax_k` | 2 | `d2_k` → 같은 주소 | 행마다 `exp(x) / sum(exp(x))` 적용 |
| `double_gemm` | 1 | `d1_0`의 행렬 2개 + `d1_1`의 행렬 2개 → `d2_0`, `d2_1` | 독립 64×64 GEMM 두 개를 한 명령으로 표현, 결과 2개 |
| `join_store` | 1 | `d2_0`, `d2_1` → `d0` | 두 결과를 행 방향으로 이어 BF16 128×64, 16,384 bytes로 저장 |

`@c.n`은 행 수, `@a.*`는 해당 버퍼의 주소다. GEMM은 64×64 고정이며, 기본 입력은 `n=64`를 사용한다.
`double_gemm`의 `ROOT tuple(%left, %right)` 순서가 출력 포트 순서다.
첫 결과는 `d2_0[addr_out0]`, 두 번째는 `d2_1[addr_out1]`에 쓴다.
두 주소가 모두 0이어도 버퍼가 다르므로 서로 덮어쓰지 않는다.

## 기본 입력과 실행

[two_gemm_concat.hlo](workloads/two_gemm_concat.hlo)는 일반 `dot` 두 개와 `concatenate`로 구성된다.
ACT가 이를 `double_gemm → join_store`로 선택하는지를 확인한다.
`double_gemm` 자체는 결과 두 개를 유지하고, 최종 출력 하나로 합치는 일은 `join_store`가 맡는다.
개별 `gemm_0`·`gemm_1`도 선택 후보로 남아 있다.

공통 [설치](../../README.md#1-설치와-act-소스-준비) 후 저장소 루트에서 실행한다.

```bash
conda activate act
./scripts/run.sh double_gemm
# 같은 ISA를 MultiPattern rewrite 구현으로 실행
./scripts/run.sh double_gemm --toolchain multipattern-rewrite
```

기본 생성물은 `build/multi-result/runs/double_gemm/two_gemm_concat/`에 있다.
`--toolchain`을 바꾸면 `build/<toolchain>/` 아래에 생성한다.
이 case는 CPU Oracle로 기능을 검증한다. 두 엔진의 동시 계산을 확인한 별도 case는
[Dual Gemmini](../gemmini/README.md), 검색 구현별 차이는
[패치 안내](../../handoff/README.md#5-패치-폴더는-각각-무엇인가)에 있다.
