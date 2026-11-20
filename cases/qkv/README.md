# QKV · 단일 결과 ISA

ACT 튜토리얼의 [QKV.py](QKV.py)를 사용하는 기본 예제다.
[attention.hlo](workloads/attention.hlo)는 BF16 64×64의 Q·K·V를 받아
`softmax(Q @ K.T) @ V`를 계산한다. 기본 실행 구성은 `baseline`이며,
생성한 명령열의 결과를 CPU Oracle에서 검증한다.

## ISA 구성

`Accelerator("QKV")`에 다음 버퍼와 **이동 5종 + 계산 2종 = 7개 명령**을 선언한다.
`d0`는 TAIDL이 자동으로 제공하는 바이트 단위 외부 메모리다.

| 버퍼 | 역할·형태 | 용량 | 주소 단위 |
| --- | --- | --- | --- |
| `d0` | 입력·최종 출력을 담는 HBM, `u8` | 명세에서 고정하지 않음 | 1 byte |
| `d1` | GEMM 입력, `bf16[128,64]` | 16 KiB, 64×64 타일 2개 | 1행 = 128 bytes |
| `d2` | 계산 결과, `bf16[64,64]` | 8 KiB, 64×64 타일 1개 | 1행 = 128 bytes |

| 명령 | 입력 → 출력 | 동작 |
| --- | --- | --- |
| `load_rm` | `d0` → `d1` | `n × 128` bytes를 BF16 `n`행으로 읽기 |
| `load_cm` | `d0` → `d1` | 읽은 BF16 행렬을 전치 |
| `store_rm` | `d1` → `d0` | BF16 `n`행을 `n × 128` bytes로 저장 |
| `store_cm` | `d1` → `d0` | BF16 행렬을 전치한 뒤 저장 |
| `mov` | `d2` → `d1` | 결과 `n`행을 다음 계산의 입력 버퍼로 복사 |
| `gemm` | `d1`의 행렬 2개 → `d2` | BF16 64×64 행렬곱, 결과 1개 |
| `softmax` | `d2` → 같은 `d2` 주소 | 각 행의 64개 원소에 `exp(x) / sum(exp(x))` 적용 |

`@c.n`은 행 수, `@a.*`는 해당 버퍼의 주소다. GEMM 크기는 64×64로 고정되어 있다.
기본 attention 입력은 `n=64`를 사용하며, 전치 명령도 이 정사각 타일을 기준으로 사용한다.
결과를 외부 메모리에 쓰려면 `d2 → mov → d1 → store_rm → d0` 경로를 거친다.

## 실행

공통 [설치](../../README.md#1-설치와-act-소스-준비) 후 저장소 루트에서 실행한다.

```bash
conda activate act
./scripts/run.sh qkv
```

compiler·Oracle 생성, 기본 HLO 컴파일, 수치 검증을 수행한다.
명령열과 검증 기록은 `build/baseline/runs/qkv/attention/`에 생성된다.
단계별 실행은 [루트 README](../../README.md#2-qkv-isa를-실행해-compiler와-oracle-만들기),
두 유닛·다중 결과로 확장한 명세는 [DoubleGEMM](../double_gemm/README.md)을 참고한다.
