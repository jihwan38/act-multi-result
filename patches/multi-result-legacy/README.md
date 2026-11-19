# Gemmini용 이전 다중 결과 구현

`act-parallelism`의 [실험 05, commit 6164b44](https://github.com/jihwan38/act-parallelism/tree/6164b44c8003f7403903930ef3cc938816121ba3/experiments/05_dual_gemmini/patches)에서
가져온 **Backend 9개 + Oracle 3개**다. 패치 내용은 원본과 같다.

적용 순서는 **고정 upstream → `baseline` → `multi-result-legacy`**다.
현재 `multi-result` 패치 위에 추가하는 구성이 아니다. 네 ACT upstream의 고정 커밋과
`baseline` 패치는 이전 저장소와 같아서 중복 복사하지 않는다.

- Backend 0001–0007: 이전 다중 결과 표현·후보 발견·추출·주소 할당.
- Backend 0008: signed 정수 타입의 Rust 이름 `I8/I32` 수정.
- Backend 0009 + Oracle 0003: 정수 상수·`minimum`·`maximum`과 포화 저장의 의미 처리.
- Oracle 0001–0002: concatenate와 구조화된 입출력 보존.

이 버전의 `GemminiDual.py`는 두 `ROOT`를 쓰는 옛 문법을 유지한다.
현재 `multi-result`의 `ROOT tuple(...)` 문법과 섞지 않는다.
이전 실험 보존을 위한 구성이며, 현재 다중 결과 구현으로 Gemmini를 이식한 것은 아니다.

실행은 [cases/gemmini](../../cases/gemmini/README.md), 하드웨어 관측 패치는
[hardware/patches](../../cases/gemmini/hardware/patches)에 있다.
