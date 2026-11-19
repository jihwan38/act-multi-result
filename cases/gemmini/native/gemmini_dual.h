#ifndef ACT_GEMMINI_DUAL_H
#define ACT_GEMMINI_DUAL_H

/* The generated lane-0 header and gemmini.h share the same guard. Use the
 * actual elaborated parameters, not a hand-written replacement header. */
#include "include/gemmini_lane0_params.h"
#include "include/gemmini.h"

_Static_assert(DIM == 16 && ADDR_LEN == 32, "Unsupported native geometry");
_Static_assert(sizeof(elem_t) == 1 && sizeof(acc_t) == 4, "Expected INT8/INT32");
_Static_assert(BANK_NUM * BANK_ROWS == 16384 && ACC_ROWS == 1024,
               "TAIDL storage must match the elaborated RTL");

/* func must remain an immediate. These are the unchanged native Gemmini
 * encodings; only the RoCC custom opcode selects a different engine. */
#define act_issue(lane, func, rs1, rs2) do { \
    asm volatile("" ::: "memory"); \
    if ((lane) == 0) { \
        ROCC_INSTRUCTION_RS1_RS2(3, (uint64_t)(rs1), (uint64_t)(rs2), func); \
    } else { \
        ROCC_INSTRUCTION_RS1_RS2(2, (uint64_t)(rs1), (uint64_t)(rs2), func); \
    } \
    asm volatile("" ::: "memory"); \
} while (0)

static inline void act_wait(void) {
    /* Rocket waits for the OR of both RoCC busy signals before this retires.
     * The clobber also prevents the C compiler from moving memory accesses. */
    asm volatile("fence" ::: "memory");
}

static inline uint64_t act_cycles(void) {
    uint64_t value;
    asm volatile("csrr %0, mcycle" : "=r"(value) :: "memory");
    return value;
}

static inline uint64_t act_tile(uint32_t address) {
    return ((uint64_t)DIM << 48) | ((uint64_t)DIM << 32) | address;
}

static inline void act_configure(unsigned lane) {
    act_issue(lane, k_FLUSH, 0, 0);
    /* WS, no transpose/activation/shift; both strides and all scales = 1. */
    const uint64_t acc_scale = acc_scale_t_to_acc_scale_t_bits(ACC_SCALE_IDENTITY);
    const uint64_t in_scale = scale_t_to_scale_t_bits(MVIN_SCALE_IDENTITY);
    act_issue(lane, k_CONFIG,
              (acc_scale << 32) | (1UL << 16) | (WEIGHT_STATIONARY << 2) | CONFIG_EX,
              1UL << 48);
    act_issue(lane, k_CONFIG,
              (in_scale << 32) | (DIM << 16) | (1 << 8) | CONFIG_LD, DIM);
    act_issue(lane, k_CONFIG, CONFIG_ST, (acc_scale << 32) | (DIM * sizeof(acc_t)));
}

static inline void act_mvin(unsigned lane, const void *memory, uint32_t output) {
    act_issue(lane, k_MVIN, memory, act_tile(output));
    act_wait();
}

static inline void act_preload(unsigned lane, uint32_t weights, uint32_t output) {
    /* bit31: accumulator; bit30=0: overwrite, NOT accumulate. */
    act_issue(lane, k_PRELOAD, act_tile(weights), act_tile(output | (1U << 31)));
}

static inline void act_compute(unsigned lane, uint32_t input) {
    /* In WS, weights came from preload; GARBAGE_ADDR supplies zero bias. */
    act_issue(lane, k_COMPUTE_PRELOADED, act_tile(input), act_tile(GARBAGE_ADDR));
}

static inline void act_gemm(unsigned lane, uint32_t left, uint32_t right, uint32_t output) {
    act_preload(lane, right, output);
    act_compute(lane, left);
    act_wait();
}

static inline void act_double_gemm(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1,
                                   uint32_t c0, uint32_t c1) {
    /* Compiler macro, not a new hardware opcode. No wait between the engines. */
    act_preload(0, b0, c0);
    act_preload(1, b1, c1);
    act_compute(0, a0);
    act_compute(1, a1);
    act_wait();
}

static inline void act_mvout(unsigned lane, uint32_t input, void *memory) {
    /* bit29 requests full INT32 accumulator rows (no INT8 saturation). */
    act_issue(lane, k_MVOUT, memory, act_tile(input | (1U << 31) | (1U << 29)));
    act_wait();
}

static inline void act_requantize_store(unsigned lane, uint32_t input, void *memory) {
    /* Identity scale followed by Gemmini's signed INT8 saturation. The narrow
     * store has a different DRAM row stride; restore full-width configuration. */
    const uint64_t scale = acc_scale_t_to_acc_scale_t_bits(ACC_SCALE_IDENTITY);
    act_issue(lane, k_CONFIG, CONFIG_ST, (scale << 32) | DIM);
    act_issue(lane, k_MVOUT, memory, act_tile(input | (1U << 31)));
    act_wait();
    act_issue(lane, k_CONFIG, CONFIG_ST, (scale << 32) | (DIM * sizeof(acc_t)));
    act_wait();
}

#endif
