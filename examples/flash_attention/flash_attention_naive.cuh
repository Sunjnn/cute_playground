#ifndef EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_NAIVE_CUH
#define EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_NAIVE_CUH

#include <cuda_fp16.h>

// Single-head, row-major Q[m, k], K[n, k], V[n, k] and O[m, k], with leading dimensions in
// elements. Half inputs/output and float scores, probabilities and accumulators. Causal attention
// exposes j <= i + n - m, and requires m <= n. Uses the default stream and waits for completion.
// Q's row, the score row and reduction scratch together must fit in 48 KiB of shared memory.
void flash_attention_naive(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    const __half *dV,
    int ldV,
    __half *dO,
    int ldO,
    float scale,
    bool causal);

#endif
