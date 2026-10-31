#ifndef EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_TMA_CUH
#define EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_TMA_CUH

#include <cuda_fp16.h>

// Single-head O = softmax(scale * Q K^T) V. All matrices are row-major fp16; Q/O are (m, k),
// K/V are (n, k). Statistics and both GEMM accumulators are fp32. This variant streams the key
// tiles with TMA (cp.async.bulk.tensor) instead of cp.async; the tiling, softmax and PV path are
// the ones in flash_attention.cuh. Needs m % 32 == 0, n % 64 == 0, k = 64 or 128 and
// vector-aligned inputs.
void flash_attention_tma(
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
