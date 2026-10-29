#ifndef EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_UPSTREAM_CUH
#define EXAMPLES_FLASH_ATTENTION_FLASH_ATTENTION_UPSTREAM_CUH

#include <cuda_fp16.h>

// Official Dao-AILab FlashAttention-2 v2.8.3.post2, nonsplit fp16 forward. Q/O are (m, k),
// K/V are (n, k), row-major. Only k = 64/128 is compiled; sequence tails are supported.
// Requires positive fp32-safe scale and 16-byte-aligned Q/K/V/O rows. The optional private
// implementation uses upstream CUTLASS and CUDA LibTorch; neither leaks through this interface.
void flash_attention_upstream(
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
