#ifndef EXAMPLES_FLASH_ATTENTION_CUBLAS_UNFUSED_ATTENTION_CUH
#define EXAMPLES_FLASH_ATTENTION_CUBLAS_UNFUSED_ATTENTION_CUH

#include <cuda_fp16.h>

// Single-head, row-major Q[m, k], K[n, k], V[n, k] and O[m, k], with leading dimensions in
// elements. Two cuBLAS GEMMs accumulate in float; the standalone masked softmax rounds P to half
// before PV. Supports arbitrary positive dimensions/strides, including sequence tails. Causal
// attention exposes j <= i + n - m and requires m <= n. Uses the default stream and waits for
// completion; scratch is a compact float score matrix and a compact half probability matrix.
void cublas_unfused_attention(
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
