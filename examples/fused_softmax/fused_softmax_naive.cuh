#ifndef EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_NAIVE_CUH
#define EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_NAIVE_CUH

#include <cuda_fp16.h>

void fused_softmax_naive(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    __half *dP,
    int ldP,
    float scale,
    bool causal);

#endif
