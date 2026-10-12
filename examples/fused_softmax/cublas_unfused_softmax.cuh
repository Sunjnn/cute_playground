#ifndef EXAMPLES_FUSED_SOFTMAX_CUBLAS_UNFUSED_SOFTMAX_CUH
#define EXAMPLES_FUSED_SOFTMAX_CUBLAS_UNFUSED_SOFTMAX_CUH

#include <cuda_fp16.h>

void cublas_unfused_softmax(
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
