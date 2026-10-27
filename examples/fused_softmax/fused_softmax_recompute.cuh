#ifndef EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_RECOMPUTE_CUH
#define EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_RECOMPUTE_CUH

#include <cuda_fp16.h>

void fused_softmax_recompute(
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
