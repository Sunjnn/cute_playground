#ifndef EXAMPLES_RMSNORM_RMSNORM_FUSED_CUH
#define EXAMPLES_RMSNORM_RMSNORM_FUSED_CUH

void rmsnorm_fused(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps);

#endif
