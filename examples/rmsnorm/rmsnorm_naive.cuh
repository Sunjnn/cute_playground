#ifndef EXAMPLES_RMSNORM_RMSNORM_NAIVE_CUH
#define EXAMPLES_RMSNORM_RMSNORM_NAIVE_CUH

void rmsnorm_naive(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps);

#endif
