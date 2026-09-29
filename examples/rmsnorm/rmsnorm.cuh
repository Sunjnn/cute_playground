#ifndef EXAMPLES_RMSNORM_RMSNORM_CUH
#define EXAMPLES_RMSNORM_RMSNORM_CUH

void rmsnorm(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps);

#endif
