#ifndef EXAMPLES_RMSNORM_CUDNN_RMSNORM_CUH
#define EXAMPLES_RMSNORM_CUDNN_RMSNORM_CUH

void rmsnorm_cudnn(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps);

#endif
