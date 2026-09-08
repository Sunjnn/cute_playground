#ifndef EXAMPLES_TRANSPOSE_CUDNN_TRANSPOSE_CUH
#define EXAMPLES_TRANSPOSE_CUDNN_TRANSPOSE_CUH

void transpose_cudnn(int m, int n, const float *dIn, int ldIn, float *dOut, int ldOut);

#endif
