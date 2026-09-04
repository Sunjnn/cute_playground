#ifndef EXAMPLES_SOFTMAX_CUDNN_SOFTMAX_CUH
#define EXAMPLES_SOFTMAX_CUDNN_SOFTMAX_CUH

void softmax_cudnn(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut);

#endif
