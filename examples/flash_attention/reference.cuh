#ifndef EXAMPLES_FLASH_ATTENTION_REFERENCE_CUH
#define EXAMPLES_FLASH_ATTENTION_REFERENCE_CUH

#include <cccl/thrust/host_vector.h>
#include <cuda_fp16.h>

// Direct CPU double QK, stable softmax and PV, widening already-quantized half inputs rather than
// comparing kernels with an unquantized float problem. Only the compact row-major O[m, k] answer
// is retained; score/probability scratch is reused one row at a time.
struct Reference {
  thrust::host_vector<double> output;
};

Reference make_reference(
    int m,
    int n,
    int k,
    const thrust::host_vector<__half> &hQ,
    int ldQ,
    const thrust::host_vector<__half> &hK,
    int ldK,
    const thrust::host_vector<__half> &hV,
    int ldV,
    float scale,
    bool causal);

// max(abs(got - want) / max(abs(want), 1.0)) over the logical output, ignoring stride padding.
// Any nonfinite expected/actual value or deviation returns infinity, never a passing comparison.
double max_rel_diff(
    const Reference &ref, int m, int k, const thrust::host_vector<__half> &hOut, int ldOut);

#endif
