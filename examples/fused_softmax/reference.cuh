#ifndef EXAMPLES_FUSED_SOFTMAX_REFERENCE_CUH
#define EXAMPLES_FUSED_SOFTMAX_REFERENCE_CUH

#include <cccl/thrust/host_vector.h>
#include <cuda_fp16.h>

// The double-precision answer the implementations are held to, for S = Q K^T scaled, causally
// masked, and row-wise softmaxed into P.
//
// Both operands are widened to double *after* being read as fp16, so what this describes is exactly
// what a kernel with fp16 operands and an fp32 accumulator can be measured against. Widening first
// and quantizing after would instead describe an fp32 GEMM, and every implementation would fail
// against it for reasons that have nothing to do with whether it is correct.
struct Reference {
  // m x n with leading dimension n. Masked positions hold -INFINITY in scores and exactly 0 in
  // probs. scores is a debugging aid, not part of the pass criterion: when an implementation fails,
  // comparing against scores separates "the GEMM is wrong" from "the row reduction is wrong", which
  // is otherwise one indistinguishable number.
  thrust::host_vector<float> scores;
  thrust::host_vector<float> probs;
};

Reference make_reference(
    int m,
    int n,
    int k,
    const thrust::host_vector<__half> &hQ,
    int ldQ,
    const thrust::host_vector<__half> &hK,
    int ldK,
    float scale,
    bool causal);

// Largest deviation of hOut from Reference::probs. A NaN deviation sticks, so an implementation
// that leaves part of the output buffer untouched cannot be rescued by the elements it did write.
//
// A masked position is exactly zero in the reference, where a relative error is undefined; those
// are measured absolutely instead. P lives in [0, 1], so the two stay on the same scale and one
// threshold covers both.
double max_rel_diff(
    const Reference &ref, int m, int n, const thrust::host_vector<__half> &hOut, int ldOut);

#endif
