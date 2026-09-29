#include "cublas_rmsnorm.cuh"

#include <cccl/thrust/execution_policy.h>
#include <cccl/thrust/fill.h>
#include <cccl/thrust/for_each.h>
#include <cccl/thrust/iterator/counting_iterator.h>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

using std::runtime_error;
using std::size_t;
using thrust::counting_iterator;
using thrust::device;
using thrust::fill_n;
using thrust::for_each_n;

namespace {

void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw runtime_error(cublasGetStatusString(status));
  }
}

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

// cublasCreate() allocates a workspace and costs milliseconds - orders of magnitude more than one
// rmsnorm - so the handle is built once for the whole process instead of per call.
struct Handle {
  cublasHandle_t value = nullptr;

  Handle() { check(cublasCreate(&value)); }

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&) = delete;
  Handle &operator=(Handle &&) = delete;

  ~Handle() { cublasDestroy(value); }
};

} // namespace

// Row-wise RMS normalization: out[i][j] = in[i][j] * rsqrt(mean_j(in[i][j]^2) + eps) * weight[j].
//
// What cuBLAS alone gets you, and the point of this row is that it is not much. cuBLAS has no
// squared reduction, and a matrix-vector product cannot square on the way in, so the row's sum of
// squares takes three steps: a thrust pass that materializes x^2 into a second m x n buffer, one
// cublasSgemv against a vector of ones to collapse each row, and a thrust pass to apply the scale.
// Five walks over m x n against the CuTe kernel's three, plus m * n * 4 bytes of scratch - 128 MB
// at the default shape - that no fused kernel needs.
//
// The one thing cuBLAS does buy is the mean for free: sgemv's alpha scales the accumulation, so
// passing 1/n makes the reduction hand back mean_j(x^2) directly rather than the sum.
void rmsnorm_cublas(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  static auto sHandle = Handle();

  // cudaMallocAsync and the harness's UINT64_MAX release threshold, as in rmsnorm_cub: the
  // default thrust allocator would cudaMalloc/cudaFree per call and measure the allocator.
  float *dSqr = nullptr;
  float *dOnes = nullptr;
  float *dSums = nullptr;
  check(cudaMallocAsync(
      &dSqr, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dOnes, static_cast<size_t>(n) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dSums, static_cast<size_t>(m) * sizeof(float), nullptr));
  fill_n(device, dOnes, n, 1.0f);

  auto rows = counting_iterator<int>(0);
  auto total = m * n;

  // dSqr is written compactly (leading dimension n) whatever ldIn is, so the sgemv below never has
  // to know about the input's padding.
  for_each_n(device, rows, total, [dIn, dSqr, n, ldIn] __device__(int t) {
    auto row = t / n;
    auto col = t - row * n;
    auto value = dIn[row * ldIn + col];
    dSqr[t] = value * value;
  });

  const auto alpha = 1.0f / static_cast<float>(n);
  const auto beta = 0.0f;
  // dSqr is m x n row-major with leading dimension n, which is the same memory as an n x m
  // column-major matrix; that matrix transposed times the ones vector is the m row sums.
  check(cublasSgemv(sHandle.value, CUBLAS_OP_T, n, m, &alpha, dSqr, n, dOnes, 1, &beta, dSums, 1));

  for_each_n(
      device, rows, total, [dIn, dOut, dWeight, dSums, n, ldIn, ldOut, eps] __device__(int t) {
        auto row = t / n;
        auto col = t - row * n;
        dOut[row * ldOut + col] = dIn[row * ldIn + col] * rsqrtf(dSums[row] + eps) * dWeight[col];
      });

  check(cudaFreeAsync(dSqr, nullptr));
  check(cudaFreeAsync(dOnes, nullptr));
  check(cudaFreeAsync(dSums, nullptr));
}
