#include "rmsnorm_naive.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

using std::int64_t;
using std::runtime_error;

namespace {

constexpr int kThreads = 256;
constexpr int kWarpNum = kThreads / 32;

// The straightforward version, and the reference the CuTe kernel's speedup is measured against:
// one row per CTA, each thread walking its columns with a stride of kThreads, straight through
// global memory with no shared-memory tile, no cp.async and no vector wider than the compiler
// happens to find. It reads the row twice and the weight once, so it moves the same three times
// the row's bytes the CuTe kernel does - the difference between them is in how those bytes are
// fetched, not in how many there are.
//
// Deliberately plain CUDA rather than CuTe: there is no layout to describe, since nothing is
// staged and the only partitioning is threadIdx.x.
__global__ void rmsnorm_naive_device(
    const float *__restrict__ dIn,
    int ldIn,
    const float *__restrict__ dWeight,
    float *__restrict__ dOut,
    int ldOut,
    int n,
    float eps) {
  __shared__ float sSums[kWarpNum];

  // int64_t, not long: long is 32-bit under MSVC, which is one of this project's targets, so a
  // row offset would narrow there and not on Linux.
  const auto row = static_cast<int64_t>(blockIdx.x);
  // Cast once here rather than at each use: threadIdx.x is unsigned, so an index derived from it
  // would be unsigned too and `col < n` would compare against a converted n.
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto *src = dIn + row * ldIn;
  auto *dst = dOut + row * ldOut;

  auto sqrSum = 0.0f;
  for (auto col = threadId; col < n; col += kThreads) {
    const auto value = src[col];
    sqrSum += value * value;
  }
#pragma unroll
  for (auto offset = 16; offset > 0; offset /= 2) {
    sqrSum += __shfl_xor_sync(0xffffffffu, sqrSum, offset);
  }
  if (threadId % 32 == 0) {
    sSums[threadId / 32] = sqrSum;
  }

  __syncthreads();
  sqrSum = 0.0f;
#pragma unroll
  for (const auto warpSum : sSums) {
    sqrSum += warpSum;
  }

  const auto scale = rsqrtf(sqrSum / static_cast<float>(n) + eps);
  for (auto col = threadId; col < n; col += kThreads) {
    dst[col] = src[col] * scale * dWeight[col];
  }
}

} // namespace

void rmsnorm_naive(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m);
  rmsnorm_naive_device<<<dimGrid, dimBlock>>>(dIn, ldIn, dWeight, dOut, ldOut, n, eps);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
