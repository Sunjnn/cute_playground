#include "fused_softmax_naive.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

using std::int64_t;
using std::runtime_error;

namespace {

constexpr int kThreads = 256;
constexpr int kWarpNum = kThreads / 32;

// 48 KB is what a block gets without opting in through cudaFuncSetAttribute, and this kernel stages
// Q's row and the whole row of scores in it - one float each, so k + n has to stay under 12288.
// kMaxSmemBytes keeps that limit in one place instead of implying it from a bound on n alone.
constexpr int kMaxSmemBytes = 48 * 1024;

// The straightforward version, and the floor the fused kernels' speedup is measured against: one
// row per CTA, SIMT multiply-add instead of tensor cores, the row's scores staged in shared memory,
// and three walks over them - one to find the maximum, one to exponentiate and sum, one to
// normalize and store. Nothing is kept in registers across a walk and no load is overlapped with
// any compute.
//
// Deliberately plain CUDA rather than CuTe: there is no tile and no copy atom to describe, since
// the only partitioning is threadIdx.x.
//
// Q's row is widened to float on the way into shared memory rather than left as fp16. That doubles
// its footprint for no bandwidth benefit - it is read from global exactly once either way - but it
// takes a __half2float() out of the k-long inner loop and, more usefully, lets the scores and the
// query share one float array so neither needs an alignment offset computed around the other.
__global__ void fused_softmax_naive_device(
    const __half *__restrict__ dQ,
    int ldQ,
    const __half *__restrict__ dK,
    int ldK,
    __half *__restrict__ dP,
    int ldP,
    int m,
    int n,
    int k,
    float scale,
    bool causal) {
  extern __shared__ float sRaw[];
  float *sQuery = sRaw;
  float *sScores = sRaw + k;

  __shared__ float sMax[kWarpNum];
  __shared__ float sSum[kWarpNum];

  // int64_t, not long: long is 32-bit under MSVC, which is one of this project's targets, so a row
  // offset would narrow there and not on Linux.
  const auto row = static_cast<int64_t>(blockIdx.x);
  // Cast once here rather than at each use: threadIdx.x is unsigned, and leaving it that way would
  // promote every index to unsigned and silently convert the int bounds below.
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto *query = dQ + row * ldQ;
  auto *prob = dP + row * ldP;

  for (auto t = threadId; t < k; t += kThreads) {
    sQuery[t] = __half2float(query[t]);
  }
  __syncthreads();

  // Column j is visible iff j <= row + n - m, which keeps the last query aligned with the last key
  // when m < n. main.cu rejects m > n, so a row here always sees at least one column and the sum
  // below is never a division by zero.
  auto visible = n;
  if (causal) {
    visible = min(max(static_cast<int>(row) + n - m + 1, 0), n);
  }

  for (auto col = threadId; col < n; col += kThreads) {
    // A masked column carries -INFINITY, which fmaxf() absorbs and __expf() turns into exactly 0,
    // so neither reduction below needs to know which columns were masked.
    auto score = -INFINITY;
    if (col < visible) {
      const auto *key = dK + static_cast<int64_t>(col) * ldK;
      auto acc = 0.0f;
      for (auto t = 0; t < k; ++t) {
        acc += sQuery[t] * __half2float(key[t]);
      }
      score = acc * scale;
    }
    sScores[col] = score;
  }
  __syncthreads();

  auto rowMax = -INFINITY;
  for (auto col = threadId; col < n; col += kThreads) {
    rowMax = fmaxf(rowMax, sScores[col]);
  }
#pragma unroll
  for (auto offset = 16; offset > 0; offset /= 2) {
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xffffffffu, rowMax, offset));
  }
  const auto warpId = threadId / 32;
  if (threadId % 32 == 0) {
    sMax[warpId] = rowMax;
  }

  // Separate arrays, so the sum reduction can write while threads are still reading the maximum's
  // result. Sharing one array would need an extra __syncthreads() between the two.
  __syncthreads();
  rowMax = sMax[0];
#pragma unroll
  for (auto i = 1; i < kWarpNum; ++i) {
    rowMax = fmaxf(rowMax, sMax[i]);
  }

  auto rowSum = 0.0f;
  for (auto col = threadId; col < n; col += kThreads) {
    auto value = __expf(sScores[col] - rowMax);
    sScores[col] = value;
    rowSum += value;
  }
#pragma unroll
  for (auto offset = 16; offset > 0; offset /= 2) {
    rowSum += __shfl_xor_sync(0xffffffffu, rowSum, offset);
  }
  if (threadId % 32 == 0) {
    sSum[warpId] = rowSum;
  }

  __syncthreads();
  rowSum = sSum[0];
#pragma unroll
  for (auto i = 1; i < kWarpNum; ++i) {
    rowSum += sSum[i];
  }

  // One reciprocal per row rather than one division per element.
  const auto invSum = 1.0f / rowSum;
  for (auto col = threadId; col < n; col += kThreads) {
    prob[col] = __float2half(sScores[col] * invSum);
  }
}

} // namespace

void fused_softmax_naive(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    __half *dP,
    int ldP,
    float scale,
    bool causal) {
  const auto smem = (k + n) * static_cast<int>(sizeof(float));
  if (smem > kMaxSmemBytes) {
    throw runtime_error(
        "fused_softmax_naive stages Q's row and one row of scores in shared memory, "
        "so k + n must fit in 12288 floats");
  }

  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m);
  fused_softmax_naive_device<<<dimGrid, dimBlock, smem>>>(
      dQ, ldQ, dK, ldK, dP, ldP, m, n, k, scale, causal);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
