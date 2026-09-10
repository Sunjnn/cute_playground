#include "softmax_regcache.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

using std::int64_t;
using std::runtime_error;

namespace {

// One row per CTA, and the whole row cached in registers: 8 x float4 = 32 floats per thread. The
// two-pass kernels stage tiles through shared memory and walk the row twice, which costs 3x the
// DRAM traffic and makes the second walk's hit rate depend on how many CTAs happen to be resident
// at once. Reading each element exactly once instead costs 2x, and because there is no reuse window
// left to miss, the result stops depending on L2 capacity, on row length, and on how large the two
// buffers are in total - the three things that make softmax_multistage's win conditional.
//
// 32 floats per thread is what the register file allows. ptxas reports 47-55 registers across the
// four instantiations below, so the largest one costs 1024 x 53 = 54272 of an SM's 65536 and still
// fits; 64 floats per thread would need around 92 and would not. That caps a row at 32768 floats.
constexpr int kVecPerThread = 8;

constexpr int kMinThreads = 128;
constexpr int kMaxThreads = 1024;

// Below kMinN some threads would hold no element at all; above kMaxN the row stops fitting.
constexpr int kMinN = 4 * kMinThreads;
constexpr int kMaxN = 4 * kVecPerThread * kMaxThreads;

// Deliberately plain CUDA rather than CuTe: there is no shared-memory tile, no copy atom and no
// layout to describe, because nothing is staged. The only partitioning is threadIdx.x * 4.
template <int kThreads>
__global__ void softmax_regcache_device(
    const float *__restrict__ dIn, int ldIn, float *__restrict__ dOut, int ldOut, int n) {
  constexpr int kWarpNum = kThreads / 32;

  // int64_t, not long: long is 32-bit under MSVC, which is this project's target, so a row offset
  // would narrow there and not on Linux.
  const auto row = static_cast<int64_t>(blockIdx.x);
  // Cast once here rather than at each use: threadIdx.x is unsigned, and leaving it that way would
  // promote every index to unsigned and silently convert the int vecNum in the bounds tests below.
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto *src = reinterpret_cast<const float4 *>(dIn + row * ldIn);
  auto *dst = reinterpret_cast<float4 *>(dOut + row * ldOut);
  const auto vecNum = n / 4;

  float4 values[kVecPerThread];
#pragma unroll
  for (auto i = 0; i < kVecPerThread; ++i) {
    const auto vecIndex = threadId + i * kThreads;
    // A padded lane contributes -INFINITY, which fmaxf() absorbs and __expf() turns into exactly
    // 0, so neither reduction needs to know which lanes are padding.
    values[i] =
        vecIndex < vecNum ? src[vecIndex] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  }

  // Separate arrays, so the sum reduction can write while threads are still reading the maximum's
  // result. Sharing one array would need an extra __syncthreads() between the two.
  __shared__ float sMax[kWarpNum];
  __shared__ float sSum[kWarpNum];

  auto rowMax = -INFINITY;
#pragma unroll
  for (const auto &value : values) {
    rowMax = fmaxf(rowMax, fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)));
  }
#pragma unroll
  for (auto offset = 16; offset > 0; offset /= 2) {
    rowMax = fmaxf(rowMax, __shfl_xor_sync(0xffffffffu, rowMax, offset));
  }
  const auto warpId = threadId / 32;
  if (threadId % 32 == 0) {
    sMax[warpId] = rowMax;
  }

  __syncthreads();
  rowMax = sMax[0];
#pragma unroll
  for (auto i = 1; i < kWarpNum; ++i) {
    rowMax = fmaxf(rowMax, sMax[i]);
  }

  // Subtracting the row maximum first is what keeps exp() away from overflow. It is free here
  // because the row is already in registers, whereas softmax() and softmax_multistage() would have
  // to walk the row an extra time to learn it.
  auto rowSum = 0.0f;
#pragma unroll
  for (auto &value : values) {
    value.x = __expf(value.x - rowMax);
    value.y = __expf(value.y - rowMax);
    value.z = __expf(value.z - rowMax);
    value.w = __expf(value.w - rowMax);
    rowSum += value.x + value.y + value.z + value.w;
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

  // One reciprocal per row. Left inline, `x / rowSum` makes ptxas re-emit MUFU.RCP at every one of
  // the unrolled sites even though rowSum is loop-invariant.
  const auto invSum = 1.0f / rowSum;
#pragma unroll
  for (auto i = 0; i < kVecPerThread; ++i) {
    const auto vecIndex = threadId + i * kThreads;
    if (vecIndex < vecNum) {
      dst[vecIndex] = make_float4(
          values[i].x * invSum, values[i].y * invSum, values[i].z * invSum, values[i].w * invSum);
    }
  }
}

} // namespace

void softmax_regcache(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  // Reading a row as float4 needs the row length and both strides to be whole vectors. cudaMalloc's
  // 256-byte alignment then makes every row start 16-byte aligned for free.
  if (n % 4 != 0 || ldIn % 4 != 0 || ldOut % 4 != 0) {
    throw runtime_error("softmax_regcache needs n, ldIn and ldOut to be multiples of 4");
  }
  if (n < kMinN || n > kMaxN) {
    throw runtime_error("softmax_regcache holds one row in registers, so 512 <= n <= 32768");
  }

  // Smallest thread count that still gives every thread at least one float4. Going larger would
  // only add padded lanes; the register budget per thread is fixed at 32 floats either way.
  auto threads = kMinThreads;
  while (threads < kMaxThreads && 4 * kVecPerThread * threads < n) {
    threads *= 2;
  }

  const dim3 dimBlock(threads);
  const dim3 dimGrid(m);
  switch (threads) {
  case 128:
    softmax_regcache_device<128><<<dimGrid, dimBlock>>>(dIn, ldIn, dOut, ldOut, n);
    break;
  case 256:
    softmax_regcache_device<256><<<dimGrid, dimBlock>>>(dIn, ldIn, dOut, ldOut, n);
    break;
  case 512:
    softmax_regcache_device<512><<<dimGrid, dimBlock>>>(dIn, ldIn, dOut, ldOut, n);
    break;
  default:
    softmax_regcache_device<1024><<<dimGrid, dimBlock>>>(dIn, ldIn, dOut, ldOut, n);
    break;
  }
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
