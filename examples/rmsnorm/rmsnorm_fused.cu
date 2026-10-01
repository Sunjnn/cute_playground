#include "rmsnorm_fused.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

using std::int64_t;
using std::runtime_error;

namespace {

// One row per CTA and the whole row cached in registers: 8 x float4 = 32 floats per thread. The
// two-pass kernels walk the row twice and pay 3x its bytes; RMSNorm - unlike softmax, which has to
// learn the row maximum before it can exponentiate - needs nothing from a first walk that a second
// one could not have produced, so the row can be read exactly once and written exactly once for 2x.
//
// The same register ceiling as softmax_regcache applies: 32 floats per thread is what fits. ptxas
// reports 47-55 registers there, so the largest instantiation costs 1024 x 53 = 54272 of an SM's
// 65536 and still leaves room; 64 floats per thread would need around 92 and would not. That caps
// a row at 32768 floats.
//
// The weight vector is deliberately not cached alongside the row. Doing so would double the
// footprint to 64 floats per thread and blow that budget, and it would buy nothing: every CTA asks
// for the same n floats, so after the first few rows the vector is L2-resident and the store loop's
// read of it costs L2 bandwidth rather than DRAM. The row, by contrast, is read by exactly one CTA
// and can never be served from cache a second time - which is why caching that one matters.
constexpr int kVecPerThread = 8;

constexpr int kMinThreads = 128;
constexpr int kMaxThreads = 1024;

// Below kMinN some threads would hold no element at all; above kMaxN the row stops fitting.
constexpr int kMinN = 4 * kMinThreads;
constexpr int kMaxN = 4 * kVecPerThread * kMaxThreads;

// Deliberately plain CUDA rather than CuTe, as in softmax_regcache: there is no shared-memory tile,
// no copy atom and no layout to describe, because nothing is staged. The only partitioning is
// threadIdx.x * 4.
template <int kThreads>
__global__ void rmsnorm_fused_device(
    const float *__restrict__ dIn,
    int ldIn,
    const float *__restrict__ dWeight,
    float *__restrict__ dOut,
    int ldOut,
    int n,
    float eps) {
  constexpr int kWarpNum = kThreads / 32;

  // int64_t, not long: long is 32-bit under MSVC, which is one of this project's targets, so a row
  // offset would narrow there and not on Linux.
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
    // A padded lane contributes zero, which adds nothing to a sum of squares, so neither the
    // reduction nor the store needs to know which lanes are padding. (softmax_regcache has to use
    // -INFINITY here instead, because its first reduction is a maximum and a zero would win it.)
    values[i] = vecIndex < vecNum ? src[vecIndex] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  __shared__ float sSums[kWarpNum];

  auto sqrSum = 0.0f;
#pragma unroll
  for (const auto &value : values) {
    sqrSum += value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
  }
#pragma unroll
  for (auto offset = 16; offset > 0; offset /= 2) {
    sqrSum += __shfl_xor_sync(0xffffffffu, sqrSum, offset);
  }
  const auto warpId = threadId / 32;
  if (threadId % 32 == 0) {
    sSums[warpId] = sqrSum;
  }

  __syncthreads();
  sqrSum = sSums[0];
#pragma unroll
  for (auto i = 1; i < kWarpNum; ++i) {
    sqrSum += sSums[i];
  }

  // One reciprocal square root per row rather than one division per element.
  const auto scale = rsqrtf(sqrSum / static_cast<float>(n) + eps);

  const auto *weight = reinterpret_cast<const float4 *>(dWeight);
#pragma unroll
  for (auto i = 0; i < kVecPerThread; ++i) {
    const auto vecIndex = threadId + i * kThreads;
    if (vecIndex < vecNum) {
      const auto w = __ldg(weight + vecIndex);
      dst[vecIndex] = make_float4(
          values[i].x * scale * w.x,
          values[i].y * scale * w.y,
          values[i].z * scale * w.z,
          values[i].w * scale * w.w);
    }
  }
}

} // namespace

void rmsnorm_fused(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  // Reading a row as float4 needs the row length and all three pointers' strides to be whole
  // vectors. cudaMalloc's 256-byte alignment then makes every row start 16-byte aligned for free,
  // dWeight included.
  if (n % 4 != 0 || ldIn % 4 != 0 || ldOut % 4 != 0) {
    throw runtime_error("rmsnorm_fused needs n, ldIn and ldOut to be multiples of 4");
  }
  if (n < kMinN || n > kMaxN) {
    throw runtime_error("rmsnorm_fused holds one row in registers, so 512 <= n <= 32768");
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
    rmsnorm_fused_device<128><<<dimGrid, dimBlock>>>(dIn, ldIn, dWeight, dOut, ldOut, n, eps);
    break;
  case 256:
    rmsnorm_fused_device<256><<<dimGrid, dimBlock>>>(dIn, ldIn, dWeight, dOut, ldOut, n, eps);
    break;
  case 512:
    rmsnorm_fused_device<512><<<dimGrid, dimBlock>>>(dIn, ldIn, dWeight, dOut, ldOut, n, eps);
    break;
  default:
    rmsnorm_fused_device<1024><<<dimGrid, dimBlock>>>(dIn, ldIn, dWeight, dOut, ldOut, n, eps);
    break;
  }
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
