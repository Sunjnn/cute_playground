#include "fmha_softmax.cuh"

#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

using std::runtime_error;

namespace {

// One CTA per row and 512 columns per tile - the thread count and tile width softmax.cu uses, so
// both kernels move the same amount of data per row.
constexpr int kThreads = 128;
constexpr int kFloatsPerThread = 4;
constexpr int kTile = kThreads * kFloatsPerThread;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

// exp(x) evaluated as exp2(x * log2(e)), the form every exponential in fmha_collective_softmax.hpp
// takes: it folds the softmax scale into the exponent and lands on the hardware ex2 instruction.
constexpr float kLog2e = 1.4426950408889634f;

__device__ inline float exp2e(float x) { return exp2f(x * kLog2e); }

// The register-resident (max, sum) pair fmha_collective_softmax.hpp carries across its K loop.
// There it tracks one row of an MMA accumulator that never leaves registers; here it tracks the
// columns of a global-memory row that one thread owns.
struct SoftmaxState {
  float max = -INFINITY;
  float sum = 0.0f;

  // Online rescale: a larger maximum invalidates every exponential accumulated so far, so the
  // running sum is multiplied by exp(max_old - max_new) before the incoming partial sum - already
  // relative to its own maximum - is folded in.
  __device__ void merge(float otherMax, float otherSum) {
    auto newMax = fmaxf(max, otherMax);
    // exp2e(-inf - (-inf)) is NaN, and NaN * 0 is still NaN, so an empty state has to be scaled by
    // a literal zero. This is reachable: for n < kTile whole warps own no column at all.
    // fmha_collective_softmax.hpp guards the same case with an explicit -INFINITY test.
    auto empty = newMax == -INFINITY;
    sum = empty ? 0.0f : sum * exp2e(max - newMax) + otherSum * exp2e(otherMax - newMax);
    max = newMax;
  }

  __device__ void merge_vector(const float4 &v) {
    auto tileMax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    merge(
        tileMax,
        exp2e(v.x - tileMax) + exp2e(v.y - tileMax) + exp2e(v.z - tileMax) + exp2e(v.w - tileMax));
  }

  // A lone element is its own maximum, so its partial sum is exp(0) = 1.
  __device__ void merge_element(float x) { merge(x, 1.0f); }
};

__device__ inline float4 scale_vector(const float4 &v, float max, float invSum) {
  return {
      exp2e(v.x - max) * invSum,
      exp2e(v.y - max) * invSum,
      exp2e(v.z - max) * invSum,
      exp2e(v.w - max) * invSum};
}

__global__ void softmax_fmha_device(
    const float *__restrict__ dIn, float *__restrict__ dOut, int n, int ldIn, int ldOut) {
  const auto *in = dIn + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(ldIn);
  auto *out = dOut + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(ldOut);

  __shared__ float sMax[kWarps];
  __shared__ float sSum[kWarps];

  // Column of the first element this thread reads, constant across tiles. Offsets are ptrdiff_t so
  // that the tile arithmetic below is already in the width a pointer difference needs.
  const auto col = static_cast<ptrdiff_t>(threadIdx.x) * kFloatsPerThread;
  const auto fullTiles = static_cast<ptrdiff_t>(n) / kTile;

  auto state = SoftmaxState();
  for (auto tile = ptrdiff_t{0}; tile < fullTiles; ++tile) {
    const auto base = tile * kTile + col;
    state.merge_vector(*reinterpret_cast<const float4 *>(in + base));
  }
  const auto tail = fullTiles * kTile + col;
  for (auto i = 0; i < kFloatsPerThread && tail + i < n; ++i) {
    state.merge_element(in[tail + i]);
  }

  // Butterfly over the lanes, then over the warps through shared memory.
  // fmha_collective_softmax.hpp needs only the first half because an MMA row is spread over a
  // single warp; a 512-column tile spans four.
  for (auto offset = kWarpSize / 2; offset > 0; offset /= 2) {
    auto otherMax = __shfl_xor_sync(0xffffffffu, state.max, offset);
    auto otherSum = __shfl_xor_sync(0xffffffffu, state.sum, offset);
    state.merge(otherMax, otherSum);
  }
  if (threadIdx.x % kWarpSize == 0) {
    sMax[threadIdx.x / kWarpSize] = state.max;
    sSum[threadIdx.x / kWarpSize] = state.sum;
  }
  __syncthreads();

  // Folding the four warp partials redundantly in every thread is cheaper than broadcasting the
  // result of a single reduction.
  auto total = SoftmaxState();
  for (auto warp = 0; warp < kWarps; ++warp) {
    total.merge(sMax[warp], sSum[warp]);
  }

  // fmha_collective_softmax.hpp's tail() divides by a reciprocal rather than by the sum itself.
  const auto invSum = __frcp_rn(total.sum);

  for (auto tile = ptrdiff_t{0}; tile < fullTiles; ++tile) {
    const auto base = tile * kTile + col;
    *reinterpret_cast<float4 *>(out + base) =
        scale_vector(*reinterpret_cast<const float4 *>(in + base), total.max, invSum);
  }
  for (auto i = 0; i < kFloatsPerThread && tail + i < n; ++i) {
    out[tail + i] = exp2e(in[tail + i] - total.max) * invSum;
  }
}

} // namespace

// Row-wise softmax over an m x n tile of floats: out[i][j] = exp(in[i][j]) / sum_j exp(in[i][j]).
// Unlike softmax() this subtracts the row maximum first, which is mathematically the same softmax
// but stays finite for large inputs. The traffic is unchanged - the statistics pass and the scaling
// pass each read the row once, and the row is written once.
//
// ldIn and ldOut must be multiples of 4: the full tiles are read and written as float4, so every
// row has to start 16-byte aligned. softmax.cu's uint128_t copy atoms impose the same constraint.
void softmax_fmha(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  const dim3 dimBlock(static_cast<unsigned>(kThreads));
  const dim3 dimGrid(static_cast<unsigned>(m));

  softmax_fmha_device<<<dimGrid, dimBlock>>>(dIn, dOut, n, ldIn, ldOut);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
