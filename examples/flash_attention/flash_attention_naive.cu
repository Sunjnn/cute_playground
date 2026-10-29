#include "flash_attention_naive.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "attention_common.cuh"

using std::int64_t;
using std::invalid_argument;
using std::runtime_error;

namespace {

constexpr int kThreads = 256;
constexpr int kWarpNum = kThreads / 32;
constexpr int kMaxSmemBytes = 48 * 1024;
constexpr int kReductionBytes = 2 * kWarpNum * static_cast<int>(sizeof(float));

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

// The correctness floor: one row per CTA, scalar SIMT QK and PV, and a stable softmax between
// them. The query and scores live in shared memory; probabilities never round to half or spill
// to global memory. This deliberately makes no tensor-core or overlapped-copy optimization.
__global__ void flash_attention_naive_device(
    const __half *__restrict__ dQ,
    int ldQ,
    const __half *__restrict__ dK,
    int ldK,
    const __half *__restrict__ dV,
    int ldV,
    __half *__restrict__ dO,
    int ldO,
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

  const auto row = static_cast<int64_t>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto *query = dQ + row * ldQ;
  auto *output = dO + row * ldO;

  for (auto t = threadId; t < k; t += kThreads) {
    sQuery[t] = __half2float(query[t]);
  }
  __syncthreads();

  // Bottom-right alignment: the last query sees the last key even when m < n. Subtract m before
  // adding row to avoid signed overflow at large dimensions; the host rejects causal m > n.
  const auto visible = causal ? n - m + 1 + static_cast<int>(row) : n;
  for (auto col = threadId; col < n; col += kThreads) {
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

  // Separate reduction arrays let the sum writers proceed while other warps still read sMax.
  __syncthreads();
  rowMax = sMax[0];
#pragma unroll
  for (auto i = 1; i < kWarpNum; ++i) {
    rowMax = fmaxf(rowMax, sMax[i]);
  }

  auto rowSum = 0.0f;
  for (auto col = threadId; col < n; col += kThreads) {
    const auto value = __expf(sScores[col] - rowMax);
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
  const auto invSum = 1.0f / rowSum;
  for (auto col = threadId; col < n; col += kThreads) {
    sScores[col] *= invSum;
  }
  // PV consumes other threads' probabilities, so normalization must finish across the whole CTA.
  __syncthreads();

  for (auto t = threadId; t < k; t += kThreads) {
    auto acc = 0.0f;
    for (auto col = 0; col < visible; ++col) {
      const auto *value = dV + static_cast<int64_t>(col) * ldV;
      acc += sScores[col] * __half2float(value[t]);
    }
    output[t] = __float2half_rn(acc);
  }
}

} // namespace

void flash_attention_naive(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    const __half *dV,
    int ldV,
    __half *dO,
    int ldO,
    float scale,
    bool causal) {
  if (m <= 0 || n <= 0 || k <= 0 || ldQ < k || ldK < k || ldV < k || ldO < k || dQ == nullptr ||
      dK == nullptr || dV == nullptr || dO == nullptr || !attention::valid_scale(k, scale)) {
    throw invalid_argument(
        "flash_attention_naive requires positive dimensions, valid strides, "
        "non-null buffers and an fp32-safe scale");
  }
  if (causal && m > n) {
    throw invalid_argument("flash_attention_naive requires m <= n for causal attention");
  }
  // Count both the dynamically staged rows and the static warp-reduction arrays. Widen before
  // addition so even an unsupported shape cannot overflow while being rejected.
  const auto smem = (static_cast<int64_t>(k) + n) * static_cast<int64_t>(sizeof(float));
  if (smem + kReductionBytes > kMaxSmemBytes) {
    throw invalid_argument(
        "flash_attention_naive requires k + n <= 12272 to fit its query, "
        "scores and reductions in 48 KiB of shared memory");
  }

  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m);
  flash_attention_naive_device<<<dimGrid, dimBlock, static_cast<unsigned int>(smem)>>>(
      dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, m, n, k, scale, causal);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(nullptr));
}
