#include "cublas_unfused_attention.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

#include "attention_common.cuh"

using std::int64_t;
using std::invalid_argument;
using std::numeric_limits;
using std::runtime_error;
using std::size_t;

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

// Handle creation allocates cuBLAS's workspace, so cache it rather than timing creation per call.
// A new handle defaults to the default stream and host pointer mode, which both GEMMs use here.
struct Handle {
  // NOLINTNEXTLINE(readability-identifier-naming)
  cublasHandle_t value_ = nullptr;

  Handle() {
    check(cublasCreate(&value_));
    try {
      // Keep tensor cores eligible, but forbid half-precision split-K reductions when C is half.
      check(cublasSetMathMode(value_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    } catch (...) {
      (void)cublasDestroy(value_);
      throw;
    }
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&) = delete;
  Handle &operator=(Handle &&) = delete;
  ~Handle() { (void)cublasDestroy(value_); }
};

// Construct the two allocations independently: if the second allocation, either GEMM or a launch
// fails, destructors enqueue frees for everything already owned, on the same stream as its uses.
// The successful path releases explicitly so errors from cudaFreeAsync are checked too.
struct Scratch {
  // NOLINTNEXTLINE(readability-identifier-naming)
  void *data_ = nullptr;

  explicit Scratch(size_t bytes) { check(cudaMallocAsync(&data_, bytes, nullptr)); }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  Scratch(Scratch &&) = delete;
  Scratch &operator=(Scratch &&) = delete;
  ~Scratch() {
    if (data_ != nullptr) {
      (void)cudaFreeAsync(data_, nullptr);
    }
  }

  void release() {
    check(cudaFreeAsync(data_, nullptr));
    data_ = nullptr;
  }
};

constexpr int kThreads = 256;
constexpr int kWarpNum = kThreads / 32;

// One row per CTA, with scalar strided walks rather than a bounded register cache or float4
// loads. There are no alignment/divisibility restrictions and no maximum sequence length. The
// extra global-memory walks keep this a simple general baseline, not another fused implementation.
__global__ void mask_softmax_device(
    const float *__restrict__ dScores, __half *__restrict__ dP, int m, int n, bool causal) {
  const auto row = static_cast<int64_t>(blockIdx.x);
  const auto threadId = static_cast<int>(threadIdx.x);
  const auto *scores = dScores + row * n;
  auto *probs = dP + row * n;
  const auto visible = causal ? n - m + 1 + static_cast<int>(row) : n;

  __shared__ float sMax[kWarpNum];
  __shared__ float sSum[kWarpNum];

  // Widen the loop counter as well as row offsets: a final col += kThreads must not wrap at
  // INT_MAX.
  auto rowMax = -INFINITY;
  for (auto col = static_cast<int64_t>(threadId); col < visible; col += kThreads) {
    rowMax = fmaxf(rowMax, scores[col]);
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

  auto rowSum = 0.0f;
  for (auto col = static_cast<int64_t>(threadId); col < visible; col += kThreads) {
    rowSum += __expf(scores[col] - rowMax);
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
  for (auto col = static_cast<int64_t>(threadId); col < n; col += kThreads) {
    // Quantize only after normalization, not the scores or unnormalized exponentials. Explicit
    // zeros for the causal suffix also make every element of the compact P allocation initialized.
    const auto prob = col < visible ? __expf(scores[col] - rowMax) * invSum : 0.0f;
    probs[col] = __float2half_rn(prob);
  }
}

} // namespace

void cublas_unfused_attention(
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
        "cublas_unfused_attention requires positive dimensions, valid strides, "
        "non-null buffers and an fp32-safe scale");
  }
  if (causal && m > n) {
    throw invalid_argument("cublas_unfused_attention requires m <= n for causal attention");
  }
  if (static_cast<size_t>(m) >
      numeric_limits<size_t>::max() / static_cast<size_t>(n) / sizeof(float)) {
    throw invalid_argument("cublas_unfused_attention score allocation size overflows size_t");
  }

  static auto sHandle = Handle();
  const auto count = static_cast<size_t>(m) * static_cast<size_t>(n);
  auto scoreScratch = Scratch(count * sizeof(float));
  auto probScratch = Scratch(count * sizeof(__half));
  auto *dScores = static_cast<float *>(scoreScratch.data_);
  auto *dP = static_cast<__half *>(probScratch.data_);

  // Row-major S = scale * Q K^T is column-major S^T = scale * K Q^T. The physical K[n, k]
  // buffer is column-major [k, n], so transpose K, but not Q. ldc = n leaves compact scores.
  const auto alpha = scale;
  const auto beta = 0.0f;
  check(cublasGemmEx(
      sHandle.value_,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      dK,
      CUDA_R_16F,
      ldK,
      dQ,
      CUDA_R_16F,
      ldQ,
      &beta,
      dScores,
      CUDA_R_32F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));

  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m);
  mask_softmax_device<<<dimGrid, dimBlock>>>(dScores, dP, m, n, causal);
  check(cudaGetLastError());

  // O = P V becomes O^T = V^T P^T. Row-major V[n, k] is already column-major V^T[k, n],
  // and compact row-major P[m, n] is already column-major P^T[n, m], so neither operand takes
  // a transpose. The result is column-major [k, m] with ldc = ldO, hence row-major O[m, k].
  // P and V are half, accumulation is float, and only the final output rounds back to half.
  const auto one = 1.0f;
  check(cublasGemmEx(
      sHandle.value_,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      k,
      m,
      n,
      &one,
      dV,
      CUDA_R_16F,
      ldV,
      dP,
      CUDA_R_16F,
      n,
      &beta,
      dO,
      CUDA_R_16F,
      ldO,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));

  scoreScratch.release();
  probScratch.release();
  // GEMM status and cudaGetLastError cover enqueue/launch errors; synchronization also surfaces
  // asynchronous execution faults rather than letting a later benchmark iteration discover them.
  check(cudaStreamSynchronize(nullptr));
}
