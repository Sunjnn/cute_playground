#include "cublas_unfused_softmax.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

using std::int64_t;
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

// cublasCreate() allocates a workspace and costs milliseconds - orders of magnitude more than one
// fused softmax - so the handle is built once for the whole process instead of per call.
struct Handle {
  cublasHandle_t value = nullptr;

  Handle() { check(cublasCreate(&value)); }

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&) = delete;
  Handle &operator=(Handle &&) = delete;

  ~Handle() { cublasDestroy(value); }
};

// Same register budget as softmax_regcache, whose structure this walks: 8 x float4 = 32 floats per
// thread, which at 1024 threads caps a row at 32768 columns.
constexpr int kVecPerThread = 8;

constexpr int kMinThreads = 128;
constexpr int kMaxThreads = 1024;

// Below kMinN some threads would hold no element at all; above kMaxN the row stops fitting.
constexpr int kMinN = 4 * kMinThreads;
constexpr int kMaxN = 4 * kVecPerThread * kMaxThreads;

// Applies the causal mask to four adjacent columns at once. Column j is visible iff
// j <= row + n - m, which keeps the last query aligned with the last key when m < n. The boundary
// falls wherever it likes, so this has to test per lane: deciding per float4 would either let a
// masked column survive into the row maximum or clip a visible one.
__device__ float4 mask_causal(float4 value, int col, int visible) {
  value.x = col + 0 < visible ? value.x : -INFINITY;
  value.y = col + 1 < visible ? value.y : -INFINITY;
  value.z = col + 2 < visible ? value.z : -INFINITY;
  value.w = col + 3 < visible ? value.w : -INFINITY;
  return value;
}

// The mask and the softmax, in one kernel over the scores cuBLAS left in global memory. One row per
// CTA with the whole row cached in registers, so the row is read exactly once and written once -
// the reductions are warp shuffles plus one float per warp in shared memory, and there is no smem
// tile and no copy. The scale is not here: cublasGemmEx's alpha applies it inside the GEMM, which
// is the one piece of this epilogue cuBLAS can absorb.
//
// Padded lanes past the row's end carry -INFINITY, which fmaxf() absorbs and __expf() turns into
// exactly 0, so neither reduction needs to know which lanes are padding.
template <int kThreads>
__global__ void mask_softmax_device(
    const float *__restrict__ dScores,
    __half *__restrict__ dP,
    int ldP,
    int m,
    int n,
    bool causal) {
  constexpr int kWarpNum = kThreads / 32;

  // int64_t, not long: long is 32-bit under MSVC, which is one of this project's targets, so a row
  // offset would narrow there and not on Linux.
  const auto row = static_cast<int64_t>(blockIdx.x);
  // Cast once here rather than at each use: threadIdx.x is unsigned, and leaving it that way would
  // promote every index to unsigned and silently convert the int vecNum in the bounds tests below.
  const auto threadId = static_cast<int>(threadIdx.x);
  // dScores is written compactly with leading dimension n whatever Q's and K's strides were, so the
  // GEMM's padding never reaches this kernel.
  const auto *src = reinterpret_cast<const float4 *>(dScores + row * n);
  auto *dst = reinterpret_cast<__half2 *>(dP + row * ldP);
  const auto vecNum = n / 4;

  auto visible = n;
  if (causal) {
    // main.cu rejects m > n, so a row here always sees at least one column and the row sum below is
    // never a division by zero.
    visible = min(max(static_cast<int>(row) + n - m + 1, 0), n);
  }

  float4 values[kVecPerThread];
#pragma unroll
  for (auto i = 0; i < kVecPerThread; ++i) {
    const auto vecIndex = threadId + i * kThreads;
    auto value = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    if (vecIndex < vecNum) {
      value = src[vecIndex];
      if (causal) {
        value = mask_causal(value, vecIndex * 4, visible);
      }
    }
    values[i] = value;
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

  // One reciprocal per row rather than one division per element.
  const auto invSum = 1.0f / rowSum;
#pragma unroll
  for (auto i = 0; i < kVecPerThread; ++i) {
    const auto vecIndex = threadId + i * kThreads;
    if (vecIndex < vecNum) {
      // Two __half2 rather than four __half: P is half the width of the scores this kernel reads,
      // so four columns are 8 bytes and one store instruction. Widened explicitly, because 2 *
      // vecIndex is a pointer offset and an int one would narrow on a large enough row.
      const auto half2Index = static_cast<int64_t>(vecIndex) * 2;
      dst[half2Index] = __floats2half2_rn(values[i].x * invSum, values[i].y * invSum);
      dst[half2Index + 1] = __floats2half2_rn(values[i].z * invSum, values[i].w * invSum);
    }
  }
}

} // namespace

// The unfused shape of this problem, and the baseline the fused kernels are measured against:
// cuBLAS writes the whole m x n score matrix to global memory, and a second kernel reads all of it
// back to mask and normalize it. Two launches, and 4*M*N bytes written plus 4*M*N read plus 2*M*N
// written - five times the 2*M*N that writing P alone costs, and m*n*4 bytes of scratch no fused
// kernel needs.
//
// What cuBLAS does buy is the scale, for free: alpha multiplies the accumulator, so passing it
// there costs nothing and takes one step out of the epilogue. It cannot buy the mask or the
// softmax, because both are properties of a whole row and cuBLAS assigns each output tile to an
// independent CTA that never sees the rest of its row. That is the whole gap this example exists to
// close.
//
// The scores are left in fp32 rather than fp16. Halving them would halve this baseline's traffic
// and make the comparison look better for it, but it would also put a second rounding in front of
// the exponent - at the default shape that pushes the output's deviation past the harness's 1e-3
// threshold for reasons that have nothing to do with being unfused.
void cublas_unfused_softmax(
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
  static auto sHandle = Handle();

  // Reading a row of scores as float4 needs n to be whole vectors, and storing it as __half2 needs
  // ldP even so every row starts 4-byte aligned.
  if (n % 4 != 0 || ldP % 2 != 0) {
    throw runtime_error("cublas_unfused_softmax needs n to be a multiple of 4 and ldP even");
  }
  if (n < kMinN || n > kMaxN) {
    throw runtime_error("cublas_unfused_softmax holds one row in registers, so 512 <= n <= 32768");
  }

  // cudaMallocAsync and the harness's UINT64_MAX release threshold, as in rmsnorm_cublas: the
  // default allocator would cudaMalloc/cudaFree per call and measure the allocator.
  float *dScores = nullptr;
  check(cudaMallocAsync(
      &dScores, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float), nullptr));

  // S = Q K^T is m x n row-major, which in cuBLAS's column-major view is the n x m matrix
  // S^T = K Q^T. K's row-major (n, k) storage is already a column-major (k, n) matrix, so it takes
  // CUBLAS_OP_T to present it as the n x k operand; Q's is column-major (k, m) and needs no
  // transpose. ldc = n makes the column-major (n, m) result the row-major (m, n) scores.
  const auto alpha = scale;
  const auto beta = 0.0f;
  check(cublasGemmEx(
      sHandle.value,
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
    mask_softmax_device<128><<<dimGrid, dimBlock>>>(dScores, dP, ldP, m, n, causal);
    break;
  case 256:
    mask_softmax_device<256><<<dimGrid, dimBlock>>>(dScores, dP, ldP, m, n, causal);
    break;
  case 512:
    mask_softmax_device<512><<<dimGrid, dimBlock>>>(dScores, dP, ldP, m, n, causal);
    break;
  default:
    mask_softmax_device<1024><<<dimGrid, dimBlock>>>(dScores, dP, ldP, m, n, causal);
    break;
  }
  auto error = cudaGetLastError();
  // Freed either way: a failed launch still owns the allocation, and the pool's release threshold
  // is what keeps the next call from paying for a real cudaMalloc.
  const auto freeError = cudaFreeAsync(dScores, nullptr);
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
  check(freeError);
}
