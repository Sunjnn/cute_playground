#include "cudnn_rmsnorm.cuh"

#include <stdexcept>

using std::runtime_error;

#ifdef PLAYGROUND_NO_CUDNN

// cmake/Cudnn.cmake defines PLAYGROUND_NO_CUDNN when it cannot find libcudnn, so the example still
// builds and runs. The row stays in the benchmark table with the reason in its error column.
void rmsnorm_cudnn(
    [[maybe_unused]] int m,
    [[maybe_unused]] int n,
    [[maybe_unused]] const float *dIn,
    [[maybe_unused]] int ldIn,
    [[maybe_unused]] const float *dWeight,
    [[maybe_unused]] float *dOut,
    [[maybe_unused]] int ldOut,
    [[maybe_unused]] float eps) {
  throw runtime_error("built without cuDNN - reconfigure with -DCUDNN_ROOT=<dir>");
}

#else

#include <algorithm>
#include <cccl/cub/device/device_segmented_reduce.cuh>
#include <cccl/thrust/execution_policy.h>
#include <cccl/thrust/fill.h>
#include <cccl/thrust/iterator/counting_iterator.h>
#include <cccl/thrust/iterator/transform_iterator.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>

using cub::DeviceSegmentedReduce;
using std::max;
using std::size_t;
using thrust::counting_iterator;
using thrust::device;
using thrust::fill_n;
using thrust::make_transform_iterator;

namespace {

void check(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw runtime_error(cudnnGetErrorString(status));
  }
}

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

// Same three functors as rmsnorm_cub, duplicated rather than shared: each implementation in this
// folder is one self-contained translation unit, and these are fifteen lines.
struct MeanSquareOp {
  float invN;

  __host__ __device__ float operator()(float x) const { return x * x * invN; }
};

struct RowBeginOp {
  int ld;

  __host__ __device__ int operator()(int row) const { return row * ld; }
};

struct RowEndOp {
  int ld;
  int n;

  __host__ __device__ int operator()(int row) const { return row * ld + n; }
};

// cudnnCreate() costs milliseconds - orders of magnitude more than one rmsnorm - so the handle and
// the six descriptors are built once for the whole process instead of per call. Refilling a
// descriptor is a handful of host-side stores, so describe() writes the problem size into them
// again on every call rather than caching and comparing it.
struct Context {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t in = nullptr;
  cudnnTensorDescriptor_t normalized = nullptr;
  cudnnTensorDescriptor_t stat = nullptr;
  cudnnTensorDescriptor_t weight = nullptr;
  cudnnTensorDescriptor_t out = nullptr;
  cudnnOpTensorDescriptor_t multiply = nullptr;

  Context() {
    check(cudnnCreate(&handle));
    check(cudnnCreateTensorDescriptor(&in));
    check(cudnnCreateTensorDescriptor(&normalized));
    check(cudnnCreateTensorDescriptor(&stat));
    check(cudnnCreateTensorDescriptor(&weight));
    check(cudnnCreateTensorDescriptor(&out));
    check(cudnnCreateOpTensorDescriptor(&multiply));
    check(cudnnSetOpTensorDescriptor(
        multiply, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;

  ~Context() {
    cudnnDestroyOpTensorDescriptor(multiply);
    cudnnDestroyTensorDescriptor(out);
    cudnnDestroyTensorDescriptor(weight);
    cudnnDestroyTensorDescriptor(stat);
    cudnnDestroyTensorDescriptor(normalized);
    cudnnDestroyTensorDescriptor(in);
    cudnnDestroy(handle);
  }
};

} // namespace

// Row-wise RMS normalization: out[i][j] = in[i][j] * rsqrt(mean_j(in[i][j]^2) + eps) * weight[j].
//
// cuDNN's only normalization primitive is batch normalization, and SPATIAL batch norm reduces over
// every mode except C. So the trick is to describe the m x n tile as N = 1, C = m, H = 1, W = n:
// C becomes the row and W the column, and the reduction over N, H and W is then a reduction over
// one row's columns - exactly mean_j. With estimatedMean forced to zero, estimatedVariance set to
// that mean square, scale 1 and bias 0, forward inference computes
// 1 * (x - 0) / sqrt(mean_j(x^2) + eps) + 0, which is RMSNorm without the weight.
//
// The weight cannot go through the same call: BN's scale and bias are per-C, which under this
// description means per-row, while weight[j] is per-column. So it is applied afterwards with
// cudnnOpTensor, whose B operand is described as (1, 1, 1, n) and broadcast over C.
//
// Two things cuDNN does not provide, and what this row therefore really measures:
// - The statistic itself. Legacy cuDNN has no row-wise reduction, so mean_j(x^2) comes from the
//   same CUB segmented reduce rmsnorm_cub uses. cuDNN only does the easy half.
// - A fused normalize-and-scale. BN writes to an m x n scratch buffer and OpTensor reads it back,
//   so this path walks m x n five times (reduce once, BN in and out, OpTensor in and out) against
//   the CuTe kernel's three and rmsnorm_fused's two.
void rmsnorm_cudnn(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  static auto sContext = Context();

  // The strides are (nStride, cStride, hStride, wStride); N and H are singletons, so theirs are
  // placeholders that cuDNN multiplies by a zero index.
  check(cudnnSetTensor4dDescriptorEx(sContext.in, CUDNN_DATA_FLOAT, 1, m, 1, n, 1, ldIn, 1, 1));
  check(
      cudnnSetTensor4dDescriptorEx(sContext.normalized, CUDNN_DATA_FLOAT, 1, m, 1, n, 1, n, 1, 1));
  check(cudnnSetTensor4dDescriptorEx(sContext.out, CUDNN_DATA_FLOAT, 1, m, 1, n, 1, ldOut, 1, 1));
  check(cudnnSetTensor4dDescriptor(sContext.stat, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, m, 1, 1));
  check(
      cudnnSetTensor4dDescriptor(sContext.weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n));

  // Per-row mean squares, via CUB - see the note above on why cuDNN cannot supply them.
  auto meanSquareIn = make_transform_iterator(dIn, MeanSquareOp{1.0f / static_cast<float>(n)});
  auto rows = counting_iterator<int>(0);
  auto inBegin = make_transform_iterator(rows, RowBeginOp{ldIn});
  auto inEnd = make_transform_iterator(rows, RowEndOp{ldIn, n});

  float *dMeanSqr = nullptr;
  auto tempBytes = size_t{0};
  check(DeviceSegmentedReduce::Sum(nullptr, tempBytes, meanSquareIn, dMeanSqr, m, inBegin, inEnd));
  tempBytes = max<size_t>(tempBytes, 1);

  float *dOnes = nullptr;
  float *dZeros = nullptr;
  float *dNormalized = nullptr;
  void *temp = nullptr;
  check(cudaMallocAsync(&dMeanSqr, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dOnes, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dZeros, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(
      &dNormalized, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float), nullptr));
  check(cudaMallocAsync(&temp, tempBytes, nullptr));

  check(DeviceSegmentedReduce::Sum(temp, tempBytes, meanSquareIn, dMeanSqr, m, inBegin, inEnd));

  fill_n(device, dOnes, m, 1.0f);
  check(cudaMemsetAsync(dZeros, 0, static_cast<size_t>(m) * sizeof(float), nullptr));

  const auto alpha = 1.0f;
  const auto beta = 0.0f;
  check(cudnnBatchNormalizationForwardInference(
      sContext.handle,
      CUDNN_BATCHNORM_SPATIAL,
      &alpha,
      &beta,
      sContext.in,
      dIn,
      sContext.normalized,
      dNormalized,
      sContext.stat,
      dOnes,
      dZeros,
      dZeros,
      dMeanSqr,
      static_cast<double>(eps)));

  // beta = 0 means the output is overwritten rather than accumulated into, so dOut is never read.
  check(cudnnOpTensor(
      sContext.handle,
      sContext.multiply,
      &alpha,
      sContext.normalized,
      dNormalized,
      &alpha,
      sContext.weight,
      dWeight,
      &beta,
      sContext.out,
      dOut));

  check(cudaFreeAsync(dNormalized, nullptr));
  check(cudaFreeAsync(dZeros, nullptr));
  check(cudaFreeAsync(dOnes, nullptr));
  check(cudaFreeAsync(dMeanSqr, nullptr));
  check(cudaFreeAsync(temp, nullptr));
}

#endif
