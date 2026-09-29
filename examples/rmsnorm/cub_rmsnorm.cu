#include "cub_rmsnorm.cuh"

#include <algorithm>
#include <cccl/cub/device/device_segmented_reduce.cuh>
#include <cccl/thrust/execution_policy.h>
#include <cccl/thrust/for_each.h>
#include <cccl/thrust/iterator/counting_iterator.h>
#include <cccl/thrust/iterator/transform_iterator.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

using cub::DeviceSegmentedReduce;
using std::max;
using std::runtime_error;
using std::size_t;
using thrust::counting_iterator;
using thrust::device;
using thrust::for_each_n;
using thrust::make_transform_iterator;

namespace {

// The 1/n is folded into the load rather than divided out afterwards, so what the segmented reduce
// hands back is already the row's mean square and the scale needs no arithmetic of its own.
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

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

} // namespace

// Row-wise RMS normalization: out[i][j] = in[i][j] * rsqrt(mean_j(in[i][j]^2) + eps) * weight[j].
//
// The library composition, and the same shape as softmax_cub: one CUB segmented reduce for the row
// statistic, one thrust elementwise pass to apply it. Two kernels, so the row is read twice and the
// output written once - 3x the row's bytes, the same as the CuTe kernel, and the reduce and the
// scale cannot overlap because the second one needs the first one's answer.
void rmsnorm_cub(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  auto meanSquareIn = make_transform_iterator(dIn, MeanSquareOp{1.0f / static_cast<float>(n)});
  auto rows = counting_iterator<int>(0);
  auto inBegin = make_transform_iterator(rows, RowBeginOp{ldIn});
  auto inEnd = make_transform_iterator(rows, RowEndOp{ldIn, n});

  float *means = nullptr;
  auto tempBytes = size_t{0};
  check(DeviceSegmentedReduce::Sum(nullptr, tempBytes, meanSquareIn, means, m, inBegin, inEnd));
  // CUB reads a null d_temp_storage as "report the required size", so the real call needs a
  // non-null pointer even when no temp storage is required.
  tempBytes = max<size_t>(tempBytes, 1);

  // cudaMallocAsync rather than thrust::device_vector: the default thrust allocator goes through
  // cudaMalloc/cudaFree, which costs milliseconds per call and would dominate the measurement. The
  // harness sets the default pool's release threshold to UINT64_MAX so these pages are not handed
  // back to the driver between iterations.
  void *temp = nullptr;
  check(cudaMallocAsync(&means, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&temp, tempBytes, nullptr));

  check(DeviceSegmentedReduce::Sum(temp, tempBytes, meanSquareIn, means, m, inBegin, inEnd));

  auto total = m * n;
  if (ldIn == n && ldOut == n) {
    // Contiguous rows: the element index is already the in/out offset, so only the per-row mean and
    // per-column weight lookups pay for a division. The device has no integer divide instruction,
    // and the general form below costs roughly 5% at these sizes - enough to skew the comparison.
    for_each_n(device, rows, total, [dIn, dOut, dWeight, means, n, eps] __device__(int t) {
      auto row = t / n;
      dOut[t] = dIn[t] * rsqrtf(means[row] + eps) * dWeight[t - row * n];
    });
  } else {
    for_each_n(
        device, rows, total, [dIn, dOut, dWeight, means, n, ldIn, ldOut, eps] __device__(int t) {
          auto row = t / n;
          auto col = t - row * n;
          dOut[row * ldOut + col] = dIn[row * ldIn + col] * rsqrtf(means[row] + eps) * dWeight[col];
        });
  }

  check(cudaFreeAsync(means, nullptr));
  check(cudaFreeAsync(temp, nullptr));
}
