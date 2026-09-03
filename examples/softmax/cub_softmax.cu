#include "cub_softmax.cuh"

#include <algorithm>
#include <cccl/cub/device/device_segmented_reduce.cuh>
#include <cccl/thrust/execution_policy.h>
#include <cccl/thrust/for_each.h>
#include <cccl/thrust/iterator/counting_iterator.h>
#include <cccl/thrust/iterator/transform_iterator.h>
#include <cuda_runtime.h>
#include <stdexcept>

using cub::DeviceSegmentedReduce;
using std::max;
using std::runtime_error;
using thrust::counting_iterator;
using thrust::device;
using thrust::for_each_n;
using thrust::make_transform_iterator;

namespace {

struct ExpOp {
  __host__ __device__ float operator()(float x) const { return expf(x); }
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

// Row-wise softmax over an m x n tile of floats: out[i][j] = exp(in[i][j]) / sum_j exp(in[i][j]).
// The row max is deliberately not subtracted, because softmax() in softmax.cu does not subtract it
// either and the benchmark has to compare the same computation. Two passes over global memory, so
// the traffic is 3 * m * n * sizeof(float) - the same as the CuTe kernel's.
void softmax_cub(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  auto expIn = make_transform_iterator(dIn, ExpOp{});
  auto rows = counting_iterator<int>(0);
  auto inBegin = make_transform_iterator(rows, RowBeginOp{ldIn});
  auto inEnd = make_transform_iterator(rows, RowEndOp{ldIn, n});

  float *sums = nullptr;
  auto tempBytes = size_t{0};
  check(DeviceSegmentedReduce::Sum(nullptr, tempBytes, expIn, sums, m, inBegin, inEnd));
  // CUB reads a null d_temp_storage as "report the required size", so the real call needs a
  // non-null pointer even when no temp storage is required.
  tempBytes = max<size_t>(tempBytes, 1);

  // cudaMallocAsync rather than thrust::device_vector: the default thrust allocator goes through
  // cudaMalloc/cudaFree, which costs milliseconds per call and would dominate the measurement.
  void *temp = nullptr;
  check(cudaMallocAsync(&sums, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&temp, tempBytes, nullptr));

  check(DeviceSegmentedReduce::Sum(temp, tempBytes, expIn, sums, m, inBegin, inEnd));

  auto total = m * n;
  if (ldIn == n && ldOut == n) {
    // Contiguous rows: the element index is already the in/out offset, so only the per-row sum
    // lookup pays for a division. The device has no integer divide instruction, and the general
    // form below costs roughly 5% at m = n = 8192 - enough to skew the comparison.
    for_each_n(device, rows, total, [dIn, dOut, sums, n] __device__(int t) {
      dOut[t] = expf(dIn[t]) / sums[t / n];
    });
  } else {
    for_each_n(device, rows, total, [dIn, dOut, sums, n, ldIn, ldOut] __device__(int t) {
      auto row = t / n;
      auto col = t - row * n;
      dOut[row * ldOut + col] = expf(dIn[row * ldIn + col]) / sums[row];
    });
  }

  check(cudaFreeAsync(sums, nullptr));
  check(cudaFreeAsync(temp, nullptr));
}
