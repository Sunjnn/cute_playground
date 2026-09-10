#include "cudnn_transpose.cuh"

#include <stdexcept>

using std::runtime_error;

#ifdef PLAYGROUND_NO_CUDNN

// cmake/Cudnn.cmake defines PLAYGROUND_NO_CUDNN when it cannot find libcudnn, so the example still
// builds and runs. The row stays in the benchmark table with the reason in its error column.
void transpose_cudnn(
    [[maybe_unused]] int m,
    [[maybe_unused]] int n,
    [[maybe_unused]] const float *dIn,
    [[maybe_unused]] int ldIn,
    [[maybe_unused]] float *dOut,
    [[maybe_unused]] int ldOut) {
  throw runtime_error("built without cuDNN - reconfigure with -DCUDNN_ROOT=<dir>");
}

#else

#include <cudnn.h>

namespace {

void check(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw runtime_error(cudnnGetErrorString(status));
  }
}

// cudnnCreate() costs milliseconds - orders of magnitude more than one transpose - so the handle
// and the two descriptors are built once for the whole process instead of per call. Refilling a
// descriptor is just a handful of host-side stores, so the problem size is written into them again
// on every call rather than cached and compared.
struct Context {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t in = nullptr;
  cudnnTensorDescriptor_t out = nullptr;

  Context() {
    check(cudnnCreate(&handle));
    check(cudnnCreateTensorDescriptor(&in));
    check(cudnnCreateTensorDescriptor(&out));
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;

  ~Context() {
    cudnnDestroyTensorDescriptor(in);
    cudnnDestroyTensorDescriptor(out);
    cudnnDestroy(handle);
  }
};

} // namespace

// Writes the transpose of an m x n row-major matrix of floats into an n x m one:
// dOut[j * ldOut + i] = dIn[i * ldIn + j].
//
// cudnnTransformTensor copies x into y converting between the layouts the two descriptors ask for,
// and a descriptor filled by cudnnSetTensor4dDescriptorEx is nothing but four strides. Both
// describe the same (1, m, n, 1) extent - N is the batch, C the row of dIn, H its column, W a
// singleton - but dIn advances ldIn per row and 1 per column while dOut advances 1 per row and
// ldOut per column, so the copy walks the two buffers perpendicular to each other.
//
// cuDNN has marked this routine deprecated since 9.0.0 in favour of the graph API, where the same
// permutation is one identity node with a strided output; it is still the only layout-transforming
// copy cuDNN exposes as a single call. The deprecation attribute compiles to nothing unless
// CUDNN_WARN_DEPRECATED is defined, so the build stays quiet about it.
void transpose_cudnn(int m, int n, const float *dIn, int ldIn, float *dOut, int ldOut) {
  static auto sContext = Context();

  // The strides are (nStride, cStride, hStride, wStride); N and W are singletons, so theirs are
  // placeholders.
  check(cudnnSetTensor4dDescriptorEx(sContext.in, CUDNN_DATA_FLOAT, 1, m, n, 1, 1, ldIn, 1, 1));
  check(cudnnSetTensor4dDescriptorEx(sContext.out, CUDNN_DATA_FLOAT, 1, m, n, 1, 1, 1, ldOut, 1));

  const auto alpha = 1.0f;
  const auto beta = 0.0f;
  // beta = 0 means cuDNN overwrites dOut rather than reading it first.
  check(cudnnTransformTensor(sContext.handle, &alpha, sContext.in, dIn, &beta, sContext.out, dOut));
}

#endif
