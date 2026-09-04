#include "cudnn_softmax.cuh"

#include <stdexcept>

using std::runtime_error;

#ifdef PLAYGROUND_NO_CUDNN

// cmake/Cudnn.cmake defines PLAYGROUND_NO_CUDNN when it cannot find libcudnn, so the example still
// builds and runs. The row stays in the benchmark table with the reason in its error column.
void softmax_cudnn(
    [[maybe_unused]] int m,
    [[maybe_unused]] int n,
    [[maybe_unused]] float *dIn,
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

// cudnnCreate() costs milliseconds - orders of magnitude more than one softmax - so the handle and
// the two descriptors are built once for the whole process instead of per call. Refilling a
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

// Row-wise softmax through cuDNN's legacy operator. Rows are the batch dimension and columns the
// channel dimension, with H and W degenerate to 1, so CUDNN_SOFTMAX_MODE_INSTANCE - which reduces
// over C, H and W for every N - is exactly one softmax per row. The strides are passed explicitly
// because ldIn and ldOut need not equal n.
//
// CUDNN_SOFTMAX_ACCURATE subtracts the row maximum before exponentiating, which puts this on the
// same numerical footing as softmax_fmha() rather than softmax().
void softmax_cudnn(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  static auto sContext = Context();

  check(cudnnSetTensor4dDescriptorEx(sContext.in, CUDNN_DATA_FLOAT, m, n, 1, 1, ldIn, 1, 1, 1));
  check(cudnnSetTensor4dDescriptorEx(sContext.out, CUDNN_DATA_FLOAT, m, n, 1, 1, ldOut, 1, 1, 1));

  const auto alpha = 1.0f;
  const auto beta = 0.0f;
  check(cudnnSoftmaxForward(
      sContext.handle,
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE,
      &alpha,
      sContext.in,
      dIn,
      &beta,
      sContext.out,
      dOut));
}

#endif
