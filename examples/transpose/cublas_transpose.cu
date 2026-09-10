#include "cublas_transpose.cuh"

#include <cublas_v2.h>
#include <stdexcept>

using std::runtime_error;

namespace {

void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw runtime_error(cublasGetStatusString(status));
  }
}

// cublasCreate() allocates a workspace and costs milliseconds - orders of magnitude more than one
// transpose - so the handle is built once for the whole process instead of per call.
struct Handle {
  cublasHandle_t value = nullptr;

  Handle() { check(cublasCreate(&value)); }

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&) = delete;
  Handle &operator=(Handle &&) = delete;

  ~Handle() { cublasDestroy(value); }
};

} // namespace

// Writes the transpose of an m x n row-major matrix of floats into an n x m one:
// dOut[j * ldOut + i] = dIn[i * ldIn + j].
//
// cuBLAS is column-major, so the m x n row-major dIn with leading dimension ldIn is the same memory
// as an n x m column-major A with lda = ldIn, and the n x m row-major dOut with leading dimension
// ldOut is an m x n column-major C with ldc = ldOut. The transpose then wants C(i, j) = A(j, i),
// which is C = A^T: cublasSgeam with alpha = 1 and beta = 0. geam is cuBLAS' out-of-place
// layout-transforming copy, so this is a transpose and not a matrix multiply in disguise.
void transpose_cublas(int m, int n, const float *dIn, int ldIn, float *dOut, int ldOut) {
  static auto sHandle = Handle();

  const auto alpha = 1.0f;
  const auto beta = 0.0f;
  // beta = 0 means cuBLAS never reads B, but the pointer still has to be valid, so dOut does duty.
  check(cublasSgeam(
      sHandle.value,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      m,
      n,
      &alpha,
      dIn,
      ldIn,
      &beta,
      dOut,
      ldOut,
      dOut,
      ldOut));
}
