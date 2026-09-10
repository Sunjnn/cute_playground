#include "cutensor_transpose.cuh"

#include <stdexcept>

using std::runtime_error;

#ifdef PLAYGROUND_NO_CUTENSOR

// cmake/Cutensor.cmake defines PLAYGROUND_NO_CUTENSOR when it cannot find libcutensor - unlike
// cuBLAS and cuDNN it does not ship with the CUDA toolkit - so the example still builds and runs.
// The row stays in the benchmark table with the reason in its error column.
void transpose_cutensor(
    [[maybe_unused]] int m,
    [[maybe_unused]] int n,
    [[maybe_unused]] const float *dIn,
    [[maybe_unused]] int ldIn,
    [[maybe_unused]] float *dOut,
    [[maybe_unused]] int ldOut) {
  throw runtime_error("built without cuTENSOR - reconfigure with -DCUTENSOR_ROOT=<dir>");
}

#else

#include <array>
#include <cstdint>
#include <cutensor.h>

using std::array;
using std::int32_t;
using std::int64_t;
using std::uint32_t;

namespace {

void check(cutensorStatus_t status) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw runtime_error(cutensorGetErrorString(status));
  }
}

// The alignment, in bytes, promised for the base pointers. cudaMalloc returns 256-byte aligned
// memory, which is where the harness's device_vectors live, and cuTENSOR uses the value to decide
// how wide its accesses can be.
constexpr uint32_t kAlignment = 128;

// A cuTENSOR transpose is two tensor descriptors, the permutation their mode labels imply, and a
// plan compiled from them. Building the plan runs cuTENSOR's heuristics and costs far more than one
// transpose, so it is cached here and only rebuilt when a call asks for a different shape.
struct Plan {
  cutensorHandle_t handle = nullptr;
  cutensorTensorDescriptor_t in = nullptr;
  cutensorTensorDescriptor_t out = nullptr;
  cutensorOperationDescriptor_t op = nullptr;
  cutensorPlanPreference_t preference = nullptr;
  cutensorPlan_t value = nullptr;
  // The shape the cached plan was built for; -1 means nothing is built yet.
  int m = -1;
  int n = -1;
  int ldIn = -1;
  int ldOut = -1;

  Plan() { check(cutensorCreate(&handle)); }

  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
  Plan(Plan &&) = delete;
  Plan &operator=(Plan &&) = delete;

  ~Plan() {
    release();
    cutensorDestroy(handle);
  }

  void release() {
    if (value != nullptr) {
      cutensorDestroyPlan(value);
      value = nullptr;
    }
    if (preference != nullptr) {
      cutensorDestroyPlanPreference(preference);
      preference = nullptr;
    }
    if (op != nullptr) {
      cutensorDestroyOperationDescriptor(op);
      op = nullptr;
    }
    if (out != nullptr) {
      cutensorDestroyTensorDescriptor(out);
      out = nullptr;
    }
    if (in != nullptr) {
      cutensorDestroyTensorDescriptor(in);
      in = nullptr;
    }
    m = -1;
    n = -1;
    ldIn = -1;
    ldOut = -1;
  }

  void build(int rows, int cols, int leadIn, int leadOut) {
    release();

    // dIn is rank 2 with mode 0 the row (stride ldIn) and mode 1 the column (stride 1). dOut holds
    // the same two modes in the other order: its mode 0 is dIn's column, which advances ldOut, and
    // its mode 1 is dIn's row, which advances 1. The labels are what tell cuTENSOR that the two
    // descriptions differ by a permutation rather than by a shape.
    const auto extentIn = array<int64_t, 2>{rows, cols};
    const auto strideIn = array<int64_t, 2>{leadIn, 1};
    const auto modeIn = array<int32_t, 2>{'i', 'j'};
    const auto extentOut = array<int64_t, 2>{cols, rows};
    const auto strideOut = array<int64_t, 2>{leadOut, 1};
    const auto modeOut = array<int32_t, 2>{'j', 'i'};

    check(cutensorCreateTensorDescriptor(
        handle, &in, 2, extentIn.data(), strideIn.data(), CUDA_R_32F, kAlignment));
    check(cutensorCreateTensorDescriptor(
        handle, &out, 2, extentOut.data(), strideOut.data(), CUDA_R_32F, kAlignment));
    check(cutensorCreatePermutation(
        handle,
        &op,
        in,
        modeIn.data(),
        CUTENSOR_OP_IDENTITY,
        out,
        modeOut.data(),
        CUTENSOR_COMPUTE_DESC_32F));
    // JIT_MODE_NONE keeps plan creation to cuTENSOR's precompiled kernels; DEFAULT would let it
    // compile a dedicated one, which costs seconds the first time.
    check(cutensorCreatePlanPreference(
        handle, &preference, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    // cutensorPermute takes no workspace pointer, so the plan is capped at zero bytes of one.
    check(cutensorCreatePlan(handle, &value, op, preference, 0));

    m = rows;
    n = cols;
    ldIn = leadIn;
    ldOut = leadOut;
  }
};

} // namespace

// Writes the transpose of an m x n row-major matrix of floats into an n x m one:
// dOut[j * ldOut + i] = dIn[i * ldIn + j]. cuTENSOR expresses it as an out-of-place permutation
// B[j, i] = A[i, j] with alpha = 1 and the identity operator, which for equal element types and no
// scaling is a pure copy.
void transpose_cutensor(int m, int n, const float *dIn, int ldIn, float *dOut, int ldOut) {
  static auto sPlan = Plan();

  if (sPlan.m != m || sPlan.n != n || sPlan.ldIn != ldIn || sPlan.ldOut != ldOut) {
    sPlan.build(m, n, ldIn, ldOut);
  }

  const auto alpha = 1.0f;
  check(cutensorPermute(sPlan.handle, sPlan.value, &alpha, dIn, dOut, nullptr));
}

#endif
