#include "cutensor_rmsnorm.cuh"

#include <stdexcept>

using std::runtime_error;

#ifdef PLAYGROUND_NO_CUTENSOR

// cmake/Cutensor.cmake defines PLAYGROUND_NO_CUTENSOR when it cannot find libcutensor - unlike
// cuBLAS and cuDNN it does not ship with the CUDA toolkit - so the example still builds and runs.
// The row stays in the benchmark table with the reason in its error column.
void rmsnorm_cutensor(
    [[maybe_unused]] int m,
    [[maybe_unused]] int n,
    [[maybe_unused]] const float *dIn,
    [[maybe_unused]] int ldIn,
    [[maybe_unused]] const float *dWeight,
    [[maybe_unused]] float *dOut,
    [[maybe_unused]] int ldOut,
    [[maybe_unused]] float eps) {
  throw runtime_error("built without cuTENSOR - reconfigure with -DCUTENSOR_ROOT=<dir>");
}

#else

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
// cuda_runtime.h must come before the cccl headers: on MSVC they reach for _BitScanForward64,
// which intrin.h (pulled in by cuda_runtime.h) declares.
#include <cccl/thrust/execution_policy.h>
#include <cccl/thrust/fill.h>
#include <cutensor.h>

using std::array;
using std::int32_t;
using std::int64_t;
using std::max;
using std::uint32_t;
using std::uint64_t;
using thrust::device;
using thrust::fill_n;

// The signatures below were checked against cuTENSOR 2.7's cutensor.h. Three things are easy to
// get wrong about this API and worth knowing before editing:
//
// - There is no unary elementwise operation. A unary operator (SQRT, RCP, ...) can only be applied
//   by cutensorCreatePermutation/cutensorPermute - a permutation whose modes happen not to move -
//   or as the per-operand opA/opB of an elementwise operation.
// - The elementwise binary cannot broadcast: omitting a mode from descC or giving it a zero stride
//   both return CUTENSOR_STATUS_NOT_SUPPORTED. The trinary elementwise broadcasts by mode omission
//   instead, so a broadcast multiply is spelled MUL(A, B) + 0 * C.
// - Only cutensorReduce takes a workspace at execution time; cutensorPermute and the elementwise
//   executes take none, so those plans are created with their (zero) workspace estimate as the
//   limit, as in examples/transpose.
namespace {

void check(cutensorStatus_t status) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw runtime_error(cutensorGetErrorString(status));
  }
}

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

// The alignment promised for the base pointers. cudaMalloc - where the harness's device_vectors
// live - returns 256-byte aligned memory, but the scratch buffers here come from cudaMallocAsync,
// which only promises enough alignment for any built-in type, so the descriptors claim 16 bytes
// rather than the 128 examples/transpose could claim. cuTENSOR uses the value to decide how wide
// its accesses may be.
constexpr uint32_t kAlignment = 16;

// Five steps. cuTENSOR's elementwise binary cannot broadcast - neither by omitting a mode from
// descC nor by a zero stride, both return CUTENSOR_STATUS_NOT_SUPPORTED - and its operator set has
// no square (SQR appears in the docs but not in the enum), so the weight and scale steps use the
// trinary elementwise, whose docs do allow broadcasting by mode omission:
//
//   kSquare  dSqr[i, j]    = dIn[i, j] * dIn[i, j]          (binary, A and C both dIn, opAC = MUL)
//   kReduce  dMeanSqr[i]   = sum_j(dSqr[i, j]) / n + eps    (reduction, alpha = 1/n, beta = 1)
//   kSqrt    dStd[i]       = sqrt(dMeanSqr[i])              (permutation, CUTENSOR_OP_SQRT)
//   kWeight  dXW[i, j]     = dIn[i, j] * dWeight[j]         (trinary MUL-ADD, gamma = 0)
//   kScale   dOut[i, j]    = dXW[i, j] * rcp(dStd[i])       (trinary MUL-ADD, opB = RCP, gamma = 0)
//
// A single cutensorCreateContraction of the form C[i] = sum_j A[i, j] * B[i, j] would collapse
// kSquare and kReduce into one step and drop the dSqr scratch entirely, but the contraction
// descriptor's argument list is the one this file has the least evidence for, so it stays out.
constexpr int kStepNum = 5;
constexpr int kSquare = 0;
constexpr int kReduce = 1;
constexpr int kSqrt = 2;
constexpr int kWeight = 3;
constexpr int kScale = 4;

// cuTENSOR plans run its heuristics and cost far more than one rmsnorm, so the whole set is cached
// and only rebuilt when a call asks for a different shape - as in examples/transpose.
struct Plan {
  cutensorHandle_t handle = nullptr;

  // (m, n):(ldIn, 1), (m, n):(n, 1), (m):(1), (n):(1), (m, n):(ldOut, 1).
  cutensorTensorDescriptor_t in = nullptr;
  cutensorTensorDescriptor_t packed = nullptr;
  cutensorTensorDescriptor_t row = nullptr;
  cutensorTensorDescriptor_t col = nullptr;
  cutensorTensorDescriptor_t out = nullptr;

  array<cutensorOperationDescriptor_t, kStepNum> ops{};
  array<cutensorPlanPreference_t, kStepNum> preferences{};
  array<cutensorPlan_t, kStepNum> plans{};
  array<uint64_t, kStepNum> workspaceBytes{};

  // The shape the cached plans were built for; -1 means nothing is built yet.
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

  void destroy(cutensorTensorDescriptor_t &descriptor) {
    if (descriptor != nullptr) {
      cutensorDestroyTensorDescriptor(descriptor);
      descriptor = nullptr;
    }
  }

  void release() {
    for (auto i = 0; i < kStepNum; ++i) {
      if (plans[i] != nullptr) {
        cutensorDestroyPlan(plans[i]);
        plans[i] = nullptr;
      }
      if (preferences[i] != nullptr) {
        cutensorDestroyPlanPreference(preferences[i]);
        preferences[i] = nullptr;
      }
      if (ops[i] != nullptr) {
        cutensorDestroyOperationDescriptor(ops[i]);
        ops[i] = nullptr;
      }
      workspaceBytes[i] = 0;
    }
    destroy(in);
    destroy(packed);
    destroy(row);
    destroy(col);
    destroy(out);
    m = -1;
    n = -1;
    ldIn = -1;
    ldOut = -1;
  }

  void makeTensor(
      cutensorTensorDescriptor_t *descriptor,
      const array<int64_t, 2> &extent,
      const array<int64_t, 2> &stride) {
    check(cutensorCreateTensorDescriptor(
        handle, descriptor, 2, extent.data(), stride.data(), CUDA_R_32F, kAlignment));
  }

  void makeTensor(
      cutensorTensorDescriptor_t *descriptor,
      const array<int64_t, 1> &extent,
      const array<int64_t, 1> &stride) {
    check(cutensorCreateTensorDescriptor(
        handle, descriptor, 1, extent.data(), stride.data(), CUDA_R_32F, kAlignment));
  }

  // JIT_MODE_NONE keeps plan creation to cuTENSOR's precompiled kernels; DEFAULT would let it
  // compile a dedicated one, which costs seconds the first time. The workspace size is estimated
  // rather than asked for after the fact: only cutensorReduce receives the buffer at execution
  // time, but every plan is created with its estimate as the limit.
  void makePlan(int step) {
    check(cutensorCreatePlanPreference(
        handle, &preferences[step], CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    check(cutensorEstimateWorkspaceSize(
        handle, ops[step], preferences[step], CUTENSOR_WORKSPACE_DEFAULT, &workspaceBytes[step]));
    check(cutensorCreatePlan(
        handle, &plans[step], ops[step], preferences[step], workspaceBytes[step]));
  }

  void build(int rows, int cols, int leadIn, int leadOut) {
    release();

    // The mode labels are what tell cuTENSOR which extents are summed over and which are broadcast:
    // a mode missing from an operand is broadcast over it, and a mode present in both inputs of a
    // reduction but absent from its output is contracted away.
    const auto modeRow = array<int32_t, 2>{'i', 'j'};
    const auto modeCol = array<int32_t, 1>{'j'};
    const auto modeRows = array<int32_t, 1>{'i'};

    makeTensor(&in, array<int64_t, 2>{rows, cols}, array<int64_t, 2>{leadIn, 1});
    makeTensor(&packed, array<int64_t, 2>{rows, cols}, array<int64_t, 2>{cols, 1});
    makeTensor(&out, array<int64_t, 2>{rows, cols}, array<int64_t, 2>{leadOut, 1});
    makeTensor(&row, array<int64_t, 1>{rows}, array<int64_t, 1>{1});
    makeTensor(&col, array<int64_t, 1>{cols}, array<int64_t, 1>{1});

    check(cutensorCreateElementwiseBinary(
        handle,
        &ops[kSquare],
        in,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        in,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        packed,
        modeRow.data(),
        CUTENSOR_OP_MUL,
        CUTENSOR_COMPUTE_DESC_32F));
    check(cutensorCreateReduction(
        handle,
        &ops[kReduce],
        packed,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        row,
        modeRows.data(),
        CUTENSOR_OP_IDENTITY,
        row,
        modeRows.data(),
        CUTENSOR_OP_ADD,
        CUTENSOR_COMPUTE_DESC_32F));
    check(cutensorCreatePermutation(
        handle,
        &ops[kSqrt],
        row,
        modeRows.data(),
        CUTENSOR_OP_SQRT,
        row,
        modeRows.data(),
        CUTENSOR_COMPUTE_DESC_32F));
    // The trinary D = opABC(opAB(A, B), C) with opAB = MUL, opABC = ADD and gamma = 0 is the only
    // elementwise shape cuTENSOR accepts here: the binary one cannot broadcast dWeight's {j} over
    // the rows, while the trinary one broadcasts a mode missing from B by construction. C is a
    // full-size tensor that gamma = 0 keeps from being read.
    check(cutensorCreateElementwiseTrinary(
        handle,
        &ops[kWeight],
        in,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        col,
        modeCol.data(),
        CUTENSOR_OP_IDENTITY,
        in,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        packed,
        modeRow.data(),
        CUTENSOR_OP_MUL,
        CUTENSOR_OP_ADD,
        CUTENSOR_COMPUTE_DESC_32F));
    check(cutensorCreateElementwiseTrinary(
        handle,
        &ops[kScale],
        packed,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        row,
        modeRows.data(),
        CUTENSOR_OP_RCP,
        packed,
        modeRow.data(),
        CUTENSOR_OP_IDENTITY,
        out,
        modeRow.data(),
        CUTENSOR_OP_MUL,
        CUTENSOR_OP_ADD,
        CUTENSOR_COMPUTE_DESC_32F));

    for (auto step = 0; step < kStepNum; ++step) {
      makePlan(step);
    }

    m = rows;
    n = cols;
    ldIn = leadIn;
    ldOut = leadOut;
  }

  uint64_t maxWorkspaceBytes() const {
    auto largest = uint64_t{0};
    for (const auto bytes : workspaceBytes) {
      largest = max(largest, bytes);
    }
    return largest;
  }
};

} // namespace

// Row-wise RMS normalization: out[i][j] = in[i][j] * rsqrt(mean_j(in[i][j]^2) + eps) * weight[j].
//
// The composition cuTENSOR forces, and the reason this row is here: it is five kernels over four
// scratch buffers, walking m x n seven times - square reads dIn and writes the scratch, reduce
// reads the scratch, weight reads dIn and overwrites the scratch, scale reads the scratch and
// writes dOut - against the CuTe kernel's three and rmsnorm_fused's two. It cannot do better
// because the operator set has no square, a unary operator costs a permutation pass of its own
// (sqrt), and only rcp rides along, as the opB of the scale's trinary. The eps is folded into the
// reduction as beta * C with C a row vector of eps, which is the only place in the pipeline that a
// scalar can enter.
void rmsnorm_cutensor(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  static auto sPlan = Plan();

  if (sPlan.m != m || sPlan.n != n || sPlan.ldIn != ldIn || sPlan.ldOut != ldOut) {
    sPlan.build(m, n, ldIn, ldOut);
  }

  // One m x n scratch serves both kSquare's output and kWeight's: the first is consumed by kReduce
  // before the second is produced, so the two lifetimes do not overlap.
  float *dPacked = nullptr;
  float *dMeanSqr = nullptr;
  float *dStd = nullptr;
  float *dEps = nullptr;
  void *workspace = nullptr;
  check(cudaMallocAsync(
      &dPacked, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dMeanSqr, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dStd, static_cast<size_t>(m) * sizeof(float), nullptr));
  check(cudaMallocAsync(&dEps, static_cast<size_t>(m) * sizeof(float), nullptr));
  const auto workspaceBytes = sPlan.maxWorkspaceBytes();
  // cutensorReduce wants its workspace 256-byte aligned, and cudaMallocAsync only promises 16;
  // over-allocate and round the pointer up. Every elementwise execute below takes no workspace.
  check(cudaMallocAsync(&workspace, max<uint64_t>(workspaceBytes, 1) + 255, nullptr));
  auto aligned =
      reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t{255});

  // Every row's epsilon is the same value; the reduction reads it as its beta * C operand.
  fill_n(device, dEps, m, eps);

  const auto alpha = 1.0f;
  const auto invN = 1.0f / static_cast<float>(n);
  const auto beta = 1.0f;
  const auto stream = cudaStream_t{nullptr};

  const auto gamma = 0.0f;
  check(cutensorElementwiseBinaryExecute(
      sPlan.handle, sPlan.plans[kSquare], &alpha, dIn, &alpha, dIn, dPacked, stream));
  check(cutensorReduce(
      sPlan.handle,
      sPlan.plans[kReduce],
      &invN,
      dPacked,
      &beta,
      dEps,
      dMeanSqr,
      aligned,
      workspaceBytes,
      stream));
  check(cutensorPermute(sPlan.handle, sPlan.plans[kSqrt], &alpha, dMeanSqr, dStd, stream));
  check(cutensorElementwiseTrinaryExecute(
      sPlan.handle,
      sPlan.plans[kWeight],
      &alpha,
      dIn,
      &alpha,
      dWeight,
      &gamma,
      dIn,
      dPacked,
      stream));
  check(cutensorElementwiseTrinaryExecute(
      sPlan.handle,
      sPlan.plans[kScale],
      &alpha,
      dPacked,
      &alpha,
      dStd,
      &gamma,
      dPacked,
      dOut,
      stream));

  check(cudaFreeAsync(workspace, nullptr));
  check(cudaFreeAsync(dEps, nullptr));
  check(cudaFreeAsync(dStd, nullptr));
  check(cudaFreeAsync(dMeanSqr, nullptr));
  check(cudaFreeAsync(dPacked, nullptr));
}

#endif
