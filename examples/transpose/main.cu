#include <array>
#include <cccl/thrust/copy.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>

#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

#include "cublas_transpose.cuh"
#include "cudnn_transpose.cuh"
#include "cutensor_transpose.cuh"
#include "transpose.cuh"

using cutlass::CommandLine;
using std::array;
using std::exception;
using std::fprintf;
using std::mt19937;
using std::printf;
using std::runtime_error;
using std::size_t;
using std::snprintf;
using std::string;
using std::to_string;
using std::uniform_real_distribution;
using thrust::copy;
using thrust::device_vector;
using thrust::host_vector;

namespace {

// Non-square by default: a kernel that mixes up m and n still passes on a square matrix.
constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 4096;
constexpr int kDefaultIterations = 20;
constexpr int kWarmupIterations = 3;

using TransposeFn = void (*)(int, int, const float *, int, float *, int);

struct Impl {
  const char *name;
  TransposeFn run;
};

constexpr size_t kImplCount = 4;

// The three library transposes come first and the CuTe kernel last. cuBLAS leads because it is the
// one that is always linked in, which makes it the reference the speedup column is measured
// against; cuTENSOR and cuDNN are optional dependencies, and the CuTe kernel is the one under
// development.
constexpr array<Impl, kImplCount> kImpls{
    {Impl{"transpose_cublas", transpose_cublas},
     Impl{"transpose_cutensor", transpose_cutensor},
     Impl{"transpose_cudnn", transpose_cudnn},
     Impl{"transpose", transpose}}};

struct Result {
  bool checked = false;
  bool verified = false;
  bool timed = false;
  size_t mismatches = 0;
  double usPerIter = 0.0;
  string detail;
  string error;
};

struct Problem {
  int m = 0;
  int n = 0;
  int iterations = 0;
};

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

host_vector<float> make_input(size_t count) {
  auto engine = mt19937(20260907);
  auto dist = uniform_real_distribution<float>(-1.0f, 1.0f);
  auto values = host_vector<float>(count);
  for (auto &value : values) {
    value = dist(engine);
  }
  return values;
}

// Bit-exact on purpose: a transpose only moves floats, so any difference is a wrong address rather
// than rounding. hIn is m x n row-major and hOut is its n x m transpose, both without padding.
// run_impl pre-fills dOut with a value outside the input range, so an element an implementation
// never reached is counted here instead of passing on a stale value.
size_t count_mismatches(
    const Problem &prob,
    const host_vector<float> &hIn,
    const host_vector<float> &hOut,
    string &detail) {
  auto mismatches = size_t{0};
  for (auto row = 0; row < prob.m; ++row) {
    for (auto col = 0; col < prob.n; ++col) {
      auto inIndex =
          static_cast<size_t>(row) * static_cast<size_t>(prob.n) + static_cast<size_t>(col);
      auto outIndex =
          static_cast<size_t>(col) * static_cast<size_t>(prob.m) + static_cast<size_t>(row);
      if (hOut[outIndex] == hIn[inIndex]) {
        continue;
      }
      ++mismatches;
      if (detail.empty()) {
        auto buffer = array<char, 128>{};
        snprintf(
            buffer.data(),
            buffer.size(),
            "first mismatch at in(%d, %d): out[%zu] = %g, want %g",
            row,
            col,
            outIndex,
            static_cast<double>(hOut[outIndex]),
            static_cast<double>(hIn[inIndex]));
        detail = {buffer.data()};
      }
    }
  }
  return mismatches;
}

double benchmark(const Impl &impl, const Problem &prob, const float *dIn, float *dOut) {
  for (auto i = 0; i < kWarmupIterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dOut, prob.m);
    check(cudaStreamSynchronize(nullptr));
  }

  auto timer = GPU_Clock();
  timer.start();
  for (auto i = 0; i < prob.iterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dOut, prob.m);
    // Synchronizing per iteration keeps an implementation that returns before its work is done
    // from overlapping successive iterations and looking faster than it is.
    check(cudaStreamSynchronize(nullptr));
  }
  return static_cast<double>(timer.milliseconds()) * 1000.0 / prob.iterations;
}

Result run_impl(
    const Impl &impl,
    const Problem &prob,
    const host_vector<float> &hIn,
    host_vector<float> &hOut,
    const device_vector<float> &dIn,
    device_vector<float> &dOut) {
  auto result = Result();
  try {
    // 0x7F bytes are a finite float far outside the range make_input() draws from, so an
    // implementation that never reaches its store cannot pass by inheriting the previous one's
    // output from the shared buffer. The softmax harness uses 0xFF here instead; its NaN would be
    // read back by any library that evaluates alpha * x + beta * y literally, and 0 * NaN is NaN.
    check(cudaMemset(dOut.data().get(), 0x7F, dOut.size() * sizeof(float)));
    check(cudaStreamSynchronize(nullptr));

    impl.run(prob.m, prob.n, dIn.data().get(), prob.n, dOut.data().get(), prob.m);
    check(cudaStreamSynchronize(nullptr));

    copy(dOut.begin(), dOut.end(), hOut.begin());
    result.mismatches = count_mismatches(prob, hIn, hOut, result.detail);
    result.checked = true;
    result.verified = result.mismatches == 0;

    // Timing an implementation that produced the wrong output would only measure how fast it is at
    // being wrong - a kernel that stores nothing tops the table on launch overhead alone.
    if (result.verified) {
      result.usPerIter = benchmark(impl, prob, dIn.data().get(), dOut.data().get());
      result.timed = true;
    }
  } catch (const exception &e) {
    result.error = e.what();
  }
  return result;
}

array<Result, kImplCount> run_all(
    const Problem &prob,
    const host_vector<float> &hIn,
    host_vector<float> &hOut,
    const device_vector<float> &dIn,
    device_vector<float> &dOut) {
  auto results = array<Result, kImplCount>{};
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    results[i] = run_impl(kImpls[i], prob, hIn, hOut, dIn, dOut);
  }
  return results;
}

string number(bool valid, const char *format, double value) {
  if (!valid) {
    return "n/a";
  }
  auto buffer = array<char, 32>{};
  snprintf(buffer.data(), buffer.size(), format, value);
  return {buffer.data()};
}

void print_table(const Problem &prob, const array<Result, kImplCount> &results) {
  // Every element is read once and written once, so this is the least any transpose can move and
  // the whole table is priced the same way.
  auto trafficBytes = 2.0 * static_cast<double>(prob.m) * static_cast<double>(prob.n) *
                      static_cast<double>(sizeof(float));
  auto cublasUs = results[0].timed ? results[0].usPerIter : 0.0;

  printf(
      "transpose of %d x %d float into %d x %d, %d timed iterations\n",
      prob.m,
      prob.n,
      prob.n,
      prob.m,
      prob.iterations);
  printf(
      "%-20s %-8s %12s %10s %8s %10s\n",
      "implementation",
      "verify",
      "mismatches",
      "time_us",
      "GB/s",
      "vs_cublas");
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    const auto &result = results[i];
    auto gbps = trafficBytes / (result.usPerIter * 1e-6) / 1e9;
    auto speedup = cublasUs / result.usPerIter;
    const char *verifyText = "-";
    if (result.checked) {
      verifyText = result.verified ? "PASS" : "FAIL";
    }
    auto mismatchText = result.checked ? to_string(result.mismatches) : string("n/a");
    printf(
        "%-20s %-8s %12s %10s %8s %10s\n",
        kImpls[i].name,
        verifyText,
        mismatchText.c_str(),
        number(result.timed, "%.1f", result.usPerIter).c_str(),
        number(result.timed, "%.1f", gbps).c_str(),
        number(result.timed && cublasUs > 0.0, "%.2fx", speedup).c_str());
    if (!result.detail.empty()) {
      printf("    %s\n", result.detail.c_str());
    }
    if (!result.error.empty()) {
      printf("    %s\n", result.error.c_str());
    }
  }
}

} // namespace

int main(int argc, char const **argv) {
  auto cmd = CommandLine(argc, argv);
  auto prob = Problem();
  cmd.get_cmd_line_argument("m", prob.m, kDefaultM);
  cmd.get_cmd_line_argument("n", prob.n, kDefaultN);
  cmd.get_cmd_line_argument("iterations", prob.iterations, kDefaultIterations);

  if (prob.m <= 0 || prob.n <= 0 || prob.iterations <= 0) {
    fprintf(stderr, "usage: %s [--m=N] [--n=N] [--iterations=N], all positive\n", argv[0]);
    return EXIT_FAILURE;
  }
  // The implementations index with int, and the last element of the output sits at n * m - 1.
  if (static_cast<int64_t>(prob.m) * prob.n > INT32_MAX) {
    fprintf(stderr, "m * n must fit in int32\n");
    return EXIT_FAILURE;
  }

  try {
    auto count = static_cast<size_t>(prob.m) * static_cast<size_t>(prob.n);
    auto hIn = make_input(count);
    auto hOut = host_vector<float>(count);
    auto dIn = device_vector<float>(hIn);
    auto dOut = device_vector<float>(count);

    print_table(prob, run_all(prob, hIn, hOut, dIn, dOut));
  } catch (const exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
