#include "cub_softmax.cuh"
#include "cudnn_softmax.cuh"
#include "fmha_softmax.cuh"
#include "softmax.cuh"

#include <array>
#include <cccl/thrust/copy.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

using cutlass::CommandLine;
using std::array;
using std::exception;
using std::exp;
using std::fabs;
using std::fprintf;
using std::isnan;
using std::mt19937;
using std::printf;
using std::runtime_error;
using std::snprintf;
using std::string;
using std::uniform_real_distribution;
using std::vector;
using thrust::copy;
using thrust::device_vector;
using thrust::host_vector;

namespace {

constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 8192;
constexpr int kDefaultIterations = 20;
constexpr int kWarmupIterations = 3;
constexpr double kMaxRelDiff = 1e-4;

using SoftmaxFn = void (*)(int, int, float *, int, float *, int);

struct Impl {
  const char *name;
  SoftmaxFn run;
};

constexpr size_t kImplCount = 4;

constexpr array<Impl, kImplCount> kImpls{
    {Impl{"softmax", softmax},
     Impl{"softmax_cub", softmax_cub},
     Impl{"softmax_fmha", softmax_fmha},
     Impl{"softmax_cudnn", softmax_cudnn}}};

struct Result {
  bool checked = false;
  bool verified = false;
  bool timed = false;
  double maxRelDiff = 0.0;
  double usPerIter = 0.0;
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
  auto engine = mt19937(20260903);
  auto dist = uniform_real_distribution<float>(-1.0f, 1.0f);
  auto values = host_vector<float>(count);
  for (auto &value : values) {
    value = dist(engine);
  }
  return values;
}

// Largest relative deviation from a row-wise reference accumulated in double. A NaN deviation
// sticks, so an implementation that leaves part of the output buffer untouched cannot be rescued
// by the elements it did write.
double
max_rel_diff(const Problem &prob, const host_vector<float> &hIn, const host_vector<float> &hOut) {
  auto maxRel = 0.0;
  auto rowExp = vector<double>(static_cast<size_t>(prob.n));
  for (auto row = 0; row < prob.m; ++row) {
    auto base = static_cast<size_t>(row) * static_cast<size_t>(prob.n);
    auto sum = 0.0;
    for (auto col = 0; col < prob.n; ++col) {
      rowExp[col] = exp(static_cast<double>(hIn[base + col]));
      sum += rowExp[col];
    }
    for (auto col = 0; col < prob.n; ++col) {
      auto expected = rowExp[col] / sum;
      auto rel = fabs(static_cast<double>(hOut[base + col]) - expected) / expected;
      if (isnan(rel) || rel > maxRel) {
        maxRel = rel;
      }
    }
  }
  return maxRel;
}

double benchmark(const Impl &impl, const Problem &prob, float *dIn, float *dOut) {
  for (auto i = 0; i < kWarmupIterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dOut, prob.n);
    check(cudaStreamSynchronize(nullptr));
  }

  auto timer = GPU_Clock();
  timer.start();
  for (auto i = 0; i < prob.iterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dOut, prob.n);
    // softmax() synchronizes internally, the other three do not. Synchronizing here keeps them on
    // the same footing instead of letting the asynchronous paths overlap successive iterations.
    check(cudaStreamSynchronize(nullptr));
  }
  return static_cast<double>(timer.milliseconds()) * 1000.0 / prob.iterations;
}

Result run_impl(
    const Impl &impl,
    const Problem &prob,
    const host_vector<float> &hIn,
    host_vector<float> &hOut,
    device_vector<float> &dIn,
    device_vector<float> &dOut) {
  auto result = Result();
  try {
    // 0xFF bytes spell NaN, so an implementation that never reaches its store cannot pass by
    // inheriting the previous implementation's output from the shared buffer.
    check(cudaMemset(dOut.data().get(), 0xFF, dOut.size() * sizeof(float)));
    check(cudaStreamSynchronize(nullptr));

    impl.run(prob.m, prob.n, dIn.data().get(), prob.n, dOut.data().get(), prob.n);
    check(cudaStreamSynchronize(nullptr));

    copy(dOut.begin(), dOut.end(), hOut.begin());
    result.maxRelDiff = max_rel_diff(prob, hIn, hOut);
    result.checked = true;
    result.verified = result.maxRelDiff <= kMaxRelDiff;

    result.usPerIter = benchmark(impl, prob, dIn.data().get(), dOut.data().get());
    result.timed = true;
  } catch (const exception &e) {
    result.error = e.what();
  }
  return result;
}

array<Result, kImplCount> run_all(
    const Problem &prob,
    const host_vector<float> &hIn,
    host_vector<float> &hOut,
    device_vector<float> &dIn,
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
  // The GB/s column prices every implementation at two reads of the input and one write of the
  // output, which is what the three kernels in this repository do and the least a two-pass softmax
  // can cost.
  auto trafficBytes = 3.0 * static_cast<double>(prob.m) * static_cast<double>(prob.n) *
                      static_cast<double>(sizeof(float));
  auto softmaxUs = results[0].timed ? results[0].usPerIter : 0.0;

  printf("softmax of %d x %d float, %d timed iterations\n", prob.m, prob.n, prob.iterations);
  printf(
      "%-16s %-8s %13s %10s %8s %10s\n",
      "implementation",
      "verify",
      "max_rel_diff",
      "time_us",
      "GB/s",
      "vs_softmax");
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    const auto &result = results[i];
    auto gbps = trafficBytes / (result.usPerIter * 1e-6) / 1e9;
    auto speedup = softmaxUs / result.usPerIter;
    const char *verifyText = "-";
    if (result.checked) {
      verifyText = result.verified ? "PASS" : "FAIL";
    }
    printf(
        "%-16s %-8s %13s %10s %8s %10s\n",
        kImpls[i].name,
        verifyText,
        number(result.checked, "%.3e", result.maxRelDiff).c_str(),
        number(result.timed, "%.1f", result.usPerIter).c_str(),
        number(result.timed, "%.1f", gbps).c_str(),
        number(result.timed && softmaxUs > 0.0, "%.2fx", speedup).c_str());
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
  // The implementations index with int, and CUB reduces an empty row to 0 and divides by it.
  if (static_cast<int64_t>(prob.m) * prob.n > INT32_MAX) {
    fprintf(stderr, "m * n must fit in int32\n");
    return EXIT_FAILURE;
  }

  try {
    // cudaFreeAsync hands pages back to the driver once the pool's release threshold (0 by
    // default) is exceeded, which would put a real allocation inside softmax_cub's timed region.
    cudaMemPool_t pool = nullptr;
    check(cudaDeviceGetDefaultMemPool(&pool, 0));
    auto threshold = uint64_t{UINT64_MAX};
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

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
