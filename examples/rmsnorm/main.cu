#include <array>
#include <cccl/thrust/copy.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
#include <cmath>
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

#include "cub_rmsnorm.cuh"
#include "cublas_rmsnorm.cuh"
#include "cudnn_rmsnorm.cuh"
#include "cutensor_rmsnorm.cuh"
#include "rmsnorm.cuh"
#include "rmsnorm_fused.cuh"
#include "rmsnorm_naive.cuh"

using cutlass::CommandLine;
using std::array;
using std::exception;
using std::fabs;
using std::fmax;
using std::fprintf;
using std::isnan;
using std::mt19937;
using std::printf;
using std::runtime_error;
using std::size_t;
using std::snprintf;
using std::sqrt;
using std::string;
using std::uniform_real_distribution;
using thrust::copy;
using thrust::device_vector;
using thrust::host_vector;

namespace {

// m rows of n hidden elements each: 8192 tokens of 4096 is a language-model shaped problem, wide
// enough that a row spans several tiles and tall enough to fill the GPU.
constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 4096;
constexpr int kDefaultIterations = 20;
constexpr int kWarmupIterations = 3;
constexpr double kMaxRelDiff = 1e-4;

// A typical production value: LLaMA-style layers use 1e-5 or 1e-6. Passed to each implementation
// rather than baked into them, so the reference and the kernels cannot drift apart on it.
constexpr float kEpsilon = 1e-5f;

using RmsnormFn = void (*)(int, int, const float *, int, const float *, float *, int, float);

struct Impl {
  const char *name;
  RmsnormFn run;
};

constexpr size_t kImplCount = 7;

// The naive kernel leads because it is the reference the speedup column is measured against; then
// the four library compositions, then this repo's two kernels, the one under development last.
constexpr array<Impl, kImplCount> kImpls{
    {Impl{"rmsnorm_naive", rmsnorm_naive},
     Impl{"rmsnorm_cub", rmsnorm_cub},
     Impl{"rmsnorm_cublas", rmsnorm_cublas},
     Impl{"rmsnorm_cudnn", rmsnorm_cudnn},
     Impl{"rmsnorm_cutensor", rmsnorm_cutensor},
     Impl{"rmsnorm", rmsnorm},
     Impl{"rmsnorm_fused", rmsnorm_fused}}};

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
  auto engine = mt19937(20260929);
  auto dist = uniform_real_distribution<float>(-1.0f, 1.0f);
  auto values = host_vector<float>(count);
  for (auto &value : values) {
    value = dist(engine);
  }
  return values;
}

// Strictly positive and away from zero, so that a column whose weight vanished cannot make the
// reference zero everywhere and hide a wrong scale behind a zero times anything.
host_vector<float> make_weight(size_t count) {
  auto engine = mt19937(20260930);
  auto dist = uniform_real_distribution<float>(0.5f, 1.5f);
  auto values = host_vector<float>(count);
  for (auto &value : values) {
    value = dist(engine);
  }
  return values;
}

// Largest deviation from a row-wise reference accumulated in double. A NaN deviation sticks, so
// an implementation that leaves part of the output buffer untouched cannot be rescued by the
// elements it did write.
//
// The denominator is max(|expected|, 1) rather than |expected| because an RMSNorm output is zero
// wherever its input is, and a relative error against a zero reference is either meaningless or
// infinite. Elements of magnitude at least one are judged relatively, the rest absolutely - which
// is what the softmax harness's relative test amounts to here anyway, since every output of this
// problem is within a small factor of one.
double max_rel_diff(
    const Problem &prob,
    const host_vector<float> &hIn,
    const host_vector<float> &hWeight,
    const host_vector<float> &hOut) {
  auto maxRel = 0.0;
  for (auto row = 0; row < prob.m; ++row) {
    auto base = static_cast<size_t>(row) * static_cast<size_t>(prob.n);
    auto sqrSum = 0.0;
    for (auto col = 0; col < prob.n; ++col) {
      auto value = static_cast<double>(hIn[base + col]);
      sqrSum += value * value;
    }
    auto scale = 1.0 / sqrt(sqrSum / static_cast<double>(prob.n) + static_cast<double>(kEpsilon));
    for (auto col = 0; col < prob.n; ++col) {
      auto expected =
          static_cast<double>(hIn[base + col]) * scale * static_cast<double>(hWeight[col]);
      auto rel = fabs(static_cast<double>(hOut[base + col]) - expected) / fmax(fabs(expected), 1.0);
      if (isnan(rel) || rel > maxRel) {
        maxRel = rel;
      }
    }
  }
  return maxRel;
}

double benchmark(
    const Impl &impl, const Problem &prob, const float *dIn, const float *dWeight, float *dOut) {
  for (auto i = 0; i < kWarmupIterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dWeight, dOut, prob.n, kEpsilon);
    check(cudaStreamSynchronize(nullptr));
  }

  auto timer = GPU_Clock();
  timer.start();
  for (auto i = 0; i < prob.iterations; ++i) {
    impl.run(prob.m, prob.n, dIn, prob.n, dWeight, dOut, prob.n, kEpsilon);
    // Both implementations synchronize internally; synchronizing here too keeps an implementation
    // that returns before its work is done from overlapping successive iterations and looking
    // faster than it is.
    check(cudaStreamSynchronize(nullptr));
  }
  return static_cast<double>(timer.milliseconds()) * 1000.0 / prob.iterations;
}

Result run_impl(
    const Impl &impl,
    const Problem &prob,
    const host_vector<float> &hIn,
    const host_vector<float> &hWeight,
    host_vector<float> &hOut,
    const device_vector<float> &dIn,
    const device_vector<float> &dWeight,
    device_vector<float> &dOut) {
  auto result = Result();
  try {
    // 0xFF bytes spell NaN. No implementation reads dOut, so unlike the transpose harness there is
    // no alpha * NaN to poison a baseline, and a NaN left behind is exactly the signal wanted: an
    // element an implementation never reached cannot pass by inheriting the previous one's output
    // from the shared buffer.
    check(cudaMemset(dOut.data().get(), 0xFF, dOut.size() * sizeof(float)));
    check(cudaStreamSynchronize(nullptr));

    impl.run(
        prob.m,
        prob.n,
        dIn.data().get(),
        prob.n,
        dWeight.data().get(),
        dOut.data().get(),
        prob.n,
        kEpsilon);
    check(cudaStreamSynchronize(nullptr));

    copy(dOut.begin(), dOut.end(), hOut.begin());
    result.maxRelDiff = max_rel_diff(prob, hIn, hWeight, hOut);
    result.checked = true;
    result.verified = result.maxRelDiff <= kMaxRelDiff;

    // Timing an implementation that produced the wrong output would only measure how fast it is
    // at being wrong.
    if (result.verified) {
      result.usPerIter =
          benchmark(impl, prob, dIn.data().get(), dWeight.data().get(), dOut.data().get());
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
    const host_vector<float> &hWeight,
    host_vector<float> &hOut,
    const device_vector<float> &dIn,
    const device_vector<float> &dWeight,
    device_vector<float> &dOut) {
  auto results = array<Result, kImplCount>{};
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    results[i] = run_impl(kImpls[i], prob, hIn, hWeight, hOut, dIn, dWeight, dOut);
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
  // output, which is what a two-pass RMSNorm does and the least one that reloads can cost. The
  // weight vector is left out: it is the same n floats for all m rows, so after the first CTAs it
  // is served from L2 and counting it as DRAM traffic would overstate everyone by a factor that
  // depends on m rather than on the kernel. A single-pass kernel that caches the row in registers
  // moves 2/3 of this, so read its GB/s against a ceiling 1.5x higher.
  auto trafficBytes = 3.0 * static_cast<double>(prob.m) * static_cast<double>(prob.n) *
                      static_cast<double>(sizeof(float));
  auto naiveUs = results[0].timed ? results[0].usPerIter : 0.0;

  printf(
      "rmsnorm of %d x %d float, eps %g, %d timed iterations\n",
      prob.m,
      prob.n,
      static_cast<double>(kEpsilon),
      prob.iterations);
  printf(
      "%-20s %-8s %13s %10s %8s %10s\n",
      "implementation",
      "verify",
      "max_rel_diff",
      "time_us",
      "GB/s",
      "vs_naive");
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    const auto &result = results[i];
    auto gbps = trafficBytes / (result.usPerIter * 1e-6) / 1e9;
    auto speedup = naiveUs / result.usPerIter;
    const char *verifyText = "-";
    if (result.checked) {
      verifyText = result.verified ? "PASS" : "FAIL";
    }
    printf(
        "%-20s %-8s %13s %10s %8s %10s\n",
        kImpls[i].name,
        verifyText,
        number(result.checked, "%.3e", result.maxRelDiff).c_str(),
        number(result.timed, "%.1f", result.usPerIter).c_str(),
        number(result.timed, "%.1f", gbps).c_str(),
        number(result.timed && naiveUs > 0.0, "%.2fx", speedup).c_str());
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
  // The implementations index with int, and the last element of a row sits at (m - 1) * n + n - 1.
  if (static_cast<int64_t>(prob.m) * prob.n > INT32_MAX) {
    fprintf(stderr, "m * n must fit in int32\n");
    return EXIT_FAILURE;
  }

  try {
    // cudaFreeAsync hands pages back to the driver once the pool's release threshold (0 by default)
    // is exceeded, which would put a real allocation inside the timed region of every baseline that
    // allocates scratch per call - cub, cublas, cudnn and cutensor all do.
    cudaMemPool_t pool = nullptr;
    check(cudaDeviceGetDefaultMemPool(&pool, 0));
    auto threshold = uint64_t{UINT64_MAX};
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    auto count = static_cast<size_t>(prob.m) * static_cast<size_t>(prob.n);
    auto hIn = make_input(count);
    auto hWeight = make_weight(static_cast<size_t>(prob.n));
    auto hOut = host_vector<float>(count);
    auto dIn = device_vector<float>(hIn);
    auto dWeight = device_vector<float>(hWeight);
    auto dOut = device_vector<float>(count);

    print_table(prob, run_all(prob, hIn, hWeight, hOut, dIn, dWeight, dOut));
  } catch (const exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
