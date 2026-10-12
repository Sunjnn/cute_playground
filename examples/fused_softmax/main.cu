#include <array>
#include <cccl/thrust/copy.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>

#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

#include "cublas_unfused_softmax.cuh"
#include "fused_softmax_naive.cuh"
#include "fused_softmax_recompute.cuh"
#include "fused_softmax_regrow.cuh"
#include "reference.cuh"

using cutlass::CommandLine;
using std::array;
using std::exception;
using std::fprintf;
using std::int64_t;
using std::mt19937;
using std::printf;
using std::runtime_error;
using std::snprintf;
using std::sqrt;
using std::string;
using std::uniform_real_distribution;
using thrust::copy;
using thrust::device_vector;
using thrust::host_vector;

namespace {

constexpr int kDefaultM = 2048;
constexpr int kDefaultN = 2048;
constexpr int kDefaultK = 128;
constexpr int kDefaultIterations = 20;
constexpr int kWarmupIterations = 3;

// P is stored as fp16, whose relative resolution is 2^-11 = 4.9e-4, so the output cannot be held to
// the 1e-4 examples/softmax uses - that example stores fp32. 1e-3 leaves room for one rounding of P
// plus the noise in an fp32 accumulation whose order differs from the reference's, without being so
// loose that a wrong row reduction could slip through.
constexpr double kMaxRelDiff = 1e-3;

constexpr char kDefaultMask[] = "causal";

using FusedSoftmaxFn =
    void (*)(int, int, int, const __half *, int, const __half *, int, __half *, int, float, bool);

struct Impl {
  const char *name;
  FusedSoftmaxFn run;
};

constexpr size_t kImplCount = 4;

constexpr array<Impl, kImplCount> kImpls{
    {Impl{"fused_softmax_naive", fused_softmax_naive},
     Impl{"cublas_unfused_softmax", cublas_unfused_softmax},
     Impl{"fused_softmax_regrow", fused_softmax_regrow},
     Impl{"fused_softmax_recompute", fused_softmax_recompute}}};

// The speedup column is priced against the unfused cuBLAS baseline rather than against kImpls[0]:
// fused_softmax_naive is a correctness floor, deliberately without tensor cores or any reuse, so
// beating it says nothing. cublas_unfused_softmax is what this looks like when you do not fuse.
constexpr size_t kBaselineIndex = 1;

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
  int k = 0;
  int iterations = 0;
  float scale = 0.0f;
  bool causal = true;
  bool verify = true;
};

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

host_vector<__half> make_input(size_t count, mt19937 &engine) {
  auto dist = uniform_real_distribution<float>(-1.0f, 1.0f);
  auto values = host_vector<__half>(count);
  for (auto &value : values) {
    value = __float2half(dist(engine));
  }
  return values;
}

double
benchmark(const Impl &impl, const Problem &prob, const __half *dQ, const __half *dK, __half *dP) {
  for (auto i = 0; i < kWarmupIterations; ++i) {
    impl.run(prob.m, prob.n, prob.k, dQ, prob.k, dK, prob.k, dP, prob.n, prob.scale, prob.causal);
    check(cudaStreamSynchronize(nullptr));
  }

  auto timer = GPU_Clock();
  timer.start();
  for (auto i = 0; i < prob.iterations; ++i) {
    impl.run(prob.m, prob.n, prob.k, dQ, prob.k, dK, prob.k, dP, prob.n, prob.scale, prob.causal);
    // fused_softmax_naive() synchronizes internally. Synchronizing here too keeps implementations
    // that do not on the same footing, instead of letting an async path overlap the next iteration
    // and be timed as free.
    check(cudaStreamSynchronize(nullptr));
  }
  return static_cast<double>(timer.milliseconds()) * 1000.0 / prob.iterations;
}

Result run_impl(
    const Impl &impl,
    const Problem &prob,
    const Reference &ref,
    host_vector<__half> &hP,
    const device_vector<__half> &dQ,
    const device_vector<__half> &dK,
    device_vector<__half> &dP) {
  auto result = Result();
  try {
    if (prob.verify) {
      // 0xFFFF spells NaN in fp16, so an implementation that never reaches its store cannot pass by
      // inheriting the previous implementation's output from the shared buffer.
      check(cudaMemset(dP.data().get(), 0xFF, dP.size() * sizeof(__half)));
      check(cudaStreamSynchronize(nullptr));
    }

    impl.run(
        prob.m,
        prob.n,
        prob.k,
        dQ.data().get(),
        prob.k,
        dK.data().get(),
        prob.k,
        dP.data().get(),
        prob.n,
        prob.scale,
        prob.causal);
    check(cudaStreamSynchronize(nullptr));

    if (prob.verify) {
      copy(dP.begin(), dP.end(), hP.begin());
      result.maxRelDiff = max_rel_diff(ref, prob.m, prob.n, hP, prob.n);
      result.checked = true;
      result.verified = result.maxRelDiff <= kMaxRelDiff;
    }

    result.usPerIter = benchmark(impl, prob, dQ.data().get(), dK.data().get(), dP.data().get());
    result.timed = true;
  } catch (const exception &e) {
    result.error = e.what();
  }
  return result;
}

array<Result, kImplCount> run_all(
    const Problem &prob,
    const Reference &ref,
    host_vector<__half> &hP,
    const device_vector<__half> &dQ,
    const device_vector<__half> &dK,
    device_vector<__half> &dP) {
  auto results = array<Result, kImplCount>{};
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    results[i] = run_impl(kImpls[i], prob, ref, hP, dQ, dK, dP);
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
  // TFLOP/s prices every implementation at the 2*M*N*K multiply-adds the score matrix needs, which
  // is the work that has to happen regardless of how it is fused.
  //
  // GB/s prices every implementation at 2*M*N bytes - the write of P and nothing else. That is the
  // irreducible traffic of this problem, so the column reads as "how close to the output floor":
  // cublas_unfused_softmax moves 5x what it is priced for (4*M*N of fp32 scores written, the same
  // read back, 2*M*N of P), and shows up here as a fraction of the card's 448 GB/s even when its
  // DRAM is saturated. The fused kernels also read Q and K, which at these shapes is ~12% on top.
  const auto flops = 2.0 * static_cast<double>(prob.m) * prob.n * prob.k;
  const auto trafficBytes = 2.0 * static_cast<double>(prob.m) * prob.n * sizeof(__half);
  const auto baselineUs = results[kBaselineIndex].timed ? results[kBaselineIndex].usPerIter : 0.0;

  printf(
      "fused softmax of (%d x %d) by (%d x %d)^T, %s mask, scale %.6f, %d timed iterations\n",
      prob.m,
      prob.k,
      prob.n,
      prob.k,
      prob.causal ? "causal" : "no",
      prob.scale,
      prob.iterations);
  printf(
      "%-24s %-8s %13s %10s %9s %8s %10s\n",
      "implementation",
      "verify",
      "max_rel_diff",
      "time_us",
      "TFLOP/s",
      "GB/s",
      "vs_cublas");
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    const auto &result = results[i];
    const auto tflops = flops / (result.usPerIter * 1e-6) / 1e12;
    const auto gbps = trafficBytes / (result.usPerIter * 1e-6) / 1e9;
    const auto speedup = baselineUs / result.usPerIter;
    const char *verifyText = "-";
    if (result.checked) {
      verifyText = result.verified ? "PASS" : "FAIL";
    }
    printf(
        "%-24s %-8s %13s %10s %9s %8s %10s\n",
        kImpls[i].name,
        verifyText,
        number(result.checked, "%.3e", result.maxRelDiff).c_str(),
        number(result.timed, "%.1f", result.usPerIter).c_str(),
        number(result.timed, "%.1f", tflops).c_str(),
        number(result.timed, "%.1f", gbps).c_str(),
        number(result.timed && baselineUs > 0.0, "%.2fx", speedup).c_str());
    if (!result.error.empty()) {
      printf("    %s\n", result.error.c_str());
    }
  }
}

} // namespace

int main(int argc, char const **argv) {
  auto cmd = CommandLine(argc, argv);
  auto prob = Problem();
  auto mask = string(kDefaultMask);
  cmd.get_cmd_line_argument("m", prob.m, kDefaultM);
  cmd.get_cmd_line_argument("n", prob.n, kDefaultN);
  cmd.get_cmd_line_argument("k", prob.k, kDefaultK);
  cmd.get_cmd_line_argument("iterations", prob.iterations, kDefaultIterations);
  cmd.get_cmd_line_argument("mask", mask, string(kDefaultMask));
  // 1/sqrt(k) is what keeps the scores' variance at 1 as k grows, and so what keeps exp() in the
  // range where a softmax is not dominated by its largest element.
  cmd.get_cmd_line_argument("scale", prob.scale, 1.0f / sqrt(static_cast<float>(prob.k)));
  cmd.get_cmd_line_argument("verify", prob.verify, true);

  if (prob.m <= 0 || prob.n <= 0 || prob.k <= 0 || prob.iterations <= 0) {
    fprintf(
        stderr,
        "usage: %s [--m=N] [--n=N] [--k=N] [--iterations=N] [--mask=causal|none] [--scale=F] "
        "[--verify=true|false], sizes positive\n",
        argv[0]);
    return EXIT_FAILURE;
  }
  if (mask != "causal" && mask != "none") {
    fprintf(stderr, "--mask must be causal or none\n");
    return EXIT_FAILURE;
  }
  prob.causal = mask == "causal";
  // A query past the key range would see no column at all under a causal mask, and its softmax
  // would be 0/0. Every implementation is allowed to assume that does not happen.
  if (prob.causal && prob.m > prob.n) {
    fprintf(stderr, "a causal mask needs m <= n, otherwise some query sees no key at all\n");
    return EXIT_FAILURE;
  }
  // The implementations index with int.
  for (const auto &product :
       {static_cast<int64_t>(prob.m) * prob.n,
        static_cast<int64_t>(prob.m) * prob.k,
        static_cast<int64_t>(prob.n) * prob.k}) {
    if (product > INT32_MAX) {
      fprintf(stderr, "m * n, m * k and n * k must each fit in int32\n");
      return EXIT_FAILURE;
    }
  }

  try {
    // cublas_unfused_softmax allocates its m x n score buffer with cudaMallocAsync on every call.
    // cudaFreeAsync hands pages back to the driver once the pool's release threshold (0 by default)
    // is exceeded, which would put a real allocation inside the timed region and measure the
    // allocator rather than the kernels.
    cudaMemPool_t pool = nullptr;
    check(cudaDeviceGetDefaultMemPool(&pool, 0));
    auto threshold = uint64_t{UINT64_MAX};
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    auto engine = mt19937(20261012);
    auto hQ = make_input(static_cast<size_t>(prob.m) * prob.k, engine);
    auto hK = make_input(static_cast<size_t>(prob.n) * prob.k, engine);
    auto hP = host_vector<__half>(static_cast<size_t>(prob.m) * prob.n);
    auto dQ = device_vector<__half>(hQ);
    auto dK = device_vector<__half>(hK);
    auto dP = device_vector<__half>(hP.size());

    // The reference is the slow part of the harness: it is an m * n * k triple loop in double on
    // the host, which at the default shape is 5.4e8 multiply-adds and costs a few seconds.
    // --verify=false skips it, and with it the check, so a sweep can time shapes the reference
    // cannot afford.
    auto ref = prob.verify
                   ? make_reference(
                         prob.m, prob.n, prob.k, hQ, prob.k, hK, prob.k, prob.scale, prob.causal)
                   : Reference();

    print_table(prob, run_all(prob, ref, hP, dQ, dK, dP));
  } catch (const exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
