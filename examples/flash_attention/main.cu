#include <algorithm>
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
#include <sstream>
#include <stdexcept>
#include <string>

#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

#include "attention_common.cuh"
#include "cublas_unfused_attention.cuh"
#include "flash_attention.cuh"
#include "flash_attention_naive.cuh"
#include "flash_attention_upstream.cuh"
#include "reference.cuh"

using cutlass::CommandLine;
using std::array;
using std::exception;
using std::fprintf;
using std::int64_t;
using std::invalid_argument;
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
// Half output and, for the tensor-core paths, half probabilities introduce rounding. Attention
// outputs can be negative or close to zero: use |got-want| / max(|want|, 1), not a probability's
// pure relative error. This is also the metric used by the RMSNorm example.
constexpr double kMaxRelDiff = 1e-3;

using AttentionFn = void (*)(
    int,
    int,
    int,
    const __half *,
    int,
    const __half *,
    int,
    const __half *,
    int,
    __half *,
    int,
    float,
    bool);

struct Impl {
  const char *name;
  AttentionFn run;
};

constexpr size_t kImplCount = 4;
constexpr array<Impl, kImplCount> kImpls{{
    Impl{"flash_attention_naive", flash_attention_naive},
    Impl{"cublas_unfused_attention", cublas_unfused_attention},
    Impl{"flash_attention", flash_attention},
    Impl{"flash_attention_upstream", flash_attention_upstream},
}};
constexpr size_t kBaselineIndex = 1;
constexpr size_t kFlashAttentionIndex = 3;

struct Problem {
  int m = 0;
  int n = 0;
  int k = 0;
  int iterations = 0;
  float scale = 0.0f;
  bool causal = true;
  bool verify = true;
  string input = "random";
};

struct Result {
  bool checked = false;
  bool verified = false;
  bool timed = false;
  bool unsupported = false;
  double maxRelDiff = 0.0;
  double usPerIter = 0.0;
  string error;
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

void prepare_input(
    const Problem &prob,
    host_vector<__half> &hQ,
    host_vector<__half> &hK,
    host_vector<__half> &hV) {
  if (prob.input == "zero-qk" || prob.input == "late-max") {
    std::fill(hQ.begin(), hQ.end(), __float2half(0.0f));
    std::fill(hK.begin(), hK.end(), __float2half(0.0f));
  }
  if (prob.input == "constant-v") {
    std::fill(hV.begin(), hV.end(), __float2half(0.5f));
  }
  if (prob.input == "late-max") {
    // A later key tile changes the maximum by 128 * scale. With --scale=1 this forces the
    // output rescale to discard the early numerator without overflowing the softmax exponent.
    for (auto row = 0; row < prob.m; ++row) {
      hQ[static_cast<size_t>(row) * prob.k] = __float2half(8.0f);
    }
    for (auto row = 0; row < prob.n; ++row) {
      hK[static_cast<size_t>(row) * prob.k] = __float2half(row < prob.n / 2 ? -8.0f : 8.0f);
    }
  }
}

void invoke(
    const Impl &impl,
    const Problem &prob,
    const device_vector<__half> &dQ,
    const device_vector<__half> &dK,
    const device_vector<__half> &dV,
    device_vector<__half> &dO) {
  impl.run(
      prob.m,
      prob.n,
      prob.k,
      dQ.data().get(),
      prob.k,
      dK.data().get(),
      prob.k,
      dV.data().get(),
      prob.k,
      dO.data().get(),
      prob.k,
      prob.scale,
      prob.causal);
  // Equal footing for wrappers that synchronize internally and wrappers that only enqueue work.
  check(cudaStreamSynchronize(nullptr));
}

double benchmark(
    const Impl &impl,
    const Problem &prob,
    const device_vector<__half> &dQ,
    const device_vector<__half> &dK,
    const device_vector<__half> &dV,
    device_vector<__half> &dO) {
  for (auto i = 0; i < kWarmupIterations; ++i) {
    invoke(impl, prob, dQ, dK, dV, dO);
  }
  auto timer = GPU_Clock();
  timer.start();
  for (auto i = 0; i < prob.iterations; ++i) {
    invoke(impl, prob, dQ, dK, dV, dO);
  }
  return static_cast<double>(timer.milliseconds()) * 1000.0 / prob.iterations;
}

Result run_impl(
    const Impl &impl,
    const Problem &prob,
    const Reference &ref,
    host_vector<__half> &hO,
    const device_vector<__half> &dQ,
    const device_vector<__half> &dK,
    const device_vector<__half> &dV,
    device_vector<__half> &dO) {
  auto result = Result();
  try {
    if (prob.verify) {
      // fp16 0xFFFF is NaN: missing stores cannot inherit another implementation's valid output.
      check(cudaMemset(dO.data().get(), 0xFF, dO.size() * sizeof(__half)));
    }
    invoke(impl, prob, dQ, dK, dV, dO);
    if (prob.verify) {
      copy(dO.begin(), dO.end(), hO.begin());
      result.maxRelDiff = max_rel_diff(ref, prob.m, prob.k, hO, prob.k);
      result.checked = true;
      result.verified = result.maxRelDiff <= kMaxRelDiff;
      if (!result.verified) {
        return result;
      }
    }
    result.usPerIter = benchmark(impl, prob, dQ, dK, dV, dO);
    result.timed = true;
  } catch (const invalid_argument &e) {
    result.unsupported = true;
    result.error = e.what();
  } catch (const exception &e) {
    result.error = e.what();
  }
  return result;
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
  // Dense-equivalent QK + PV multiply-add count. Causal kernels can skip masked work, so this is
  // not a hardware utilization measure. GB/s is only O's write floor, not measured DRAM traffic.
  const auto flops = 4.0 * static_cast<double>(prob.m) * prob.n * prob.k;
  const auto trafficBytes = static_cast<double>(prob.m) * prob.k * sizeof(__half);
  const auto baselineUs = results[kBaselineIndex].timed ? results[kBaselineIndex].usPerIter : 0.0;
  const auto upstreamUs =
      results[kFlashAttentionIndex].timed ? results[kFlashAttentionIndex].usPerIter : 0.0;
  printf(
      "attention Q(%d x %d), K/V(%d x %d), %s mask, scale %.6f, input %s, %d timed iterations\n",
      prob.m,
      prob.k,
      prob.n,
      prob.k,
      prob.causal ? "causal" : "no",
      prob.scale,
      prob.input.c_str(),
      prob.iterations);
  printf(
      "%-24s %-8s %13s %10s %9s %8s %10s %10s\n",
      "implementation",
      "verify",
      "max_rel_diff",
      "time_us",
      "TFLOP/s",
      "GB/s",
      "vs_cublas",
      "vs_fa2");
  for (auto i = size_t{0}; i < kImplCount; ++i) {
    const auto &result = results[i];
    const auto tflops = result.timed ? flops / (result.usPerIter * 1e-6) / 1e12 : 0.0;
    const auto gbps = result.timed ? trafficBytes / (result.usPerIter * 1e-6) / 1e9 : 0.0;
    const auto speedup = result.timed ? baselineUs / result.usPerIter : 0.0;
    const auto upstreamSpeedup = result.timed ? upstreamUs / result.usPerIter : 0.0;
    const char *verifyText = result.unsupported ? "n/a" : "-";
    if (result.checked) {
      verifyText = result.verified ? "PASS" : "FAIL";
    }
    printf(
        "%-24s %-8s %13s %10s %9s %8s %10s %10s\n",
        kImpls[i].name,
        verifyText,
        number(result.checked, "%.3e", result.maxRelDiff).c_str(),
        number(result.timed, "%.1f", result.usPerIter).c_str(),
        number(result.timed, "%.1f", tflops).c_str(),
        number(result.timed, "%.1f", gbps).c_str(),
        number(result.timed && baselineUs > 0.0, "%.2fx", speedup).c_str(),
        number(result.timed && upstreamUs > 0.0, "%.2fx", upstreamSpeedup).c_str());
    if (!result.error.empty()) {
      printf("    %s\n", result.error.c_str());
    }
  }
}

template <class Value> bool valid_number(const string &text) {
  auto stream = std::istringstream(text);
  auto value = Value();
  stream >> std::noskipws >> value;
  return !stream.fail() && stream.eof();
}

// Keep the neighboring examples' CommandLine interface, but do not let its unchecked stream
// extraction silently accept --m=32junk, empty values, or misspelled booleans.
bool valid_options(const CommandLine &cmd) {
  if (!cmd.args.empty()) {
    fprintf(stderr, "options must use --name=value, with no positional arguments\n");
    return false;
  }
  for (auto i = size_t{0}; i < cmd.keys.size(); ++i) {
    const auto &key = cmd.keys[i];
    const auto &value = cmd.values[i];
    auto valid = false;
    if (key == "m" || key == "n" || key == "k" || key == "iterations") {
      valid = valid_number<int>(value);
    } else if (key == "scale") {
      valid = valid_number<float>(value);
    } else if (key == "verify") {
      valid = value == "true" || value == "false" || value == "1" || value == "0";
    } else if (key == "mask") {
      valid = value == "causal" || value == "none";
    } else if (key == "input") {
      valid =
          value == "random" || value == "zero-qk" || value == "constant-v" || value == "late-max";
    }
    if (!valid) {
      fprintf(stderr, "unknown option or invalid value: --%s=%s\n", key.c_str(), value.c_str());
      return false;
    }
  }
  return true;
}

void print_usage(const char *program) {
  fprintf(
      stderr,
      "usage: %s [--m=N] [--n=N] [--k=N] [--iterations=N] [--mask=causal|none] [--scale=F] "
      "[--verify=true|false] [--input=random|zero-qk|constant-v|late-max]\n",
      program);
}

} // namespace

int main(int argc, char const **argv) {
  auto cmd = CommandLine(argc, argv);
  if (cmd.check_cmd_line_flag("help")) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  if (!valid_options(cmd)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  auto prob = Problem();
  auto mask = string("causal");
  cmd.get_cmd_line_argument("m", prob.m, kDefaultM);
  cmd.get_cmd_line_argument("n", prob.n, kDefaultN);
  cmd.get_cmd_line_argument("k", prob.k, kDefaultK);
  cmd.get_cmd_line_argument("iterations", prob.iterations, kDefaultIterations);
  cmd.get_cmd_line_argument("mask", mask, string("causal"));
  cmd.get_cmd_line_argument("verify", prob.verify, true);
  cmd.get_cmd_line_argument("input", prob.input, string("random"));
  if (prob.m <= 0 || prob.n <= 0 || prob.k <= 0 || prob.iterations <= 0) {
    print_usage(argv[0]);
    fprintf(stderr, "sizes and iterations must be positive\n");
    return EXIT_FAILURE;
  }
  cmd.get_cmd_line_argument("scale", prob.scale, 1.0f / sqrt(static_cast<float>(prob.k)));
  if (!attention::valid_scale(prob.k, prob.scale)) {
    fprintf(stderr, "--scale must be finite and keep fp16-input logits within fp32 range\n");
    return EXIT_FAILURE;
  }
  if (mask != "causal" && mask != "none") {
    fprintf(stderr, "--mask must be causal or none\n");
    return EXIT_FAILURE;
  }
  prob.causal = mask == "causal";
  if (prob.causal && prob.m > prob.n) {
    fprintf(stderr, "a causal mask needs m <= n\n");
    return EXIT_FAILURE;
  }
  if (prob.input != "random" && prob.input != "zero-qk" && prob.input != "constant-v" &&
      prob.input != "late-max") {
    fprintf(stderr, "--input must be random, zero-qk, constant-v or late-max\n");
    return EXIT_FAILURE;
  }
  for (const auto &product :
       {static_cast<int64_t>(prob.m) * prob.n,
        static_cast<int64_t>(prob.m) * prob.k,
        static_cast<int64_t>(prob.n) * prob.k}) {
    if (product > INT32_MAX) {
      fprintf(stderr, "m*n, m*k and n*k must each fit in int32\n");
      return EXIT_FAILURE;
    }
  }

  try {
    // The cuBLAS baseline allocates S/P using the default pool. Keep released pages cached so
    // the timing measures kernels rather than a fresh allocation at every iteration.
    int device = 0;
    check(cudaGetDevice(&device));
    cudaMemPool_t pool = nullptr;
    check(cudaDeviceGetDefaultMemPool(&pool, device));
    auto threshold = std::uint64_t{UINT64_MAX};
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    auto engine = mt19937(20261027);
    auto hQ = make_input(static_cast<size_t>(prob.m) * prob.k, engine);
    auto hK = make_input(static_cast<size_t>(prob.n) * prob.k, engine);
    auto hV = make_input(static_cast<size_t>(prob.n) * prob.k, engine);
    prepare_input(prob, hQ, hK, hV);
    auto hO = host_vector<__half>(static_cast<size_t>(prob.m) * prob.k);
    auto dQ = device_vector<__half>(hQ);
    auto dK = device_vector<__half>(hK);
    auto dV = device_vector<__half>(hV);
    auto dO = device_vector<__half>(hO.size());
    auto ref = prob.verify ? make_reference(
                                 prob.m,
                                 prob.n,
                                 prob.k,
                                 hQ,
                                 prob.k,
                                 hK,
                                 prob.k,
                                 hV,
                                 prob.k,
                                 prob.scale,
                                 prob.causal)
                           : Reference();

    auto results = array<Result, kImplCount>{};
    auto failed = false;
    auto ran = false;
    for (auto i = size_t{0}; i < kImplCount; ++i) {
      results[i] = run_impl(kImpls[i], prob, ref, hO, dQ, dK, dV, dO);
      const auto &result = results[i];
      failed = failed || (result.checked && !result.verified) ||
               (!result.error.empty() && !result.unsupported);
      ran = ran || result.timed;
    }
    print_table(prob, results);
    return failed || !ran ? EXIT_FAILURE : EXIT_SUCCESS;
  } catch (const exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return EXIT_FAILURE;
  }
}
