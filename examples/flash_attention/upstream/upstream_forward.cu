#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

#include "cutlass/numeric_types.h"

#include "attention_common.cuh"
#include "flash.h"
#include "flash_attention_upstream.cuh"

using std::int64_t;
using std::invalid_argument;
using std::numeric_limits;
using std::runtime_error;
using std::size_t;
using std::uintptr_t;

namespace {

constexpr float kLog2E = 1.4426950408889634f;
constexpr int kSequenceAlignment = 128;

void check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

// LSE is a required output of the upstream kernel even though this example only consumes O.
// Match the cuBLAS wrapper's asynchronous scratch allocation/release inside each invocation;
// main's retained default memory pool keeps real page allocation out of warmed-up timings.
struct LseScratch {
  void *data_ = nullptr;

  explicit LseScratch(size_t bytes) { check(cudaMallocAsync(&data_, bytes, nullptr)); }
  LseScratch(const LseScratch &) = delete;
  LseScratch &operator=(const LseScratch &) = delete;
  LseScratch(LseScratch &&) = delete;
  LseScratch &operator=(LseScratch &&) = delete;
  ~LseScratch() {
    if (data_ != nullptr) {
      (void)cudaFreeAsync(data_, nullptr);
    }
  }

  void release() {
    check(cudaFreeAsync(data_, nullptr));
    data_ = nullptr;
  }
};

int rounded_sequence(int length) {
  const auto rounded = (static_cast<int64_t>(length) + kSequenceAlignment - 1) /
                       kSequenceAlignment * kSequenceAlignment;
  if (rounded > numeric_limits<int>::max()) {
    throw invalid_argument("FlashAttention-2 rounded sequence length must fit in int32");
  }
  return static_cast<int>(rounded);
}

void dispatch(flash::Flash_fwd_params &params, int k, bool causal) {
  // Call only the four explicitly compiled upstream specializations. The Python-facing generic
  // dispatcher would also reference other head dimensions, BF16 and split-KV translation units.
  if (k == 64) {
    if (causal) {
      flash::run_mha_fwd_<cutlass::half_t, 64, true>(params, nullptr);
    } else {
      flash::run_mha_fwd_<cutlass::half_t, 64, false>(params, nullptr);
    }
  } else {
    if (causal) {
      flash::run_mha_fwd_<cutlass::half_t, 128, true>(params, nullptr);
    } else {
      flash::run_mha_fwd_<cutlass::half_t, 128, false>(params, nullptr);
    }
  }
}

} // namespace

void flash_attention_upstream(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    const __half *dV,
    int ldV,
    __half *dO,
    int ldO,
    float scale,
    bool causal) {
  if (m <= 0 || n <= 0 || (k != 64 && k != 128) || ldQ < k || ldK < k || ldV < k || ldO < k ||
      dQ == nullptr || dK == nullptr || dV == nullptr || dO == nullptr) {
    throw invalid_argument(
        "FlashAttention-2 needs positive sequence lengths, k = 64/128, valid strides and buffers");
  }
  if (causal && m > n) {
    throw invalid_argument("FlashAttention-2 baseline requires m <= n for causal attention");
  }
  // Upstream reduces unscaled scores before applying scale. With masks or tails, scale <= 0 can
  // turn -inf into NaN/+inf; leave these cases to the other implementations rather than fake FA2.
  if (!attention::valid_scale(k, scale) || scale <= 0.0f) {
    throw invalid_argument("FlashAttention-2 baseline requires a positive fp32-safe scale");
  }
  if (ldQ % 8 != 0 || ldK % 8 != 0 || ldV % 8 != 0 || ldO % 8 != 0 ||
      reinterpret_cast<uintptr_t>(dQ) % 16 != 0 || reinterpret_cast<uintptr_t>(dK) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(dV) % 16 != 0 || reinterpret_cast<uintptr_t>(dO) % 16 != 0) {
    throw invalid_argument(
        "FlashAttention-2 vector loads/stores need 16-byte-aligned Q/K/V/O rows");
  }
  const auto qRounded = rounded_sequence(m);
  const auto kRounded = rounded_sequence(n);
  if (static_cast<size_t>(m) > numeric_limits<size_t>::max() / sizeof(float)) {
    throw invalid_argument("FlashAttention-2 LSE allocation size overflows size_t");
  }

  auto scratch = LseScratch(static_cast<size_t>(m) * sizeof(float));
  // This is the pinned upstream set_params_fprop layout, without Torch tensor wrappers. Strides
  // are in ELEMENTS, not bytes. Batch/head strides are harmless for b=h=1 but set explicitly.
  auto params = flash::Flash_fwd_params{};
  params.q_ptr = const_cast<__half *>(dQ);
  params.k_ptr = const_cast<__half *>(dK);
  params.v_ptr = const_cast<__half *>(dV);
  params.o_ptr = dO;
  params.q_row_stride = ldQ;
  params.k_row_stride = ldK;
  params.v_row_stride = ldV;
  params.o_row_stride = ldO;
  params.q_head_stride = k;
  params.k_head_stride = k;
  params.v_head_stride = k;
  params.o_head_stride = k;
  params.q_batch_stride = static_cast<int64_t>(m) * ldQ;
  params.k_batch_stride = static_cast<int64_t>(n) * ldK;
  params.v_batch_stride = static_cast<int64_t>(n) * ldV;
  params.o_batch_stride = static_cast<int64_t>(m) * ldO;
  params.b = 1;
  params.h = 1;
  params.h_k = 1;
  params.h_h_k_ratio = 1;
  params.seqlen_q = m;
  params.seqlen_k = n;
  params.seqlen_q_rounded = qRounded;
  params.seqlen_k_rounded = kRounded;
  params.d = k;
  params.d_rounded = k;
  params.softmax_lse_ptr = static_cast<float *>(scratch.data_);
  params.scale_softmax = scale;
  params.scale_softmax_log2 = scale * kLog2E;
  params.scale_softmax_rp_dropout = scale;
  // Upstream's p_dropout stores KEEP probability, unlike a user-facing dropout probability.
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.softcap = 0.0f;
  params.num_splits = 1;
  params.is_bf16 = false;
  params.is_causal = causal;
  params.window_size_left = causal ? n : -1;
  params.window_size_right = causal ? 0 : -1;
  params.is_seqlens_k_cumulative = true;
  params.unpadded_lse = false;
  params.seqlenq_ngroups_swapped = false;
  // Philox unpacking is present even in stable no-dropout kernels. Give it a benign state without
  // requesting a generator or allocating a RNG output; optional pointers remain null.
  params.philox_args = at::PhiloxCudaState(0, 0);

  dispatch(params, k, causal);
  check(cudaGetLastError());
  scratch.release();
  check(cudaStreamSynchronize(nullptr));
}
