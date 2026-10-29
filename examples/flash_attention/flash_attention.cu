#include "flash_attention.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cute/tensor.hpp" // IWYU pragma: keep

#include "../fused_softmax/fused_softmax_mainloop.cuh"
#include "attention_common.cuh"

using cute::_;
using cute::_1;
using cute::Copy_Atom;
using cute::half_t;
using cute::Int;
using cute::Layout;
using cute::local_tile;
using cute::make_coord;
using cute::make_gmem_ptr;
using cute::make_identity_tensor;
using cute::make_layout;
using cute::make_shape;
using cute::make_smem_ptr;
using cute::make_stride;
using cute::make_tensor;
using cute::make_tiled_copy;
using cute::make_tiled_mma;
using cute::Shape;
using cute::size;
using cute::SM80_16x8x16_F32F16F16F32_TN;
using cute::Step;
using cute::Stride;
using cute::Tensor;
using cute::Tile;
using cute::uint128_t;
using cute::UniversalCopy;
using cute::X;
using fused_softmax::compute_tile;
using fused_softmax::kBlockK;
using fused_softmax::kBlockM;
using fused_softmax::kMmaAtomsM;
using fused_softmax::kMmaAtomsN;
using fused_softmax::kMmaK;
using fused_softmax::kMmaM;
using fused_softmax::kMmaN;
using fused_softmax::kThreads;
using fused_softmax::make_k_major_smem_layout;
using std::invalid_argument;
using std::runtime_error;

namespace {

constexpr int kBlockN = 64;
constexpr int kWarps = kThreads / 32;

// No buffers alias in this first version. Q/K staging is reused across head-dimension tiles,
// while S, P and V remain distinct until PV has consumed them. Even k = 128 fits below 48 KB.
template <int kHeadDim> constexpr std::size_t smem_bytes() {
  return ((static_cast<std::size_t>(kBlockM) + kBlockN) * kBlockK +
          static_cast<std::size_t>(kBlockM) * kBlockN +
          static_cast<std::size_t>(kBlockN) * kHeadDim) *
             sizeof(half_t) +
         static_cast<std::size_t>(kBlockM) * kBlockN * sizeof(float);
}

static_assert(smem_bytes<128>() + 3ULL * kBlockM * sizeof(float) <= 48ULL * 1024);

// A warp owns a logical row here, independently of the MMA's fragment layout. This staging costs
// an extra shared-memory round trip, but keeps the online recurrence and PV operand conversion
// visible without assuming that a QK output fragment can also be used as a PV input fragment.
__device__ void update_softmax(
    const float *scores,
    half_t *probs,
    float *rowMax,
    float *rowSum,
    float *rowScale,
    int rowBase,
    int colBase,
    int m,
    int n,
    float scale,
    bool causal) {
  const auto warpId = static_cast<int>(threadIdx.x) / 32;
  const auto laneId = static_cast<int>(threadIdx.x) % 32;
  for (auto row = warpId; row < kBlockM; row += kWarps) {
    float values[kBlockN / 32];
    auto tileMax = -INFINITY;
    CUTE_UNROLL
    for (auto c = 0; c < kBlockN / 32; ++c) {
      const auto col = laneId + c * 32;
      auto value = scores[row * kBlockN + col] * scale;
      if (causal && colBase + col > rowBase + row + n - m) {
        value = -INFINITY;
      }
      values[c] = value;
      tileMax = fmaxf(tileMax, value);
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, offset));
    }

    const auto oldMax = rowMax[row];
    const auto newMax = fmaxf(oldMax, tileMax);
    // An empty tile contributes no probabilities. An empty old state contributes no numerator;
    // neither needs the undefined exp(-inf - -inf). Causal m <= n makes the first tile nonempty.
    const auto correction = oldMax == -INFINITY ? 0.0f : __expf(oldMax - newMax);
    auto tileSum = 0.0f;
    CUTE_UNROLL
    for (auto c = 0; c < kBlockN / 32; ++c) {
      const auto value = values[c] == -INFINITY ? 0.0f : __expf(values[c] - newMax);
      tileSum += value;
      probs[row * kBlockN + laneId + c * 32] = half_t(value);
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      tileSum += __shfl_xor_sync(0xffffffffu, tileSum, offset);
    }
    if (laneId == 0) {
      rowMax[row] = newMax;
      rowSum[row] = correction * rowSum[row] + tileSum;
      rowScale[row] = correction;
    }
  }
}

template <int kHeadDim>
__global__ __launch_bounds__(kThreads) void flash_attention_device(
    const __half *__restrict__ dQ,
    int ldQ,
    const __half *__restrict__ dK,
    int ldK,
    const __half *__restrict__ dV,
    int ldV,
    __half *__restrict__ dO,
    int ldO,
    int m,
    int n,
    float scale,
    bool causal) {
  extern __shared__ __align__(16) char sRaw[];
  __shared__ float sRowMax[kBlockM];
  __shared__ float sRowSum[kBlockM];
  __shared__ float sRowScale[kBlockM];

  const auto bM = Int<kBlockM>{};
  const auto bN = Int<kBlockN>{};
  const auto bK = Int<kBlockK>{};
  const auto bD = Int<kHeadDim>{};
  const auto ctaTiler = make_shape(bM, bN, bK);
  const auto rowBlock = static_cast<int>(blockIdx.x);
  const auto rowBase = rowBlock * kBlockM;
  const auto threadId = static_cast<int>(threadIdx.x);

  const auto *query = reinterpret_cast<const half_t *>(dQ);
  const auto *key = reinterpret_cast<const half_t *>(dK);
  const auto *value = reinterpret_cast<const half_t *>(dV);
  auto *out = reinterpret_cast<half_t *>(dO);
  const Tensor mQ = make_tensor(make_gmem_ptr(query), make_shape(m, bD), make_stride(ldQ, _1{}));
  const Tensor mK = make_tensor(make_gmem_ptr(key), make_shape(n, bD), make_stride(ldK, _1{}));
  const Tensor mV = make_tensor(make_gmem_ptr(value), make_shape(n, bD), make_stride(ldV, _1{}));
  Tensor mO = make_tensor(make_gmem_ptr(out), make_shape(m, bD), make_stride(ldO, _1{}));
  const auto ctaCoord = make_coord(rowBlock, _, _);
  const Tensor gQ = local_tile(mQ, ctaTiler, ctaCoord, Step<_1, X, _1>{});
  const Tensor gK = local_tile(mK, ctaTiler, ctaCoord, Step<X, _1, _1>{});
  const Tensor gV = local_tile(mV, make_shape(bN, bD), make_coord(_, 0));
  Tensor gO = local_tile(mO, make_shape(bM, bD), make_coord(rowBlock, 0));

  auto *sQPtr = reinterpret_cast<half_t *>(sRaw);
  auto *sKPtr = sQPtr + static_cast<std::size_t>(kBlockM) * kBlockK;
  auto *sPPtr = sKPtr + static_cast<std::size_t>(kBlockN) * kBlockK;
  auto *sVPtr = sPPtr + static_cast<std::size_t>(kBlockM) * kBlockN;
  auto *sScoresPtr =
      reinterpret_cast<float *>(sVPtr + static_cast<std::size_t>(kBlockN) * kHeadDim);
  const Tensor sQ = make_tensor(make_smem_ptr(sQPtr), make_k_major_smem_layout(bM, bK));
  const Tensor sK = make_tensor(make_smem_ptr(sKPtr), make_k_major_smem_layout(bN, bK));
  const Tensor sP =
      make_tensor(make_smem_ptr(sPPtr), make_layout(make_shape(bM, bN), make_stride(bN, _1{})));
  const Tensor sV =
      make_tensor(make_smem_ptr(sVPtr), make_layout(make_shape(bN, bD), make_stride(bD, _1{})));
  // MMA expresses C(M,N) = A(M,K) B(N,K)^T. For PV its reduction K is the key axis:
  // B(feature,key) must be V(key,feature), a transpose VIEW of the actual row-major V staging.
  // V is deliberately unswizzled here, so stride (1, kHeadDim) is precisely that transpose.
  const Tensor sVt =
      make_tensor(make_smem_ptr(sVPtr), make_layout(make_shape(bD, bN), make_stride(_1{}, bD)));
  Tensor sScores = make_tensor(
      make_smem_ptr(sScoresPtr), make_layout(make_shape(bM, bN), make_stride(bN, _1{})));

  auto tiledCopy = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
      Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  auto tiledMma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<Int<kMmaAtomsM>, Int<kMmaAtomsN>>>{},
      Tile<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>{});
  const auto thrMma = tiledMma.get_slice(threadIdx.x);
  const auto thrCopy = decltype(tiledCopy)::get_slice(threadIdx.x);
  const Tensor tVgV = thrCopy.partition_S(gV);
  Tensor tVsV = thrCopy.partition_D(sV);
  Tensor tSsS = thrMma.partition_C(sScores);
  Tensor tSrS = thrMma.partition_fragment_C(sScores);
  Tensor tOrO = thrMma.partition_fragment_C(gO);
  Tensor tOgO = thrMma.partition_C(gO);
  const Tensor tOcO = thrMma.partition_C(make_identity_tensor(make_shape(bM, bD)));

  // P and V are repartitioned as MMA OPERANDS. The QK accumulator is neither the correct shape
  // nor the correct register order for this. Generic copies preserve the logical coordinates;
  // QK's K-major ldmatrix B atom cannot simply be reused for the N-major V transpose.
  const Tensor tPsP = thrMma.partition_A(sP);
  const Tensor tVsVt = thrMma.partition_B(sVt);
  Tensor tPrP = thrMma.partition_fragment_A(sP);
  Tensor tVrV = thrMma.partition_fragment_B(sVt);
  static_assert(size(tPsP) == size(tPrP));
  static_assert(size(tVsVt) == size(tVrV));

  cute::clear(tOrO);
  for (auto row = threadId; row < kBlockM; row += kThreads) {
    sRowMax[row] = -INFINITY;
    sRowSum[row] = 0.0f;
    sRowScale[row] = 0.0f;
  }
  __syncthreads();

  const int nTiles = size<2>(gK);
  const int kTiles = size<2>(gQ);
  const auto lastVisible = causal ? rowBase + kBlockM - 1 + n - m : n - 1;
  const auto activeTiles = min(nTiles, lastVisible / kBlockN + 1);
  for (auto tile = 0; tile < activeTiles; ++tile) {
    // Only the TEMPORARY score fragment is cleared by compute_tile. O stays alive across tiles.
    compute_tile(tiledCopy, tiledMma, gQ, sQ, gK(_, _, tile, _), sK, tSrS, kTiles);
    cute::copy(tSrS, tSsS);
    cute::copy(tiledCopy, tVgV(_, _, _, tile), tVsV);
    __syncthreads();

    update_softmax(
        sScoresPtr,
        sPPtr,
        sRowMax,
        sRowSum,
        sRowScale,
        rowBase,
        tile * kBlockN,
        m,
        n,
        scale,
        causal);
    __syncthreads();

    CUTE_UNROLL
    for (auto i = 0; i < size(tOrO); ++i) {
      const auto row = static_cast<int>(cute::get<0>(tOcO(i)));
      tOrO(i) *= sRowScale[row];
    }
    CUTE_UNROLL
    for (auto kb = 0; kb < size<2>(tPrP); ++kb) {
      cute::copy(tPsP(_, _, kb), tPrP(_, _, kb));
      cute::copy(tVsVt(_, _, kb), tVrV(_, _, kb));
      cute::gemm(tiledMma, tPrP(_, _, kb), tVrV(_, _, kb), tOrO);
    }
    // Every warp must finish consuming P/V before another tile overwrites their shared buffers.
    __syncthreads();
  }

  CUTE_UNROLL
  for (auto i = 0; i < size(tOrO); ++i) {
    const auto row = static_cast<int>(cute::get<0>(tOcO(i)));
    tOgO(i) = half_t(tOrO(i) / sRowSum[row]);
  }
}

template <int kHeadDim>
void launch_attention(
    int m,
    int n,
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
  flash_attention_device<kHeadDim><<<m / kBlockM, kThreads, smem_bytes<kHeadDim>()>>>(
      dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, m, n, scale, causal);
}

} // namespace

void flash_attention(
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
  static_assert(sizeof(__half) == sizeof(half_t));
  if (m <= 0 || n <= 0 || (k != 64 && k != 128) || m % kBlockM != 0 || n % kBlockN != 0) {
    throw invalid_argument("flash_attention needs m % 32 = 0, n % 64 = 0 and k = 64 or 128");
  }
  if (causal && m > n) {
    throw invalid_argument("a causal mask needs m <= n");
  }
  if (!attention::valid_scale(k, scale) || ldQ < k || ldK < k || ldV < k || ldO < k) {
    throw invalid_argument("flash_attention needs an fp32-safe scale and leading dimensions >= k");
  }
  if (ldQ % 8 != 0 || ldK % 8 != 0 || ldV % 8 != 0 ||
      reinterpret_cast<std::uintptr_t>(dQ) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(dK) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(dV) % 16 != 0) {
    throw invalid_argument("flash_attention's vector loads need 16-byte-aligned Q/K/V rows");
  }
  if (k == 64) {
    launch_attention<64>(m, n, dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, scale, causal);
  } else {
    launch_attention<128>(m, n, dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, scale, causal);
  }
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
