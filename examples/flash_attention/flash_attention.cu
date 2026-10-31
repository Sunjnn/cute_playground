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
using cute::cp_async_fence;
using cute::cp_async_wait;
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
using cute::SM80_CP_ASYNC_CACHEALWAYS;
using cute::Step;
using cute::Stride;
using cute::Tensor;
using cute::Tile;
using cute::uint128_t;
using cute::X;
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

// No score staging: the QK accumulator stays in registers and the online softmax reduces it there
// (see the loop). Q is staged once for the CTA; K and V are single buffers overwritten by the next
// tile as soon as the current one has consumed them, with cp.async prefetch overlapping the
// transfer with the softmax and PV that run in between.
template <int kHeadDim> constexpr std::size_t smem_bytes() {
  constexpr std::size_t kKTiles = kHeadDim / kBlockK;
  return ((static_cast<std::size_t>(kBlockM) * kBlockK * kKTiles) +
          (static_cast<std::size_t>(kBlockN) * kBlockK * kKTiles) +
          static_cast<std::size_t>(kBlockM) * kBlockN +
          static_cast<std::size_t>(kBlockN) * kHeadDim) *
         sizeof(half_t);
}

static_assert(smem_bytes<128>() <= 48ULL * 1024,
              "the register-softmax tile should stay under the default shared-memory limit");

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
  constexpr int kKTiles = kHeadDim / kBlockK;
  extern __shared__ __align__(16) char sRaw[];
  __shared__ float sRowMax[kBlockM];
  __shared__ float sRowSum[kBlockM];
  __shared__ float sRowScale[kBlockM];
  // Per-tile, per-N-atom partials for the register reduction: four lanes of a warp share a row,
  // and the MMA splits each row across kMmaAtomsN warps, so the row reduction is a shuffle plus a
  // four-entry flush through shared memory.
  __shared__ float sTileMax[kBlockM][kMmaAtomsN];
  __shared__ float sTileSum[kBlockM][kMmaAtomsN];

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
  auto *sKPtr = sQPtr + static_cast<std::size_t>(kBlockM) * kBlockK * kKTiles;
  auto *sPPtr = sKPtr + static_cast<std::size_t>(kBlockN) * kBlockK * kKTiles;
  auto *sVPtr = sPPtr + static_cast<std::size_t>(kBlockM) * kBlockN;
  const Tensor sQ = make_tensor(make_smem_ptr(sQPtr), make_k_major_smem_layout<kKTiles>(bM, bK));
  const Tensor sK = make_tensor(make_smem_ptr(sKPtr), make_k_major_smem_layout<kKTiles>(bN, bK));
  const Tensor sP =
      make_tensor(make_smem_ptr(sPPtr), make_layout(make_shape(bM, bN), make_stride(bN, _1{})));
  const Tensor sV =
      make_tensor(make_smem_ptr(sVPtr), make_layout(make_shape(bN, bD), make_stride(bD, _1{})));
  // MMA expresses C(M,N) = A(M,K) B(N,K)^T. For PV its reduction K is the key axis:
  // B(feature,key) must be V(key,feature), a transpose VIEW of the actual row-major V staging.
  // V is deliberately unswizzled here, so stride (1, kHeadDim) is precisely that transpose.
  const Tensor sVt =
      make_tensor(make_smem_ptr(sVPtr), make_layout(make_shape(bD, bN), make_stride(_1{}, bD)));

  // cp.async, not UniversalCopy: the key/value prefetches issued between phases only overlap if
  // the copy does not block the issuing warp.
  auto tiledCopy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  auto tiledMma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<Int<kMmaAtomsM>, Int<kMmaAtomsN>>>{},
      Tile<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>{});
  const auto thrMma = tiledMma.get_slice(threadIdx.x);
  const auto thrCopy = decltype(tiledCopy)::get_slice(threadIdx.x);
  const Tensor tAgA = thrCopy.partition_S(gQ); // (CPY,CPY_M,CPY_K,kTiles)
  const Tensor tAsQ = thrCopy.partition_D(sQ); // (CPY,CPY_M,CPY_K,kTiles)
  const Tensor tBsK = thrCopy.partition_D(sK); // (CPY,CPY_N,CPY_K,kTiles)
  const Tensor tVgV = thrCopy.partition_S(gV);
  Tensor tVsV = thrCopy.partition_D(sV);
  Tensor tOrO = thrMma.partition_fragment_C(gO);
  Tensor tOgO = thrMma.partition_C(gO);
  const Tensor tOcO = thrMma.partition_C(make_identity_tensor(make_shape(bM, bD)));

  // The score accumulator is registers only. It is partitioned from a coordinate tensor, so the
  // C-layout (and therefore the row/column mapping below) is exactly the MMA's; the pointer is
  // never dereferenced.
  const Tensor tCcS = thrMma.partition_C(make_identity_tensor(make_shape(bM, bN)));
  Tensor tSrS = thrMma.partition_fragment_C(
      make_tensor(make_smem_ptr(reinterpret_cast<float *>(sRaw)),
                  make_layout(make_shape(bM, bN), make_stride(bN, _1{}))));

  // Q and K are partitioned over the whole head dimension with the stage mode kept, so the QK
  // mainloop is one uninterrupted run of ldmatrix + MMA per key tile.
  auto tCrA = thrMma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K,kTiles)
  auto tCrB = thrMma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K,kTiles)
  const auto s2rAtomA = Copy_Atom<cute::SM75_U32x4_LDSM_N, half_t>{};
  const auto s2rAtomB = Copy_Atom<cute::SM75_U32x2_LDSM_N, half_t>{};
  const auto s2rCopyA = make_tiled_copy_A(s2rAtomA, tiledMma);
  const auto s2rCopyB = make_tiled_copy_B(s2rAtomB, tiledMma);
  const auto thrS2rA = s2rCopyA.get_slice(threadIdx.x);
  const auto thrS2rB = s2rCopyB.get_slice(threadIdx.x);
  auto tXsA = thrS2rA.partition_S(sQ); // (CPY,MMA_M,MMA_K,kTiles)
  auto tXsB = thrS2rB.partition_S(sK); // (CPY,MMA_N,MMA_K,kTiles)
  auto tXrA = thrS2rA.retile_D(tCrA);  // (CPY,MMA_M,MMA_K,kTiles)
  auto tXrB = thrS2rB.retile_D(tCrB);  // (CPY,MMA_N,MMA_K,kTiles)

  // P and V are repartitioned as MMA OPERANDS. The QK accumulator is neither the correct shape
  // nor the correct register order for this. Generic copies preserve the logical coordinates;
  // QK's K-major ldmatrix B atom cannot simply be reused for the N-major V transpose.
  const Tensor tPsP = thrMma.partition_A(sP);
  const Tensor tVsVt = thrMma.partition_B(sVt);
  Tensor tPrP = thrMma.partition_fragment_A(sP);
  Tensor tVrV = thrMma.partition_fragment_B(sVt);
  static_assert(size(tPsP) == size(tPrP));
  static_assert(size(tVsVt) == size(tVrV));

  const auto warpId = threadId / 32;
  const auto laneId = threadId % 32;
  const auto nAtom = warpId / kMmaAtomsM;
  // The two rows this thread owns in the C fragment: element 0 is the first row, element 2 (after
  // the two adjacent columns of the first) the second.
  const auto rowA = static_cast<int>(cute::get<0>(tCcS(0)));
  const auto rowB = static_cast<int>(cute::get<0>(tCcS(2)));

  cute::clear(tOrO);
  for (auto row = threadId; row < kBlockM; row += kThreads) {
    sRowMax[row] = -INFINITY;
    sRowSum[row] = 0.0f;
    sRowScale[row] = 0.0f;
  }
  __syncthreads();

  const int nTiles = size<2>(gK);
  const auto lastVisible = causal ? rowBase + kBlockM - 1 + n - m : n - 1;
  const auto activeTiles = min(nTiles, lastVisible / kBlockN + 1);

  // Prime the pipeline: Q, key tile 0 and value tile 0 in one commit group.
  CUTE_UNROLL
  for (auto kt = 0; kt < kKTiles; ++kt) {
    cute::copy(tiledCopy, tAgA(_, _, _, kt), tAsQ(_, _, _, kt));
  }
  {
    const auto tBgFirst = thrCopy.partition_S(gK(_, _, 0, _));
    CUTE_UNROLL
    for (auto kt = 0; kt < kKTiles; ++kt) {
      cute::copy(tiledCopy, tBgFirst(_, _, _, kt), tBsK(_, _, _, kt));
    }
  }
  cute::copy(tiledCopy, tVgV(_, _, _, 0), tVsV);
  cute::cp_async_fence();

  // Group retirement schedule (two groups can be in flight):
  //   start of tile t : K(t) landed, V(t) may still be in flight -> wait<1> (t > 0)
  //   before PV of t  : V(t) landed, K(t + 1) may stay in flight -> wait<1>
  // Tile 0 starts from the single priming group, the last tile waits for everything.
  for (auto tile = 0; tile < activeTiles; ++tile) {
    if (tile == 0) {
      cute::cp_async_wait<0>();
    } else {
      cute::cp_async_wait<1>();
    }
    __syncthreads();

    cute::clear(tSrS);
    CUTE_UNROLL
    for (auto kt = 0; kt < kKTiles; ++kt) {
      CUTE_UNROLL
      for (auto kb = 0; kb < size<2>(tCrA); ++kb) {
        cute::copy(s2rAtomA, tXsA(_, _, kb, kt), tXrA(_, _, kb, kt));
        cute::copy(s2rAtomB, tXsB(_, _, kb, kt), tXrB(_, _, kb, kt));
        cute::gemm(tiledMma, tCrA(_, _, kb, kt), tCrB(_, _, kb, kt), tSrS);
      }
    }

    // ---- online softmax straight from the accumulator, no 8 KB score staging ----
    // Scale and mask in place, reducing each thread's rows to a per-warp maximum.
    auto maxA = -INFINITY;
    auto maxB = -INFINITY;
    CUTE_UNROLL
    for (auto i = 0; i < size(tSrS); ++i) {
      const auto row = static_cast<int>(cute::get<0>(tCcS(i)));
      auto value = tSrS(i) * scale;
      if (causal && tile * kBlockN + static_cast<int>(cute::get<1>(tCcS(i))) > rowBase + row + n - m) {
        value = -INFINITY;
      }
      tSrS(i) = value;
      if (row == rowA) {
        maxA = fmaxf(maxA, value);
      } else {
        maxB = fmaxf(maxB, value);
      }
    }
#pragma unroll
    for (auto offset = 1; offset < 4; offset *= 2) {
      maxA = fmaxf(maxA, __shfl_xor_sync(0xffffffffu, maxA, offset));
      maxB = fmaxf(maxB, __shfl_xor_sync(0xffffffffu, maxB, offset));
    }
    if (laneId % 4 == 0) {
      sTileMax[rowA][nAtom] = maxA;
      sTileMax[rowB][nAtom] = maxB;
    }
    // This barrier also proves every warp has finished reading sK, so the prefetch may overwrite it.
    __syncthreads();

    if (tile + 1 < activeTiles) {
      const auto tBgNext = thrCopy.partition_S(gK(_, _, tile + 1, _));
      CUTE_UNROLL
      for (auto kt = 0; kt < kKTiles; ++kt) {
        cute::copy(tiledCopy, tBgNext(_, _, _, kt), tBsK(_, _, _, kt));
      }
      cute::cp_async_fence();
    }

    // Combine the four N-atom partials and fold in the running maximum.
    auto tileMaxA = sTileMax[rowA][0];
    auto tileMaxB = sTileMax[rowB][0];
    CUTE_UNROLL
    for (auto part = 1; part < kMmaAtomsN; ++part) {
      tileMaxA = fmaxf(tileMaxA, sTileMax[rowA][part]);
      tileMaxB = fmaxf(tileMaxB, sTileMax[rowB][part]);
    }
    const auto oldMaxA = sRowMax[rowA];
    const auto oldMaxB = sRowMax[rowB];
    const auto newMaxA = fmaxf(oldMaxA, tileMaxA);
    const auto newMaxB = fmaxf(oldMaxB, tileMaxB);
    // An empty tile contributes no probabilities. An empty old state contributes no numerator;
    // neither needs the undefined exp(-inf - -inf). Causal m <= n makes the first tile nonempty.
    const auto correctionA = oldMaxA == -INFINITY ? 0.0f : __expf(oldMaxA - newMaxA);
    const auto correctionB = oldMaxB == -INFINITY ? 0.0f : __expf(oldMaxB - newMaxB);

    // Exponentiate, stage fp16 probabilities in shared memory (PV's A operand) and sum per row.
    auto sumA = 0.0f;
    auto sumB = 0.0f;
    CUTE_UNROLL
    for (auto i = 0; i < size(tSrS); i += 2) {
      const auto row = static_cast<int>(cute::get<0>(tCcS(i)));
      const auto col = static_cast<int>(cute::get<1>(tCcS(i)));
      const auto newMax = row == rowA ? newMaxA : newMaxB;
      const auto p0 = tSrS(i) == -INFINITY ? 0.0f : __expf(tSrS(i) - newMax);
      const auto p1 = tSrS(i + 1) == -INFINITY ? 0.0f : __expf(tSrS(i + 1) - newMax);
      tSrS(i) = p0;
      tSrS(i + 1) = p1;
      if (row == rowA) {
        sumA += p0 + p1;
      } else {
        sumB += p0 + p1;
      }
      // The two elements of an MMA atom are adjacent columns of one row.
      *reinterpret_cast<__half2 *>(sPPtr + row * kBlockN + col) = __floats2half2_rn(p0, p1);
    }
#pragma unroll
    for (auto offset = 1; offset < 4; offset *= 2) {
      sumA += __shfl_xor_sync(0xffffffffu, sumA, offset);
      sumB += __shfl_xor_sync(0xffffffffu, sumB, offset);
    }
    if (laneId % 4 == 0) {
      sTileSum[rowA][nAtom] = sumA;
      sTileSum[rowB][nAtom] = sumB;
    }
    __syncthreads();

    auto sumTotalA = sTileSum[rowA][0];
    auto sumTotalB = sTileSum[rowB][0];
    CUTE_UNROLL
    for (auto part = 1; part < kMmaAtomsN; ++part) {
      sumTotalA += sTileSum[rowA][part];
      sumTotalB += sTileSum[rowB][part];
    }
    // Only one N atom per row writes the state; the other three would store identical values but
    // race on the read-modify-write of rowSum.
    if (nAtom == 0 && laneId % 4 == 0) {
      sRowMax[rowA] = newMaxA;
      sRowMax[rowB] = newMaxB;
      sRowSum[rowA] = sRowSum[rowA] * correctionA + sumTotalA;
      sRowSum[rowB] = sRowSum[rowB] * correctionB + sumTotalB;
      sRowScale[rowA] = correctionA;
      sRowScale[rowB] = correctionB;
    }

    if (tile + 1 == activeTiles) {
      cute::cp_async_wait<0>();
    } else if (tile > 0) {
      cute::cp_async_wait<1>();
    }
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

    if (tile + 1 < activeTiles) {
      cute::copy(tiledCopy, tVgV(_, _, _, tile + 1), tVsV);
      cute::cp_async_fence();
    }
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
  constexpr auto kSmemBytes = smem_bytes<kHeadDim>();
  flash_attention_device<kHeadDim><<<m / kBlockM, kThreads, kSmemBytes>>>(
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
