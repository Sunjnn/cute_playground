#include "flash_attention_tma.cuh"

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cute/tensor.hpp" // IWYU pragma: keep

#include "../fused_softmax/fused_softmax_mainloop.cuh"
#include "attention_common.cuh"

// The key tile is streamed with TMA: one 64x64 fp16 box per 64-wide K slice, written straight
// into the swizzled shared tile. CUDA's 128-byte TMA swizzle for a 64-half row is exactly CuTe's
// Swizzle<3,3,3> over this atom (verified bit-for-bit), so no layout or read-side change is
// needed. Only thread 0 issues the copy, so every other thread skips the per-tile copy
// instructions and their address arithmetic.
__device__ __forceinline__ void mbarrier_init(unsigned long long *barrier, int count) {
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(unsigned long long *barrier, int bytes) {
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(address), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(unsigned long long *barrier, int parity) {
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
  asm volatile("{\n\t.reg .pred P;\n"
               "WAIT%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
               "@!P bra WAIT%=;\n\t}\n" ::"r"(address),
               "r"(parity)
               : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    void *smem, const CUtensorMap *map, int coord0, int coord1, unsigned long long *barrier) {
  const auto dest = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const auto bar = static_cast<unsigned>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dest),
      "l"(map), "r"(coord0), "r"(coord1), "r"(bar)
      : "memory");
}

// A tensor map has to live at a 64-byte aligned address; a plain by-value kernel parameter is
// only guaranteed its natural alignment, so the descriptor travels in an explicit wrapper.
struct alignas(64) KernelTmaMap {
  CUtensorMap desc;
};


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
// Eight halves of padding on the P and V rows. Both are read by 16-byte row slices (P by the
// generic PV copy, V by ldmatrix.trans); without the pad consecutive rows are a multiple of the
// 32-bank stride apart and every slice of a warp lands on the same banks. A pad of 8 halves moves
// each row by four banks, so eight rows cover all 32 banks exactly once.
constexpr int kPad = 8;

// TMA writes a tile as plain rows of 64 halves, XOR-swizzling each 16-byte unit by its row index
// modulo 8. That is this atom -- eight row-major rows composed with Swizzle<3,3,3> -- tiled over
// the stages, which is not the 8x8-blocked atom of fused_softmax_mainloop.cuh. The ldmatrix B
// partition reads it through the same logical coordinates, so only the layout, not the MMA, differs.
template <int kStages, class BShape, class KShape>
CUTE_HOST_DEVICE constexpr auto make_tma_k_smem_layout(BShape const &bN, KShape const &bK) {
  return cute::tile_to_shape(
      cute::composition(cute::Swizzle<3, 3, 3>{},
                        cute::Layout<cute::Shape<cute::_8, cute::_64>,
                                     cute::Stride<cute::_64, cute::_1>>{}),
      cute::make_shape(bN, bK, cute::Int<kStages>{}));
}

// TMA writes the key tile with the 128-byte swizzle evaluated on the *shared-window address*, so
// the destination has to sit on a 1024-byte boundary (the swizzle repeats every 8 rows x 128 B).
// The extra 1024 bytes let the kernel round the key tile up past Q without moving the other
// buffers; every stage is 8 KiB apart and therefore stays aligned.
constexpr int kTmaAlignPad = 1024;

// No score staging: the QK accumulator stays in registers and the online softmax reduces it there
// (see the loop). Q is staged once for the CTA; K and V are single buffers overwritten by the next
// tile as soon as the current one has consumed them, with cp.async prefetch overlapping the
// transfer with the softmax and PV that run in between.
template <int kHeadDim> constexpr std::size_t smem_bytes() {
  constexpr std::size_t kKTiles = kHeadDim / kBlockK;
  return kTmaAlignPad + ((static_cast<std::size_t>(kBlockM) * kBlockK * kKTiles) +
                         (static_cast<std::size_t>(kBlockN) * kBlockK * kKTiles) +
                         static_cast<std::size_t>(kBlockM) * (kBlockN + kPad) +
                         static_cast<std::size_t>(kBlockN) * (kHeadDim + kPad)) *
                            sizeof(half_t);
}

static_assert(smem_bytes<128>() <= 48ULL * 1024,
              "the register-softmax tile should stay under the default shared-memory limit");

template <int kHeadDim>
__global__ __launch_bounds__(kThreads) void flash_attention_tma_device(
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
    bool causal,
    const __grid_constant__ KernelTmaMap kMap) {
  constexpr int kKTiles = kHeadDim / kBlockK;
  constexpr int kTileBytes = kBlockN * kBlockK * static_cast<int>(sizeof(half_t)) * kKTiles;
  // TMA with the 128-byte swizzle requires its shared destination to be 128-byte aligned, and the
  // K stages sit at 128-byte multiples inside this buffer.
  extern __shared__ __align__(128) char sRaw[];
  __shared__ float sRowMax[kBlockM];
  __shared__ float sRowSum[kBlockM];
  // Per-tile, per-N-atom partials for the register reduction: four lanes of a warp share a row,
  // and the MMA splits each row across kMmaAtomsN warps, so the row reduction is a shuffle plus a
  // four-entry flush through shared memory.
  __shared__ float sTileMax[kBlockM][kMmaAtomsN];
  __shared__ float sTileSum[kBlockM][kMmaAtomsN];
  __shared__ alignas(8) unsigned long long sKBarrier;

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
  const auto sQEnd = reinterpret_cast<std::uintptr_t>(sRaw) +
                     static_cast<std::uintptr_t>(kBlockM) * kBlockK * kKTiles * sizeof(half_t);
  auto *sKPtr = reinterpret_cast<half_t *>((sQEnd + kTmaAlignPad - 1) &
                                           ~static_cast<std::uintptr_t>(kTmaAlignPad - 1));
  auto *sPPtr = sKPtr + static_cast<std::size_t>(kBlockN) * kBlockK * kKTiles;
  auto *sVPtr = sPPtr + static_cast<std::size_t>(kBlockM) * (kBlockN + kPad);
  const Tensor sQ = make_tensor(make_smem_ptr(sQPtr), make_k_major_smem_layout<kKTiles>(bM, bK));
  const Tensor sK = make_tensor(make_smem_ptr(sKPtr), make_tma_k_smem_layout<kKTiles>(bN, bK));
  const Tensor sP = make_tensor(make_smem_ptr(sPPtr),
                                make_layout(make_shape(bM, bN), make_stride(bN + kPad, _1{})));
  const Tensor sV = make_tensor(make_smem_ptr(sVPtr),
                                make_layout(make_shape(bN, bD), make_stride(bD + kPad, _1{})));
  // MMA expresses C(M,N) = A(M,K) B(N,K)^T. For PV its reduction K is the key axis:
  // B(feature,key) must be V(key,feature), a transpose VIEW of the actual row-major V staging.
  // V is deliberately unswizzled here, so stride (1, kHeadDim + kPad) is precisely that transpose;
  // the pad is what keeps ldmatrix.trans's row slices off a single bank per warp.
  const Tensor sVt = make_tensor(
      make_smem_ptr(sVPtr),
      make_layout(make_shape(bD, bN), make_stride(_1{}, Int<kHeadDim + kPad>{})));

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
  // nor the correct register order for this. P is consumed through the generic copy (its A
  // operand is written row-major by the softmax). V's logical B operand is the transpose of the
  // row-major V staging, which is exactly what ldmatrix.trans reads: SM75_U16x4_LDSM_T turns the
  // 16 32-bit shared loads per thread per tile into four transposed ldmatrix instructions.
  const Tensor tPsP = thrMma.partition_A(sP);
  const Tensor tVsVt = thrMma.partition_B(sVt);
  Tensor tPrP = thrMma.partition_fragment_A(sP);
  Tensor tVrV = thrMma.partition_fragment_B(sVt);
  static_assert(size(tPsP) == size(tPrP));
  static_assert(size(tVsVt) == size(tVrV));
  const auto s2rAtomVT = Copy_Atom<cute::SM75_U16x4_LDSM_T, half_t>{};
  const auto s2rCopyVT = make_tiled_copy_B(s2rAtomVT, tiledMma);
  const auto thrS2rVT = s2rCopyVT.get_slice(threadIdx.x);
  auto tXsVT = thrS2rVT.partition_S(sVt); // (CPY,MMA_N,MMA_K)
  auto tXrVT = thrS2rVT.retile_D(tVrV);   // (CPY,MMA_N,MMA_K)

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
  }
  __syncthreads();

  const int nTiles = size<2>(gK);
  const auto lastVisible = causal ? rowBase + kBlockM - 1 + n - m : n - 1;
  const auto activeTiles = min(nTiles, lastVisible / kBlockN + 1);

  // Prime the pipeline: Q and V(0) through cp.async, K(0) through TMA. The mbarrier is initialised
  // once, by the same thread that will arrive on it, and thread 0 stays the single TMA producer.
  if (threadId == 0) {
    mbarrier_init(&sKBarrier, 1);
  }
  __syncthreads();
  CUTE_UNROLL
  for (auto kt = 0; kt < kKTiles; ++kt) {
    cute::copy(tiledCopy, tAgA(_, _, _, kt), tAsQ(_, _, _, kt));
  }
  cute::copy(tiledCopy, tVgV(_, _, _, 0), tVsV);
  cute::cp_async_fence();
  if (threadId == 0) {
    mbarrier_expect_tx(&sKBarrier, kTileBytes);
    CUTE_UNROLL
    for (auto kt = 0; kt < kKTiles; ++kt) {
      tma_load_2d(sKPtr + static_cast<std::size_t>(kt) * kBlockN * kBlockK, &kMap.desc, kt * kBlockK, 0,
                  &sKBarrier);
    }
  }

  // K arrives through the mbarrier, which every thread waits on and which therefore doubles as the
  // CTA-wide arrival point for the key tile. V stays on cp.async, one commit group per tile.
  int kPhase = 0;
  for (auto tile = 0; tile < activeTiles; ++tile) {
    if (tile == 0) {
      cute::cp_async_wait<0>(); // Q
    }
    mbarrier_wait_parity(&sKBarrier, kPhase);
    kPhase ^= 1;
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

    if (tile + 1 < activeTiles && threadId == 0) {
      mbarrier_expect_tx(&sKBarrier, kTileBytes);
      CUTE_UNROLL
      for (auto kt = 0; kt < kKTiles; ++kt) {
        tma_load_2d(sKPtr + static_cast<std::size_t>(kt) * kBlockN * kBlockK, &kMap.desc, kt * kBlockK,
                    (tile + 1) * kBlockN, &sKBarrier);
      }
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
      *reinterpret_cast<__half2 *>(sPPtr + row * (kBlockN + kPad) + col) =
          __floats2half2_rn(p0, p1);
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
    // V(t) is the only cp.async group still in flight, so retiring everything here is exact.
    cute::cp_async_wait<0>();
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
    }

    // Each thread still holds the correction for the two rows it owns, so rescaling O needs no
    // shared-memory read and no extra barrier after the per-row state is stored above.
    CUTE_UNROLL
    for (auto i = 0; i < size(tOrO); ++i) {
      const auto row = static_cast<int>(cute::get<0>(tOcO(i)));
      tOrO(i) *= row == rowA ? correctionA : correctionB;
    }
    CUTE_UNROLL
    for (auto kb = 0; kb < size<2>(tPrP); ++kb) {
      cute::copy(tPsP(_, _, kb), tPrP(_, _, kb));
      cute::copy(s2rAtomVT, tXsVT(_, _, kb), tXrVT(_, _, kb));
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

// One 2D descriptor describes every key tile: a kBlockK-wide, kBlockN-row box of the row-major
// (n, k) matrix, copied with CUDA's 128-byte swizzle (bit-for-bit the shared layout the ldmatrix
// B atom expects). Descriptors depend only on the pointer, shape and leading dimension, so the
// harness's repeated calls with the same buffers reuse one encoding.
CUtensorMap encode_k_map(const __half *dK, int n, int k, int ldK) {
  CUtensorMap map{};
  // The driver API lists the innermost dimension first: dims = (k columns, n rows), and the
  // single stride is the byte distance between rows.
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(n)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(ldK) * sizeof(__half)};
  cuuint32_t box[2] = {static_cast<cuuint32_t>(kBlockK), static_cast<cuuint32_t>(kBlockN)};
  cuuint32_t elementStrides[2] = {1, 1};
  const auto result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<__half *>(dK), dims, strides, box,
      elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    throw runtime_error(
        "flash_attention_tma could not encode the K TMA descriptor (is a CUDA context current?)");
  }
  return map;
}

const CUtensorMap &cached_k_map(const __half *dK, int n, int k, int ldK) {
  struct Entry {
    const __half *ptr;
    int n;
    int k;
    int ld;
    CUtensorMap map;
  };
  static Entry entries[4];
  static int count = 0;
  for (auto i = 0; i < count; ++i) {
    const auto &entry = entries[i];
    if (entry.ptr == dK && entry.n == n && entry.k == k && entry.ld == ldK) {
      return entry.map;
    }
  }
  if (count == 4) {
    count = 0;
  }
  entries[count] = Entry{dK, n, k, ldK, encode_k_map(dK, n, k, ldK)};
  return entries[count++].map;
}

template <int kHeadDim>
void launch_attention_tma(
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
  const auto &kMap = cached_k_map(dK, n, kHeadDim, ldK);
  flash_attention_tma_device<kHeadDim><<<m / kBlockM, kThreads, kSmemBytes>>>(
      dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, m, n, scale, causal, KernelTmaMap{kMap});
}

} // namespace

void flash_attention_tma(
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
    throw invalid_argument("flash_attention_tma needs m % 32 = 0, n % 64 = 0 and k = 64 or 128");
  }
  if (causal && m > n) {
    throw invalid_argument("a causal mask needs m <= n");
  }
  if (!attention::valid_scale(k, scale) || ldQ < k || ldK < k || ldV < k || ldO < k) {
    throw invalid_argument("flash_attention_tma needs an fp32-safe scale and leading dimensions >= k");
  }
  if (ldQ % 8 != 0 || ldK % 8 != 0 || ldV % 8 != 0 ||
      reinterpret_cast<std::uintptr_t>(dQ) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(dK) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(dV) % 16 != 0) {
    throw invalid_argument("flash_attention_tma's vector loads need 16-byte-aligned Q/K/V rows");
  }
  if (k == 64) {
    launch_attention_tma<64>(m, n, dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, scale, causal);
  } else {
    launch_attention_tma<128>(m, n, dQ, ldQ, dK, ldK, dV, ldV, dO, ldO, scale, causal);
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
