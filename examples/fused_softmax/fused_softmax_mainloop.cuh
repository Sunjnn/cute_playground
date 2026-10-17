#ifndef EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_MAINLOOP_CUH
#define EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_MAINLOOP_CUH

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

#include "cute/tensor.hpp" // IWYU pragma: keep

// What the two fused kernels share. Both compute S = Q K^T with the same MMA configuration and
// both hand the accumulator to shared memory through the same scale-and-mask step; they differ only
// in what happens next, which is why the softmax itself is not here.
//
// The causal predicate is the part worth sharing. It is one comparison, but if the two kernels ever
// disagreed about it the harness would report a mismatch with no way to tell a wrong GEMM from a
// wrong mask, which is exactly the ambiguity Reference::scores exists to remove.
namespace fused_softmax {

// Tile configuration. kBlockM and kBlockK are common to both kernels; the column extent is not, and
// each kernel declares its own - holding a whole row on chip is what caps fused_softmax_regrow's,
// while fused_softmax_recompute streams columns and is bounded by n instead.
constexpr int kBlockM = 32;
constexpr int kBlockK = 64;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// The MMA is 2 x 4 atoms of 16x8x16, so it covers 32 x 32 x 16 per instruction group and needs
// exactly 2 * 4 * 32 = 256 threads - the same kThreads the copies are partitioned over.
constexpr int kMmaAtomsM = 2;
constexpr int kMmaAtomsN = 4;
constexpr int kMmaM = kMmaAtomsM * 16;
constexpr int kMmaN = kMmaAtomsN * 8;
constexpr int kMmaK = 16;

// The layout both kernels' operand tiles live in: a 128-bit-swizzled, K-major half tile. The
// swizzle is what makes the 16-byte vectors that ldmatrix reads - and that the global-to-shared
// copy writes - land in distinct banks. The atom is 8 rows x 64 halves, the shape SM75_U32x4_LDSM_N
// expects, tiled to the (bM, bK) of the CTA tile. Swizzle<3,3,3> XORs offset bits [3,6) with bits
// [6,9), so the low three bits are untouched: an 8-half (16-byte) vector stays contiguous and
// aligned, and only its position within a 128-byte group changes. That is why a plain 128-bit copy
// can write this layout.
template <class BShape, class KShape>
CUTE_HOST_DEVICE constexpr auto make_k_major_smem_layout(BShape const &bM, KShape const &bK) {
  return cute::tile_to_shape(
      cute::composition(
          cute::Swizzle<3, 3, 3>{},
          cute::Layout<
              cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
              cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}),
      cute::make_shape(bM, bK));
}

// The same tile with a pipeline stage appended: (bM, bK, kStages), where stage s starts at
// cosize(bM, bK) * s. That is the layout the cp.async ring in compute_tile_async() writes into.
template <int kStages, class BShape, class KShape>
CUTE_HOST_DEVICE constexpr auto make_k_major_smem_layout(BShape const &bM, KShape const &bK) {
  static_assert(kStages >= 1, "a ring needs at least one stage");
  return cute::tile_to_shape(
      cute::composition(
          cute::Swizzle<3, 3, 3>{},
          cute::Layout<
              cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
              cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}),
      cute::make_shape(bM, bK, cute::Int<kStages>{}));
}

// Scale, mask, and write the accumulator to shared memory in one pass. The identity tensor behind
// tCcC is what turns a flat fragment index into the (row, column) it belongs to, which is the only
// way to ask whether the causal mask bites at that particular element; rowBase and colBase place
// the tile in the full matrix.
template <class TCrC, class TCcC, class TCsScores>
__device__ void store_scores(
    const TCrC &tCrC,
    const TCcC &tCcC,
    TCsScores &tCsScores,
    int rowBase,
    int colBase,
    int m,
    int n,
    float scale,
    bool causal) {
  CUTE_UNROLL
  for (auto i = 0; i < cute::size(tCrC); ++i) {
    const auto row = rowBase + static_cast<int>(cute::get<0>(tCcC(i)));
    const auto col = colBase + static_cast<int>(cute::get<1>(tCcC(i)));
    auto value = tCrC(i) * scale;
    // Column j is visible iff j <= row + n - m, which keeps the last query aligned with the last
    // key when m < n. main.cu rejects m > n, so every row keeps at least column 0 and no row sum
    // downstream is a division by zero.
    if (causal && col > row + n - m) {
      value = -INFINITY;
    }
    tCsScores(i) = value;
  }
}

// Fold one kColsPerLane*32-column tile of scores into the running (max, sum) of each of its rows.
// Both arrays hold one float per row and live in shared memory; they are the online-softmax state,
// so a tile can be folded in as soon as it is computed and then thrown away.
//
// The correction factor is what makes that legal: rescaling the running sum into the new maximum's
// frame costs one exp per row per tile, and it is why this needs no second look at a tile it has
// already discarded. On the first tile rowMax is -INFINITY and __expf(-INFINITY - finite) is
// exactly 0, so the running sum starts from nothing with no special case. A fully masked tile
// leaves tileMax at -INFINITY, so rowMax does not move, the correction is exp(0) = 1, and the tile
// contributes 0 to the sum.
//
// The one state this cannot survive is a row with no visible column at all, where both maxima stay
// -INFINITY and the correction becomes exp(-INFINITY + INFINITY) = NaN. main.cu rejects m > n under
// a causal mask, and without one every column is visible, so every row sees column 0 in the first
// tile and rowMax is finite from then on.
template <int kColsPerLane>
__device__ void online_row_stats(const float *scores, float *rowMax, float *rowSum) {
  static_assert(kColsPerLane % 2 == 0, "the store goes out as __half2 pairs");
  constexpr int kCols = kColsPerLane * 32;

  const auto warpId = static_cast<int>(threadIdx.x) / 32;
  const auto laneId = static_cast<int>(threadIdx.x) % 32;
  const auto laneCol = laneId * kColsPerLane;

  for (auto r = warpId; r < kBlockM; r += kWarps) {
    const auto *row = scores + static_cast<std::int64_t>(r) * kCols + laneCol;

    float values[kColsPerLane];
    CUTE_UNROLL
    for (auto c = 0; c < kColsPerLane; ++c) {
      values[c] = row[c];
    }

    auto tileMax = -INFINITY;
    CUTE_UNROLL
    for (const auto value : values) {
      tileMax = fmaxf(tileMax, value);
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, offset));
    }

    const auto oldMax = rowMax[r];
    const auto newMax = fmaxf(oldMax, tileMax);
    const auto correction = __expf(oldMax - newMax);

    auto tileSum = 0.0f;
    CUTE_UNROLL
    for (const auto value : values) {
      tileSum += __expf(value - newMax);
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      tileSum += __shfl_xor_sync(0xffffffffu, tileSum, offset);
    }

    rowMax[r] = newMax;
    rowSum[r] = rowSum[r] * correction + tileSum;
  }
}

// Write one tile of scores out as probabilities, given the row statistics every tile has already
// contributed to. Each lane owns kColsPerLane contiguous columns, which is what lets the store go
// out as __half2 pairs.
template <int kColsPerLane>
__device__ void normalize_rows(
    const float *scores,
    __half *dP,
    int ldP,
    int rowBase,
    int colBase,
    const float *rowMax,
    const float *rowSum) {
  static_assert(kColsPerLane % 2 == 0, "the store goes out as __half2 pairs");
  constexpr int kCols = kColsPerLane * 32;

  const auto warpId = static_cast<int>(threadIdx.x) / 32;
  const auto laneId = static_cast<int>(threadIdx.x) % 32;
  const auto laneCol = laneId * kColsPerLane;

  for (auto r = warpId; r < kBlockM; r += kWarps) {
    const auto *row = scores + static_cast<std::int64_t>(r) * kCols + laneCol;
    auto *out = reinterpret_cast<__half2 *>(
        dP + static_cast<std::int64_t>(rowBase + r) * ldP + colBase + laneCol);

    // One reciprocal and one maximum per row rather than per element. A masked column carries
    // -INFINITY, which __expf() turns into exactly 0, so it needs no special case here.
    const auto invSum = 1.0f / rowSum[r];
    const auto max = rowMax[r];

    float values[kColsPerLane];
    CUTE_UNROLL
    for (auto c = 0; c < kColsPerLane; ++c) {
      values[c] = __expf(row[c] - max) * invSum;
    }
    CUTE_UNROLL
    for (auto c = 0; c < kColsPerLane; c += 2) {
      out[c / 2] = __floats2half2_rn(values[c], values[c + 1]);
    }
  }
}

// Write zeros to one kBlockM x kBlockN tile of P. A fully masked causal tile still has to reach the
// output, and zero is exactly what normalize_rows() would have stored for it: its columns arrive
// from store_scores() as -INFINITY, __expf(-INFINITY - max) is exactly 0, and 0 * invSum is 0.
// Skipping the tile's compute is therefore unobservable, but skipping its store would leave the NaN
// the harness pre-fills.
template <int kBlockN> __device__ void zero_tile(__half *dP, int ldP, int rowBase, int colBase) {
  static_assert(kBlockN % 2 == 0, "the store goes out as __half2 pairs");
  constexpr int kHalf2PerRow = kBlockN / 2;
  static_assert(kThreads % kHalf2PerRow == 0, "one step has to cover whole rows");
  constexpr int kRowsPerStep = kThreads / kHalf2PerRow;

  const auto threadId = static_cast<int>(threadIdx.x);
  const auto laneCol = (threadId % kHalf2PerRow) * 2;
  const auto zero = __floats2half2_rn(0.0f, 0.0f);
  for (auto r = threadId / kHalf2PerRow; r < kBlockM; r += kRowsPerStep) {
    *reinterpret_cast<__half2 *>(
        dP + static_cast<std::int64_t>(rowBase + r) * ldP + colBase + laneCol) = zero;
  }
}

// One GEMM over the whole of K for a single (row tile, column tile), leaving the scaled-nothing,
// masked-nothing accumulator in tCrC. gB is expected to be already sliced to one column tile, which
// is what makes this indifferent to whether the caller owns the whole row (fused_softmax_regrow) or
// streams it (fused_softmax_recompute).
//
// The partitioning is redone on every call rather than hoisted into the caller. It is layout
// algebra over compile-time shapes plus threadIdx.x, so ptxas folds it into the addressing it was
// going to compute anyway, and keeping it here is what holds the signature to eight parameters
// instead of thirteen.
template <
    class TTiledCopy,
    class TTiledMma,
    class TGmemA,
    class TSmemA,
    class TGmemB,
    class TSmemB,
    class TAcc>
__device__ void compute_tile(
    const TTiledCopy &tiledCopy,
    const TTiledMma &tiledMma,
    const TGmemA &gA,
    const TSmemA &sA,
    const TGmemB &gB,
    const TSmemB &sB,
    TAcc &tCrC,
    int kTiles) {
  // A using-declaration rather than a file-scope one: this is a header, and leaking cute's
  // placeholder into every includer is not this function's business.
  using cute::_;

  // get_slice is static on TiledCopy but not on TiledMMA, so the two calls below look different,
  // and the static one has to go through the template parameter: decltype() of a reference-typed
  // parameter keeps the reference, which has no members to name.
  const auto thrCopy = TTiledCopy::get_slice(threadIdx.x);
  auto tAgA = thrCopy.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  auto tAsA = thrCopy.partition_D(sA); // (CPY,CPY_M,CPY_K)
  auto tBgB = thrCopy.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  auto tBsB = thrCopy.partition_D(sB); // (CPY,CPY_N,CPY_K)

  const auto thrMma = tiledMma.get_slice(threadIdx.x);
  auto tCrA = thrMma.partition_fragment_A(sA); // (MMA,MMA_M,MMA_K)
  auto tCrB = thrMma.partition_fragment_B(sB); // (MMA,MMA_N,MMA_K)

  // The shared-to-register operand fetches go through ldmatrix: one instruction moves a group of
  // 8 x 8 x 16-bit smem tiles per warp, where an elementwise copy issues one scalar load per
  // element. That is why the smem layouts above are swizzled - ldmatrix reads 16-byte rows per
  // lane, and the swizzle is what keeps those rows in distinct banks across the warp.
  //
  // A takes x4 (8 halves per lane per fetch) because this MMA owns 8 halves of A per k-step; B
  // takes x2 (4 halves) because its 2 x 4 atom layout already spans the full 32-column tile,
  // leaving each lane only the 4 halves of its own atom. Asking x4 of B is a static error: the
  // tiled copy would have fewer values than the atom moves.
  const auto s2rAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::half_t>{};
  const auto s2rAtomB = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, cute::half_t>{};
  const auto s2rCopyA = cute::make_tiled_copy_A(s2rAtomA, tiledMma);
  const auto s2rCopyB = cute::make_tiled_copy_B(s2rAtomB, tiledMma);
  const auto thrS2rA = s2rCopyA.get_slice(threadIdx.x);
  const auto thrS2rB = s2rCopyB.get_slice(threadIdx.x);
  auto tXsA = thrS2rA.partition_S(sA); // (CPY,MMA_M,MMA_K)
  auto tXsB = thrS2rB.partition_S(sB); // (CPY,MMA_N,MMA_K)
  auto tXrA = thrS2rA.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)
  auto tXrB = thrS2rB.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

  cute::clear(tCrC);
  for (auto kt = 0; kt < kTiles; ++kt) {
    // Synchronous on purpose: this is the version that is easy to see is correct. A cp.async ring
    // replaces these two copies and the barriers around them.
    cute::copy(tiledCopy, tAgA(_, _, _, kt), tAsA);
    cute::copy(tiledCopy, tBgB(_, _, _, kt), tBsB);
    __syncthreads();

    CUTE_UNROLL
    for (auto kb = 0; kb < cute::size<2>(tCrA); ++kb) {
      cute::copy(s2rAtomA, tXsA(_, _, kb), tXrA(_, _, kb));
      cute::copy(s2rAtomB, tXsB(_, _, kb), tXrB(_, _, kb));
      cute::gemm(tiledMma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }
    __syncthreads();
  }
}

// The cp.async form of compute_tile(): the same tile, staged through a ring of kStages shared
// buffers, so the global-to-shared copy of tile kt + kStages can overlap the ldmatrix and MMA of
// tile kt. kStages = 1 is the degenerate ring: it still uses cp.async, but only issues the next
// tile's copy after the current one has been consumed. That is what fused_softmax_recompute uses
// today - a two-stage ring costs a CTA of residency at both column widths, which measured worse
// than the overlap was worth - but the ring is kept because the answer depends on the tile.
//
// Each tile is its own cp.async commit group, so the wait before tile kt retires exactly that tile:
// with a full ring, leaving kStages - 1 groups in flight proves the oldest one has landed. The last
// tile has no successor to leave in flight, so it waits for all of them.
template <
    int kStages,
    class TTiledCopy,
    class TTiledMma,
    class TGmemA,
    class TSmemA,
    class TGmemB,
    class TSmemB,
    class TAcc>
__device__ void compute_tile_async(
    const TTiledCopy &tiledCopy,
    const TTiledMma &tiledMma,
    const TGmemA &gA,
    const TSmemA &sA,
    const TGmemB &gB,
    const TSmemB &sB,
    TAcc &tCrC,
    int kTiles) {
  using cute::_;
  static_assert(kStages >= 1, "a ring needs at least one stage");

  const auto thrCopy = TTiledCopy::get_slice(threadIdx.x);
  auto tAgA = thrCopy.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  auto tAsA = thrCopy.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)
  auto tBgB = thrCopy.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  auto tBsB = thrCopy.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

  const auto thrMma = tiledMma.get_slice(threadIdx.x);
  // The fragment is per-stage, so it is partitioned from a single stage's slice; the pipelined
  // operand tensors below keep the PIPE mode.
  auto tCrA = thrMma.partition_fragment_A(sA(_, _, cute::Int<0>{})); // (MMA,MMA_M,MMA_K)
  auto tCrB = thrMma.partition_fragment_B(sB(_, _, cute::Int<0>{})); // (MMA,MMA_N,MMA_K)

  // Same ldmatrix atoms as compute_tile(); see the comment there for why A is x4 and B is x2.
  const auto s2rAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::half_t>{};
  const auto s2rAtomB = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, cute::half_t>{};
  const auto s2rCopyA = cute::make_tiled_copy_A(s2rAtomA, tiledMma);
  const auto s2rCopyB = cute::make_tiled_copy_B(s2rAtomB, tiledMma);
  const auto thrS2rA = s2rCopyA.get_slice(threadIdx.x);
  const auto thrS2rB = s2rCopyB.get_slice(threadIdx.x);
  auto tXsA = thrS2rA.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  auto tXsB = thrS2rB.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  auto tXrA = thrS2rA.retile_D(tCrA);  // (CPY,MMA_M,MMA_K)
  auto tXrB = thrS2rB.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

  // Fill the ring. Every tile commits its own group so the waits can retire them one at a time.
  const auto prefetch = kTiles < kStages ? kTiles : kStages;
  for (auto s = 0; s < prefetch; ++s) {
    cute::copy(tiledCopy, tAgA(_, _, _, s), tAsA(_, _, _, s));
    cute::copy(tiledCopy, tBgB(_, _, _, s), tBsB(_, _, _, s));
    cute::cp_async_fence();
  }

  cute::clear(tCrC);
  for (auto kt = 0; kt < kTiles; ++kt) {
    const auto stage = kt % kStages;
    if (kt + 1 < kTiles) {
      cute::cp_async_wait<kStages - 1>();
    } else {
      cute::cp_async_wait<0>();
    }
    __syncthreads();

    CUTE_UNROLL
    for (auto kb = 0; kb < cute::size<2>(tCrA); ++kb) {
      cute::copy(s2rAtomA, tXsA(_, _, kb, stage), tXrA(_, _, kb));
      cute::copy(s2rAtomB, tXsB(_, _, kb, stage), tXrB(_, _, kb));
      cute::gemm(tiledMma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }
    __syncthreads();

    if (kt + kStages < kTiles) {
      cute::copy(tiledCopy, tAgA(_, _, _, kt + kStages), tAsA(_, _, _, stage));
      cute::copy(tiledCopy, tBgB(_, _, _, kt + kStages), tBsB(_, _, _, stage));
      cute::cp_async_fence();
    }
  }
}

} // namespace fused_softmax

#endif
