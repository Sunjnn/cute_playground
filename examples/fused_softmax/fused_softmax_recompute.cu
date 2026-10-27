#include "fused_softmax_recompute.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cute/tensor.hpp" // IWYU pragma: keep

#include "fused_softmax_mainloop.cuh"

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
using cute::X;
using fused_softmax::accumulate_tile_stats;
using fused_softmax::compute_tile_async;
using fused_softmax::kBlockK;
using fused_softmax::kBlockM;
using fused_softmax::kMmaAtomsM;
using fused_softmax::kMmaAtomsN;
using fused_softmax::kMmaK;
using fused_softmax::kMmaM;
using fused_softmax::kMmaN;
using fused_softmax::kThreads;
using fused_softmax::make_k_major_smem_layout;
using fused_softmax::normalize_rows;
using fused_softmax::stage_prepared_scores;
using fused_softmax::store_scores;
using fused_softmax::zero_tile;
using std::int64_t;
using std::runtime_error;

namespace {

// What fused_softmax_regrow trades away, and what it buys. That kernel holds a whole row on chip,
// so it reads Q and K once and writes P once - 2*M*N bytes, the floor - but the row has to fit, and
// at kBlockM = 32 that caps n at 256 before shared memory needs opting in.
//
// This one streams the columns instead. It walks all of n twice: the first pass computes each tile,
// folds it into the row's running (max, sum) straight from the accumulator, and throws the tile
// away; the second recomputes it and writes exp(s - max) / sum directly. So P is still written
// exactly once and S is still never written at all - the 2*M*N floor is kept - and the price is 2x
// the multiply-adds instead of extra traffic.
//
// Whether that trade is worth it is a roofline question, not a matter of taste. The problem does
// 2*M*N*K flops and moves 2*M*N bytes, so its arithmetic intensity is K flops per byte; recomputing
// halves the effective intensity to K/2. On a card whose balance point is B = peak_flops /
// peak_bandwidth, this kernel stays memory-bound while 2*K < B and the second GEMM is then close to
// free. At K = 128 that needs B > 256, which is roughly where a consumer Blackwell's fp16 tensor
// throughput against its 448 GB/s of DRAM lands - close enough that the README measures it rather
// than asserting it.

// Column tile width. The kernel is instantiated at 128 and 64:
//   - 128 is the default: fewer, larger tiles per CTA, and with the aliased score buffer it fits
//     four CTAs per SM;
//   - 64 exists for widths that are a multiple of 64 but not 128 (192, 320, ...). It was also the
//     large-grid choice before the score buffer was aliased; once the 128-wide residency reached
//     four it measured slower at every shape where both fit.
// Both fit under the 48 KB a block gets without opting in, so neither needs an attribute.
template <int kBlockN> constexpr int operand_smem_bytes() {
  return (kBlockM + kBlockN) * kBlockK * static_cast<int>(sizeof(__half));
}

template <int kBlockN> constexpr int scores_smem_bytes() {
  return kBlockM * kBlockN * static_cast<int>(sizeof(float));
}

// The score staging aliases the operand buffers, as in fused_softmax_regrow. With one cp.async
// stage they are disjoint in time: a tile's operands are consumed by its GEMM before store_scores()
// writes the scores, and the next tile's copy is only issued after normalize_rows() has read them
// (the last-tile reuse follows the same order across the two passes). That drops the footprint from
// operand + scores (36 KB at 128 wide) to their maximum (20 KB), which is what takes the 128-wide
// build from two CTAs per SM to four.
//
// kStages is 1 for both instantiations today. A two-stage ring was measured and lost a CTA of
// residency at each width, which cost more at the large shapes than the overlap bought; the
// parameter stays so the trade can be retried when the tile changes. What did pay is the cp.async
// copy itself, with one commit group per tile and a wait before the read: it removed the
// synchronous load/store chain and is worth 7-18% on the 128-wide shapes with fewer CTAs
// (1024 x 2048, 1024 x 8192, and the causal squares).
template <int kBlockN, int kStages> constexpr int smem_bytes() {
  constexpr auto operands = kStages * operand_smem_bytes<kBlockN>();
  constexpr auto scores = scores_smem_bytes<kBlockN>();
  return operands > scores ? operands : scores;
}

static_assert(
    smem_bytes<64, 1>() <= 48 * 1024, "bump cudaFuncSetAttribute before growing the tile");
static_assert(
    smem_bytes<128, 1>() <= 48 * 1024, "bump cudaFuncSetAttribute before growing the tile");

// The scores land in shared memory row-major so the reductions can walk a row contiguously. The
// GEMM tiles are K-major for the same reason on their side: the copy atom moves 8 contiguous
// halves, and both operands are K-contiguous in global memory.
template <int kBlockN, int kStages>
__global__ __launch_bounds__(kThreads) void fused_softmax_recompute_device(
    const __half *__restrict__ dQ,
    int ldQ,
    const __half *__restrict__ dK,
    int ldK,
    __half *__restrict__ dP,
    int ldP,
    int m,
    int n,
    int k,
    float scale,
    bool causal) {
  extern __shared__ __align__(16) char sRaw[];
  __shared__ float sRowMax[kBlockM];
  __shared__ float sRowSum[kBlockM];
  // Per-tile reduction scratch for pass 1: one max and one sum per row per N atom.
  __shared__ float sTileMax[2][kBlockM][kMmaAtomsN];
  __shared__ float sTileSum[2][kBlockM][kMmaAtomsN];

  const auto bM = Int<kBlockM>{};
  const auto bN = Int<kBlockN>{};
  const auto bK = Int<kBlockK>{};
  const auto ctaTiler = make_shape(bM, bN, bK);

  // Q is (M, K) row-major and K is (N, K) row-major, so both are K-contiguous and the pair is
  // already the TN that SM80_16x8x16_F32F16F16F32_TN wants: S = Q K^T needs no transpose anywhere.
  // half_t and __half are layout-compatible, which is what the casts below rely on.
  const auto *query = reinterpret_cast<const half_t *>(dQ);
  const auto *key = reinterpret_cast<const half_t *>(dK);
  Tensor mA = make_tensor(make_gmem_ptr(query), make_shape(m, k), make_stride(ldQ, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(key), make_shape(n, k), make_stride(ldK, Int<1>{}));

  // Under a causal mask the last row blocks are the heaviest: they see every column, while the
  // first block sees only its own 32. Blocks are dispatched in increasing blockIdx, so leaving the
  // mapping alone starts the heaviest CTAs last and lets the shortest ones idle the SMs while the
  // tail runs. Reversing it puts the heavy CTAs first, the classic longest-processing-time-first
  // schedule, and needs no persistent grid to enforce it. Unmasked shapes have equal work in every
  // block, so they keep the natural order (and its sequential Q reads). Both the Q tile below and
  // the output rows have to follow this mapping, not blockIdx.
  const auto rowBlock =
      causal ? static_cast<int>(gridDim.x - 1 - blockIdx.x) : static_cast<int>(blockIdx.x);

  // Both the N and K coordinates stay residual, so one CTA walks the whole row in tiles instead of
  // owning it outright the way fused_softmax_regrow does.
  const auto ctaCoord = make_coord(rowBlock, _, _);
  const Tensor gA = local_tile(mA, ctaTiler, ctaCoord, Step<_1, X, _1>{});
  const Tensor gB = local_tile(mB, ctaTiler, ctaCoord, Step<X, _1, _1>{});

  auto *sAPtr = reinterpret_cast<half_t *>(sRaw);
  auto *sBPtr = sAPtr + static_cast<int64_t>(kBlockM) * kBlockK * kStages;
  // The score staging aliases the operands; see smem_bytes() for why their lifetimes do not
  // overlap.
  auto *scoresPtr = reinterpret_cast<float *>(sRaw);
  const Tensor sA = make_tensor(make_smem_ptr(sAPtr), make_k_major_smem_layout<kStages>(bM, bK));
  const Tensor sB = make_tensor(make_smem_ptr(sBPtr), make_k_major_smem_layout<kStages>(bN, bK));
  Tensor sScores = make_tensor(
      make_smem_ptr(scoresPtr), make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

  // One 128-bit atom is 8 halves, and the thread layout puts K fastest so a warp's eight lanes
  // cover 128 contiguous bytes of one row rather than eight separate rows.
  // cp.async, not UniversalCopy: this copy is what the ring in compute_tile_async() overlaps with
  // the ldmatrix and MMA of the previous tile.
  auto tiledCopy = make_tiled_copy(
      Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  auto tiledMma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<Int<kMmaAtomsM>, Int<kMmaAtomsN>>>{},
      Tile<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>{});
  const auto thrMma = tiledMma.get_slice(threadIdx.x);

  // Only the accumulator side of the MMA is partitioned here; compute_tile_async() does the operand
  // side itself, from the same sA and sB.
  Tensor tCsScores = thrMma.partition_C(sScores);     // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thrMma.partition_fragment_C(sScores); // (MMA,MMA_M,MMA_N)
  const Tensor tCcC = thrMma.partition_C(make_identity_tensor(make_shape(bM, bN)));

  const int kTiles = size<2>(gA); // (BLK_M,BLK_K,k)
  const int nTiles = size<2>(gB); // (BLK_N,BLK_K,n,k)

  // Cast once here rather than at each use: blockIdx.x is unsigned, and leaving it that way would
  // narrow to int at every use below.
  const auto rowBase = rowBlock * kBlockM;
  const auto threadId = static_cast<int>(threadIdx.x);

  // Under a causal mask, the tile's last row sees no column past this one: column j is visible iff
  // j <= row + n - m, and the last row is the most permissive row in the block. A kBlockN-wide
  // column tile that starts past it is entirely masked - its scores would all be -INFINITY and its
  // contribution to (max, sum) exactly nothing - so both passes can skip computing it. The output
  // still has to be written, which zero_tile() does at the end.
  const auto visMax = causal ? rowBase + kBlockM - 1 + (n - m) : n - 1;
  const auto activeTiles = min(nTiles, max(0, visMax / kBlockN + 1));

  for (auto r = threadId; r < kBlockM; r += kThreads) {
    sRowMax[r] = -INFINITY;
    sRowSum[r] = 0.0f;
  }
  __syncthreads();

  // Pass 1: statistics from the accumulator. The tile is folded into (max, sum) straight out of
  // registers, so no score staging is written or read; the one exception is the last active tile,
  // which is staged because pass 2 reuses it.
  for (auto nt = 0; nt < activeTiles; ++nt) {
    compute_tile_async<kStages>(tiledCopy, tiledMma, gA, sA, gB(_, _, nt, _), sB, tCrC, kTiles);
    accumulate_tile_stats(
        tCrC,
        tCcC,
        sTileMax,
        sTileSum,
        sRowMax,
        sRowSum,
        nt & 1,
        rowBase,
        nt * kBlockN,
        m,
        n,
        scale,
        causal);
    if (nt + 1 == activeTiles) {
      stage_prepared_scores(tCrC, tCsScores);
    }
  }
  __syncthreads();

  // Pass 2: write, now that every row's final statistics are known. The last active tile is still
  // in shared memory from pass 1 - the statistics pass only read it - so it is normalized in place
  // instead of being recomputed, and is consumed before the loop below stores anything. That saves
  // one GEMM and one copy per CTA: a sixth of the second pass at n = 256 (two tiles) and all of it
  // at n = 128 (one tile).
  normalize_rows<kBlockN / 32>(
      scoresPtr, dP, ldP, rowBase, (activeTiles - 1) * kBlockN, sRowMax, sRowSum);
  __syncthreads();
  for (auto tile = 0; tile + 1 < activeTiles; ++tile) {
    compute_tile_async<kStages>(tiledCopy, tiledMma, gA, sA, gB(_, _, tile, _), sB, tCrC, kTiles);
    store_scores(tCrC, tCcC, tCsScores, rowBase, tile * kBlockN, m, n, scale, causal);
    __syncthreads();
    normalize_rows<kBlockN / 32>(scoresPtr, dP, ldP, rowBase, tile * kBlockN, sRowMax, sRowSum);
    __syncthreads();
  }

  // The tiles past the causal boundary never reached normalize_rows(), so write their zeros here.
  for (auto nt = activeTiles; nt < nTiles; ++nt) {
    zero_tile<kBlockN>(dP, ldP, rowBase, nt * kBlockN);
  }
}

template <int kBlockN, int kStages>
void launch_recompute(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    __half *dP,
    int ldP,
    float scale,
    bool causal) {
  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m / kBlockM);
  fused_softmax_recompute_device<kBlockN, kStages>
      <<<dimGrid, dimBlock, smem_bytes<kBlockN, kStages>()>>>(
          dQ, ldQ, dK, ldK, dP, ldP, m, n, k, scale, causal);
}

} // namespace

void fused_softmax_recompute(
    int m,
    int n,
    int k,
    const __half *dQ,
    int ldQ,
    const __half *dK,
    int ldK,
    __half *dP,
    int ldP,
    float scale,
    bool causal) {
  static_assert(sizeof(__half) == sizeof(half_t), "the casts above need these to match");

  // Unlike fused_softmax_regrow, n is a multiple rather than a fixed value: the columns are
  // streamed in 64- or 128-wide tiles, so the row never has to fit. There is no N tail, so n has to
  // divide. 64 is the common multiple, and the only one the narrow instantiation needs.
  if (n % 64 != 0) {
    throw runtime_error("fused_softmax_recompute needs n to be a multiple of 64");
  }
  if (m % kBlockM != 0) {
    throw runtime_error("fused_softmax_recompute needs m to be a multiple of 32");
  }
  if (k % kBlockK != 0) {
    throw runtime_error("fused_softmax_recompute needs k to be a multiple of 64");
  }
  // The store goes out as __half2 pairs, so every row has to start 4-byte aligned.
  if (ldP % 2 != 0) {
    throw runtime_error("fused_softmax_recompute needs ldP to be even");
  }

  // 128 wide is the default: with the score staging aliased onto the operands it holds four CTAs
  // per SM, and it halves the number of tiles per CTA against the 64-wide build. The narrow
  // instantiation exists for widths that are a multiple of 64 but not 128 (192, 320, ...); it
  // measured slower at every shape where both fit once the alias raised the 128-wide residency,
  // so there is no occupancy-based dispatch to make.
  if (n % 128 != 0) {
    launch_recompute<64, 1>(m, n, k, dQ, ldQ, dK, ldK, dP, ldP, scale, causal);
  } else {
    launch_recompute<128, 1>(m, n, k, dQ, ldQ, dK, ldK, dP, ldP, scale, causal);
  }
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
