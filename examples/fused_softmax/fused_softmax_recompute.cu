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
using fused_softmax::normalize_rows;
using fused_softmax::online_row_stats;
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
// folds it into the row's running (max, sum) with online_row_stats, and throws the tile away; the
// second recomputes it and writes exp(s - max) / sum directly. So P is still written exactly once
// and S is still never written at all - the 2*M*N floor is kept - and the price is 2x the
// multiply-adds instead of extra traffic.
//
// Whether that trade is worth it is a roofline question, not a matter of taste. The problem does
// 2*M*N*K flops and moves 2*M*N bytes, so its arithmetic intensity is K flops per byte; recomputing
// halves the effective intensity to K/2. On a card whose balance point is B = peak_flops /
// peak_bandwidth, this kernel stays memory-bound while 2*K < B and the second GEMM is then close to
// free. At K = 128 that needs B > 256, which is roughly where a consumer Blackwell's fp16 tensor
// throughput against its 448 GB/s of DRAM lands - close enough that the README measures it rather
// than asserting it.
constexpr int kBlockN = 128;

// The three buffers are live at the same time here, unlike in fused_softmax_regrow where the score
// staging can alias the operand tiles: the next tile's operands are being loaded while the current
// tile's scores are still being reduced. (32 + 128) * 64 * 2 + 32 * 128 * 4 = 36 KB, under the
// 48 KB every block gets without opting in.
constexpr int kOperandSmemBytes = (kBlockM + kBlockN) * kBlockK * static_cast<int>(sizeof(__half));
constexpr int kScoresSmemBytes = kBlockM * kBlockN * static_cast<int>(sizeof(float));
constexpr int kSmemBytes = kOperandSmemBytes + kScoresSmemBytes;

static_assert(kSmemBytes <= 48 * 1024, "bump cudaFuncSetAttribute before growing the tile");

// kBlockN / 32 lanes, so each lane owns this many contiguous columns of one row.
constexpr int kColsPerLane = kBlockN / 32;

// The scores land in shared memory row-major so the reductions can walk a row contiguously. The
// GEMM tiles are K-major for the same reason on their side: the copy atom moves 8 contiguous
// halves, and both operands are K-contiguous in global memory.
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

  // Both the N and K coordinates stay residual, so one CTA walks the whole row in tiles instead of
  // owning it outright the way fused_softmax_regrow does.
  const auto ctaCoord = make_coord(blockIdx.x, _, _);
  const Tensor gA = local_tile(mA, ctaTiler, ctaCoord, Step<_1, X, _1>{});
  const Tensor gB = local_tile(mB, ctaTiler, ctaCoord, Step<X, _1, _1>{});

  auto *sAPtr = reinterpret_cast<half_t *>(sRaw);
  auto *sBPtr = sAPtr + static_cast<int64_t>(kBlockM) * kBlockK;
  auto *scoresPtr = reinterpret_cast<float *>(sBPtr + static_cast<int64_t>(kBlockN) * kBlockK);
  const Tensor sA = make_tensor(make_smem_ptr(sAPtr), make_k_major_smem_layout(bM, bK));
  const Tensor sB = make_tensor(make_smem_ptr(sBPtr), make_k_major_smem_layout(bN, bK));
  Tensor sScores = make_tensor(
      make_smem_ptr(scoresPtr), make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

  // One 128-bit atom is 8 halves, and the thread layout puts K fastest so a warp's eight lanes
  // cover 128 contiguous bytes of one row rather than eight separate rows.
  auto tiledCopy = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
      Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  auto tiledMma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<Int<kMmaAtomsM>, Int<kMmaAtomsN>>>{},
      Tile<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>{});
  const auto thrMma = tiledMma.get_slice(threadIdx.x);

  // Only the accumulator side of the MMA is partitioned here; compute_tile() does the operand side
  // itself, from the same sA and sB.
  Tensor tCsScores = thrMma.partition_C(sScores);     // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thrMma.partition_fragment_C(sScores); // (MMA,MMA_M,MMA_N)
  const Tensor tCcC = thrMma.partition_C(make_identity_tensor(make_shape(bM, bN)));

  const int kTiles = size<2>(gA); // (BLK_M,BLK_K,k)
  const int nTiles = size<2>(gB); // (BLK_N,BLK_K,n,k)

  // Cast once here rather than at each use: blockIdx.x is unsigned, and leaving it that way would
  // narrow to int at every use below.
  const auto rowBase = static_cast<int>(blockIdx.x) * kBlockM;
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

  // Pass 1: statistics. Every active tile is computed, folded into (max, sum), and dropped.
  for (auto nt = 0; nt < activeTiles; ++nt) {
    compute_tile(tiledCopy, tiledMma, gA, sA, gB(_, _, nt, _), sB, tCrC, kTiles);
    store_scores(tCrC, tCcC, tCsScores, rowBase, nt * kBlockN, m, n, scale, causal);
    __syncthreads();
    online_row_stats<kColsPerLane>(scoresPtr, sRowMax, sRowSum);
    __syncthreads();
  }

  // Pass 2: recompute and write, now that every row's final statistics are known. Masked columns
  // come back as -INFINITY from store_scores and leave as exactly 0, so this pass needs no mask
  // logic of its own.
  for (auto nt = 0; nt < activeTiles; ++nt) {
    compute_tile(tiledCopy, tiledMma, gA, sA, gB(_, _, nt, _), sB, tCrC, kTiles);
    store_scores(tCrC, tCcC, tCsScores, rowBase, nt * kBlockN, m, n, scale, causal);
    __syncthreads();
    normalize_rows<kColsPerLane>(scoresPtr, dP, ldP, rowBase, nt * kBlockN, sRowMax, sRowSum);
    __syncthreads();
  }

  // The tiles past the causal boundary never reached normalize_rows(), so write their zeros here.
  for (auto nt = activeTiles; nt < nTiles; ++nt) {
    zero_tile<kBlockN>(dP, ldP, rowBase, nt * kBlockN);
  }
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
  // streamed in kBlockN-wide tiles, so the row never has to fit. There is no N tail, so n has to
  // divide.
  if (n % kBlockN != 0) {
    throw runtime_error("fused_softmax_recompute needs n to be a multiple of 128");
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

  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m / kBlockM);
  fused_softmax_recompute_device<<<dimGrid, dimBlock, kSmemBytes>>>(
      dQ, ldQ, dK, ldK, dP, ldP, m, n, k, scale, causal);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
