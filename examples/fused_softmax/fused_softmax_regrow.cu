#include "fused_softmax_regrow.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cute/tensor.hpp" // IWYU pragma: keep

using cute::_;
using cute::_1;
using cute::clear;
using cute::copy;
using cute::Copy_Atom;
using cute::gemm;
using cute::get;
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
using std::int64_t;
using std::runtime_error;

namespace {

// One CTA owns kBlockM rows and *all* n columns, so the whole row of scores ends up in that CTA's
// MMA accumulator and the softmax is a reduction over registers rather than over global memory.
// That is the entire trick: P is written once and S is never written at all, so this kernel moves
// 2*M*N bytes against cublas_unfused_softmax's 10*M*N. The price is that the row has to fit, which
// is what caps n here.
constexpr int kBlockM = 32;
constexpr int kBlockN = 256;
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

// Shared memory: the GEMM tiles want (kBlockM + kBlockN) * kBlockK * 2 = 36 KB and the score
// staging wants kBlockM * kBlockN * 4 = 32 KB, but the two phases never overlap, so one buffer
// sized to the larger serves both. 36 KB stays under the 48 KB every block gets without opting in
// through cudaFuncSetAttribute, and two such CTAs fit an SM's 100 KB.
constexpr int kGemmSmemBytes = (kBlockM + kBlockN) * kBlockK * static_cast<int>(sizeof(__half));
constexpr int kScoresSmemBytes = kBlockM * kBlockN * static_cast<int>(sizeof(float));
constexpr int kSmemBytes = kGemmSmemBytes > kScoresSmemBytes ? kGemmSmemBytes : kScoresSmemBytes;

static_assert(kSmemBytes <= 48 * 1024, "bump cudaFuncSetAttribute before growing the tile");

// Registers, measured rather than counted: ptxas reports 64 per thread with no spills for this
// exact configuration. Counting the fragments statically predicts ~112 - 32 floats of accumulator
// (kBlockM * kBlockN / kThreads) plus 64 registers of tCrB (4 halves * MMA_N(8) * MMA_K(4)) plus 16
// of tCrA - and is wrong because the k_block loop is unrolled and each k_block's B fragment is
// loaded immediately before the gemm that consumes it. Only one k_block's worth stays live, so tCrB
// costs 16 registers rather than 64.
//
// Which means the accumulator, not the operands, is what a wider tile would grow: at kBlockN = 512
// it doubles to 64 floats and the live B fragment to 32 registers, still short of a spill. The real
// cost of 512 is shared memory - 68 KB, past the opt-in threshold - and half as many CTAs. So
// kBlockN = 256 is where the sweep starts, not where it has to stop.
constexpr int kColsPerLane = kBlockN / 32;

// Scale, mask, and hand the accumulator to shared memory in one pass. The identity tensor is what
// turns a flat fragment index into the (row, column) it belongs to, which is the only way to ask
// whether the causal mask bites at that particular element.
template <class TCrC, class TCcC, class TCsScores>
__device__ void store_scores(
    const TCrC &tCrC,
    const TCcC &tCcC,
    TCsScores &tCsScores,
    int rowBase,
    int m,
    int n,
    float scale,
    bool causal) {
  CUTE_UNROLL
  for (auto i = 0; i < size(tCrC); ++i) {
    const auto row = rowBase + static_cast<int>(get<0>(tCcC(i)));
    const auto col = static_cast<int>(get<1>(tCcC(i)));
    auto value = tCrC(i) * scale;
    // Column j is visible iff j <= row + n - m, which keeps the last query aligned with the last
    // key when m < n. main.cu rejects m > n, so every row keeps at least column 0 and the row sum
    // below is never a division by zero.
    if (causal && col > row + n - m) {
      value = -INFINITY;
    }
    tCsScores(i) = value;
  }
}

// One warp per row, kWarps rows at a time. Every column of a row lives in the one warp that owns
// it, so both reductions are pure shuffles and need no shared-memory scratch - unlike
// cublas_unfused_softmax's kernel, where a row spans the whole CTA.
//
// Each lane owns kColsPerLane contiguous columns, which is what lets the store go out as __half2
// pairs. It also means the shared-memory reads stride 32 bytes between lanes and collide in banks;
// that is a bandwidth problem and not a correctness one, and it is the first thing to fix once the
// numbers say this kernel is worth tuning.
__device__ void softmax_rows(const float *scores, __half *dP, int ldP, int rowBase) {
  const auto warpId = static_cast<int>(threadIdx.x) / 32;
  const auto laneId = static_cast<int>(threadIdx.x) % 32;
  const auto laneCol = laneId * kColsPerLane;

  for (auto r = warpId; r < kBlockM; r += kWarps) {
    const auto *row = scores + static_cast<int64_t>(r) * kBlockN + laneCol;
    auto *out = reinterpret_cast<__half2 *>(dP + static_cast<int64_t>(rowBase + r) * ldP + laneCol);

    float values[kColsPerLane];
    CUTE_UNROLL
    for (auto c = 0; c < kColsPerLane; ++c) {
      values[c] = row[c];
    }

    auto rowMax = -INFINITY;
    CUTE_UNROLL
    for (const auto value : values) {
      rowMax = fmaxf(rowMax, value);
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      rowMax = fmaxf(rowMax, __shfl_xor_sync(0xffffffffu, rowMax, offset));
    }

    // A masked column carries -INFINITY, which fmaxf() absorbed above and __expf() turns into
    // exactly 0 here, so it drops out of both the sum and the output without a special case.
    auto rowSum = 0.0f;
    CUTE_UNROLL
    for (auto &value : values) {
      value = __expf(value - rowMax);
      rowSum += value;
    }
#pragma unroll
    for (auto offset = 16; offset > 0; offset /= 2) {
      rowSum += __shfl_xor_sync(0xffffffffu, rowSum, offset);
    }

    // One reciprocal per row rather than one division per element.
    const auto invSum = 1.0f / rowSum;
    CUTE_UNROLL
    for (auto c = 0; c < kColsPerLane; c += 2) {
      out[c / 2] = __floats2half2_rn(values[c] * invSum, values[c + 1] * invSum);
    }
  }
}

// The scores land in shared memory row-major so softmax_rows() can walk a row contiguously. The
// GEMM tiles are K-major for the same reason on their side: the copy atom moves 8 contiguous
// halves, and both operands are K-contiguous in global memory.
__global__ __launch_bounds__(kThreads) void fused_softmax_regrow_device(
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

  // n is exactly kBlockN for this build, so the N tile index is always the static zero and one CTA
  // spans the whole row.
  const auto ctaCoord = make_coord(blockIdx.x, Int<0>{}, _);
  Tensor gA = local_tile(mA, ctaTiler, ctaCoord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, ctaTiler, ctaCoord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)

  auto *sAPtr = reinterpret_cast<half_t *>(sRaw);
  auto *sBPtr = sAPtr + static_cast<int64_t>(kBlockM) * kBlockK;
  Tensor sA =
      make_tensor(make_smem_ptr(sAPtr), make_layout(make_shape(bM, bK), make_stride(bK, Int<1>{})));
  Tensor sB =
      make_tensor(make_smem_ptr(sBPtr), make_layout(make_shape(bN, bK), make_stride(bK, Int<1>{})));

  // One 128-bit atom is 8 halves, and the thread layout puts K fastest so a warp's eight lanes
  // cover 128 contiguous bytes of one row rather than eight separate rows.
  auto tiledCopy = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
      Layout<Shape<Int<32>, Int<8>>, Stride<Int<8>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  // get_slice is static on TiledCopy but not on TiledMMA, so the two calls below look different.
  // transpose.cu gets away with the instance form because its TiledCopy is a template parameter and
  // clang-tidy cannot see through a dependent type.
  const auto thrCopy = decltype(tiledCopy)::get_slice(threadIdx.x);
  Tensor tAgA = thrCopy.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thrCopy.partition_D(sA); // (CPY,CPY_M,CPY_K)
  Tensor tBgB = thrCopy.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thrCopy.partition_D(sB); // (CPY,CPY_N,CPY_K)

  auto tiledMma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<Int<kMmaAtomsM>, Int<kMmaAtomsN>>>{},
      Tile<Int<kMmaM>, Int<kMmaN>, Int<kMmaK>>{});
  const auto thrMma = tiledMma.get_slice(threadIdx.x);

  // Aliases sA and sB. Declared here because the accumulator's shape is derived from it, but not
  // written to until the barrier after the K loop retires both operand tiles.
  auto *scoresPtr = reinterpret_cast<float *>(sRaw);
  Tensor sScores = make_tensor(
      make_smem_ptr(scoresPtr), make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

  Tensor tCsScores = thrMma.partition_C(sScores);     // (MMA,MMA_M,MMA_N)
  Tensor tCrA = thrMma.partition_fragment_A(sA);      // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thrMma.partition_fragment_B(sB);      // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thrMma.partition_fragment_C(sScores); // (MMA,MMA_M,MMA_N)
  const Tensor tCcC = thrMma.partition_C(make_identity_tensor(make_shape(bM, bN)));
  Tensor tCsA = thrMma.partition_A(sA);               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thrMma.partition_B(sB);               // (MMA,MMA_N,MMA_K)

  clear(tCrC);

  const int kTiles = size<3>(tAgA);
  for (auto kt = 0; kt < kTiles; ++kt) {
    // Synchronous on purpose: this is the version that is easy to see is correct. The cp.async ring
    // replaces these two copies and the barriers around them.
    copy(tiledCopy, tAgA(_, _, _, kt), tAsA);
    copy(tiledCopy, tBgB(_, _, _, kt), tBsB);
    __syncthreads();

    CUTE_UNROLL
    for (auto kb = 0; kb < size<2>(tCrA); ++kb) {
      // Elementwise rather than through an SM75_U32x4_LDSM_N atom: ldmatrix needs a swizzled smem
      // layout to be worth anything, and this build has none.
      copy(tCsA(_, _, kb), tCrA(_, _, kb));
      copy(tCsB(_, _, kb), tCrB(_, _, kb));
      gemm(tiledMma, tCrA(_, _, kb), tCrB(_, _, kb), tCrC);
    }
    __syncthreads();
  }

  // Cast once here rather than at each use: blockIdx.x is unsigned, and leaving it that way would
  // narrow to int at the call sites below.
  const auto rowBase = static_cast<int>(blockIdx.x) * kBlockM;

  store_scores(tCrC, tCcC, tCsScores, rowBase, m, n, scale, causal);
  __syncthreads();

  softmax_rows(scoresPtr, dP, ldP, rowBase);
}

} // namespace

void fused_softmax_regrow(
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

  // The whole row has to be one CTA tile, so n is not a range here but a single supported value.
  // Relaxing it means either a second instantiation or the recompute variant.
  if (n != kBlockN) {
    throw runtime_error("fused_softmax_regrow holds a whole row on chip, so n must be 256");
  }
  if (m % kBlockM != 0) {
    throw runtime_error("fused_softmax_regrow needs m to be a multiple of 32");
  }
  if (k % kBlockK != 0) {
    throw runtime_error("fused_softmax_regrow needs k to be a multiple of 64");
  }
  // The store goes out as __half2 pairs, so every row has to start 4-byte aligned.
  if (ldP % 2 != 0) {
    throw runtime_error("fused_softmax_regrow needs ldP to be even");
  }

  const dim3 dimBlock(kThreads);
  const dim3 dimGrid(m / kBlockM);
  fused_softmax_regrow_device<<<dimGrid, dimBlock, kSmemBytes>>>(
      dQ, ldQ, dK, ldK, dP, ldP, m, n, k, scale, causal);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
