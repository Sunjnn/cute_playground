#ifndef EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_MAINLOOP_CUH
#define EXAMPLES_FUSED_SOFTMAX_FUSED_SOFTMAX_MAINLOOP_CUH

#include <cmath>
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

} // namespace fused_softmax

#endif
