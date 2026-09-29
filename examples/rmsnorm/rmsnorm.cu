#include "rmsnorm.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

// cute/atom/copy_atom.hpp is deliberately absent: it and cute/algorithm/copy.hpp include each
// other, so entering the cycle at copy_atom.hpp leaves Copy_Atom undeclared by the time copy.hpp
// needs it. cute/tensor.hpp enters from the other side and pulls both in the working order.
#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor.hpp" // IWYU pragma: keep
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include "cutlass/uint128.h"

using cute::_;
using cute::ceil_div;
using cute::copy;
using cute::Copy_Atom;
using cute::cosize_v;
using cute::cp_async_fence;
using cute::cp_async_wait;
using cute::Int;
using cute::local_partition;
using cute::local_tile;
using cute::make_coord;
using cute::make_gmem_ptr;
using cute::make_layout;
using cute::make_shape;
using cute::make_smem_ptr;
using cute::make_stride;
using cute::make_tensor;
using cute::make_tiled_copy;
using cute::size;
using cute::SM80_CP_ASYNC_CACHEALWAYS;
using cute::uint128_t;
using cute::UniversalCopy;
using std::runtime_error;

namespace {

// One float per warp for the cross-warp step of the row reduction. 32 of them cover the 1024
// threads a block may have, which is the most this kernel could ever ask for.
constexpr int kMaxWarpNum = 32;

// One CTA per row, two walks over it, each tile staged through shared memory with cp.async - the
// same shape as softmax(), whose row reduction RMSNorm's sum of squares is a variant of.
//
// Pass one accumulates the row's sum of squares in float - good to about 1e-6 relative at these
// row lengths, well inside the harness's 1e-4 - pass two reloads each tile, scales it by
// rsqrt(mean + eps) * weight and stores it. Reloading rather than keeping the row is what makes
// this the two-pass version: it moves three times the row's bytes where rmsnorm_fused, which
// caches the row in registers instead, moves two.
//
// The weight tile is loaded alongside the input tile in pass two. Every CTA asks for the same n
// floats, so after the first few CTAs the whole vector is L2-resident and the loads cost L2
// bandwidth rather than DRAM; staging it through shared memory keeps the multiply off the
// critical path either way and lets the compute loop read both operands at the same index. The
// vector is described to CuTe as a (1, n) tensor with a broadcasting row stride and tiled at row
// 0, so one descriptor serves every blockIdx.
//
// Pass two's multiply writes the tile in place: a thread only ever touches the elements it reads,
// so the loop needs no barrier of its own, and the barrier after it is what makes the scaled tile
// visible to the copy out, which is partitioned differently and so does read other threads'
// elements.
template <
    class ProblemShape,
    class CtaTiler,
    class StrideIn,
    class StrideWeight,
    class StrideOut,
    class SmemLayoutTile,
    class TiledCopyIn,
    class TiledCopyOut,
    class ComputeLayout>
__global__ void rmsnorm_device(
    ProblemShape probShape,
    CtaTiler ctaTiler,
    const float *dIn,
    StrideIn strideIn,
    const float *dWeight,
    StrideWeight strideWeight,
    float *dOut,
    StrideOut strideOut,
    SmemLayoutTile smemLayoutTile,
    TiledCopyIn tiledCopyIn,
    TiledCopyOut tiledCopyOut,
    ComputeLayout computeLayout,
    int n,
    float eps) {
  const auto mIn = make_tensor(make_gmem_ptr(dIn), probShape, strideIn);
  auto mOut = make_tensor(make_gmem_ptr(dOut), probShape, strideOut);
  const auto mWeight = make_tensor(make_gmem_ptr(dWeight), make_shape(Int<1>{}, n), strideWeight);

  const auto ctaCoord = make_coord(blockIdx.x, _);
  const auto gIn = local_tile(mIn, ctaTiler, ctaCoord);
  auto gOut = local_tile(mOut, ctaTiler, ctaCoord);
  const auto gWeight = local_tile(mWeight, ctaTiler, make_coord(0, _));

  __shared__ float sTileMem[cosize_v<SmemLayoutTile>];
  __shared__ float sWeightMem[cosize_v<SmemLayoutTile>];
  __shared__ float sSums[kMaxWarpNum];
  const auto sTile = make_tensor(make_smem_ptr(sTileMem), smemLayoutTile);
  const auto sWeight = make_tensor(make_smem_ptr(sWeightMem), smemLayoutTile);

  const auto thrCopyIn = tiledCopyIn.get_slice(threadIdx.x);
  const auto tIngIn = thrCopyIn.partition_S(gIn);
  auto tInsTile = thrCopyIn.partition_D(sTile);
  const auto tWgWeight = thrCopyIn.partition_S(gWeight);
  auto tWsWeight = thrCopyIn.partition_D(sWeight);

  const auto tCsTile = local_partition(sTile, computeLayout, threadIdx.x);
  const auto tCsWeight = local_partition(sWeight, computeLayout, threadIdx.x);

  const auto blockNum = size<2>(gIn);

  // Pass one: sum of squares, folded per thread so the block reduction sees one float each.
  auto thrSqrSum = 0.0f;
  for (auto blockCount = 0; blockCount < blockNum; ++blockCount) {
    __syncthreads();
    copy(tiledCopyIn, tIngIn(_, _, _, blockCount), tInsTile(_, _, _));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (auto i = 0; i < size(tCsTile); ++i) {
      thrSqrSum += tCsTile[i] * tCsTile[i];
    }
  }

  auto warpSqrSum = thrSqrSum;
  for (auto offset = 16; offset > 0; offset /= 2) {
    warpSqrSum += __shfl_xor_sync(0xffffffffu, warpSqrSum, offset);
  }
  const auto warpId = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    sSums[warpId] = warpSqrSum;
  }

  __syncthreads();
  const auto warpNum = blockDim.x / 32;
  auto rowSqrSum = 0.0f;
  for (auto i = 0u; i < warpNum; ++i) {
    rowSqrSum += sSums[i];
  }

  // One reciprocal square root per row rather than one division per element.
  const auto scale = rsqrtf(rowSqrSum / static_cast<float>(n) + eps);

  const auto thrCopyOut = tiledCopyOut.get_slice(threadIdx.x);
  const auto tOutsTile = thrCopyOut.partition_S(sTile);
  auto tOutgOut = thrCopyOut.partition_D(gOut);

  // Pass two: reload, scale, store.
  for (auto blockCount = 0; blockCount < blockNum; ++blockCount) {
    __syncthreads();
    copy(tiledCopyIn, tIngIn(_, _, _, blockCount), tInsTile(_, _, _));
    copy(tiledCopyIn, tWgWeight(_, _, _, blockCount), tWsWeight(_, _, _));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (auto i = 0; i < size(tCsTile); ++i) {
      tCsTile[i] = tCsTile[i] * scale * tCsWeight[i];
    }

    __syncthreads();
    copy(tiledCopyOut, tOutsTile(_, _, _), tOutgOut(_, _, _, blockCount));
  }
}

} // namespace

// Normalizes each row of an m x n row-major matrix and scales it by a per-column weight:
// dOut[i * ldOut + j] = dIn[i * ldIn + j] * rsqrt(mean_j(dIn[i * ldIn + j]^2) + eps) * dWeight[j].
//
// n must be a multiple of 512 and both leading dimensions multiples of 4: the CTA tile below is
// not predicated and the copies are 16 bytes wide, so a ragged row would read and write out of
// bounds and an unaligned row start would fault.
void rmsnorm(
    int m,
    int n,
    const float *dIn,
    int ldIn,
    const float *dWeight,
    float *dOut,
    int ldOut,
    float eps) {
  auto bM = Int<1>{};
  auto bN = Int<512>{};

  if (n % bN != 0 || ldIn % 4 != 0 || ldOut % 4 != 0) {
    throw runtime_error("rmsnorm needs n a multiple of 512 and ldIn, ldOut multiples of 4");
  }

  auto probShape = make_shape(m, n);
  auto ctaTiler = make_shape(bM, bN);

  auto strideIn = make_stride(ldIn, Int<1>{});
  auto strideOut = make_stride(ldOut, Int<1>{});
  auto strideWeight = make_stride(Int<0>{}, Int<1>{});

  // Both smem buffers are dense (1, 512) row-major tiles. Nothing reads a tile along its columns
  // the way transpose does - the compute loop and the copy out both walk n - so there is no bank
  // conflict to swizzle away and the layout stays static, which is what lets the kernel size its
  // shared memory with cosize_v.
  auto smemLayoutTile = make_layout(make_shape(bM, bN));

  // cp.async in, plain st.global out, both 16 bytes per thread: 128 threads x 4 floats covers the
  // tile in one instruction each, so a warp's store is one 128-byte segment of dOut.
  auto copyIn = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
      make_layout(make_shape(Int<1>{}, Int<128>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));
  auto copyOut = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, float>{},
      make_layout(make_shape(Int<1>{}, Int<128>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));

  // The compute partition: 128 threads over the (1, 512) tile, so thread t owns elements
  // t, t + 128, t + 256 and t + 384. Strided rather than blocked, which is what makes the
  // shared-memory reads of a warp cover 32 distinct banks.
  auto computeLayout = make_layout(make_shape(Int<1>{}, Int<128>{}));

  const dim3 dimBlock(size(computeLayout));
  const dim3 dimGrid(size(ceil_div(m, bM)));

  rmsnorm_device<<<dimGrid, dimBlock>>>(
      probShape,
      ctaTiler,
      dIn,
      strideIn,
      dWeight,
      strideWeight,
      dOut,
      strideOut,
      smemLayoutTile,
      copyIn,
      copyOut,
      computeLayout,
      n,
      eps);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
