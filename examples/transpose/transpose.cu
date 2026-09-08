#include "transpose.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

// cute/atom/copy_atom.hpp is deliberately absent: it and cute/algorithm/copy.hpp
// include each other, so entering the cycle at copy_atom.hpp leaves Copy_Atom
// undeclared by the time copy.hpp needs it. cute/tensor.hpp enters from the
// other side and pulls both in the working order.
#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor.hpp" // IWYU pragma: keep
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include "cutlass/uint128.h"

using cute::_;
using cute::ceil_div;
using cute::composition;
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
using cute::Swizzle;
using cute::uint128_t;
using cute::uint32_t;
using cute::UniversalCopy;
using std::runtime_error;

namespace {

template <
    class ProblemShape,
    class CtaTiler,
    class StrideIn,
    class StrideOut,
    class SmemLayoutIn,
    class SmemLayoutOut,
    class TiledCopyIn,
    class TiledCopyOut,
    class ComputeLayout>
__global__ void transpose_device(
    ProblemShape probShape,
    CtaTiler ctaTiler,
    const float *dIn,
    StrideIn strideIn,
    float *dOut,
    StrideOut strideOut,
    SmemLayoutIn smemLayoutIn,
    SmemLayoutOut smemlayoutOut,
    TiledCopyIn tiledCopyIn,
    TiledCopyOut tiledCopyOut,
    ComputeLayout computeLayout) {
  const auto mIn = make_tensor(make_gmem_ptr(dIn), probShape, strideIn);
  auto mOut = make_tensor(make_gmem_ptr(dOut), probShape, strideOut);

  const auto ctaCoord = make_coord(blockIdx.x, blockIdx.y);
  const auto gIn = local_tile(mIn, ctaTiler, ctaCoord);
  auto gOut = local_tile(mOut, ctaTiler, ctaCoord);

  // cosize_v is a namespace-scope constexpr variable, so nvcc only accepts it in
  // device code where it folds into a constant expression - fine for the array
  // bound below, but not for the pointer arithmetic that offsets sOut. Naming it
  // once sidesteps that and keeps the offset and the bound from drifting apart.
  constexpr int kSmemIn = cosize_v<SmemLayoutIn>;
  __shared__ float sMem[kSmemIn + cosize_v<SmemLayoutOut>];
  const auto sIn = make_tensor(make_smem_ptr(sMem), smemLayoutIn);
  auto sOut = make_tensor(make_smem_ptr(sMem) + kSmemIn, smemlayoutOut);

  const auto thrCopyIn = tiledCopyIn.get_slice(threadIdx.x);
  const auto tIngIn = thrCopyIn.partition_S(gIn);
  auto tInsIn = thrCopyIn.partition_D(sIn);

  copy(tiledCopyIn, tIngIn(_, _, _), tInsIn(_, _, _));
  cp_async_fence();

  auto tCsIn = local_partition(sIn, computeLayout, threadIdx.x);
  auto tCsOut = local_partition(sOut, computeLayout, threadIdx.x);

  cp_async_wait<0>();
  __syncthreads();
  for (auto i = 0; i < size(tCsIn); ++i) {
    tCsOut[i] = tCsIn[i];
  }

  __syncthreads();
  const auto thrCopyOut = tiledCopyOut.get_slice(threadIdx.x);
  const auto tOutsOut = thrCopyOut.partition_S(sOut);
  auto tOutgOut = thrCopyOut.partition_D(gOut);

  copy(tiledCopyOut, tOutsOut, tOutgOut);
  cp_async_fence();
  cp_async_wait<0>();
}

} // namespace

// Writes the transpose of an m x n row-major matrix of floats into an n x m one:
// dOut[j * ldOut + i] = dIn[i * ldIn + j]. m must be a multiple of 32 and n a multiple of 32 - the
// CTA tile below is not predicated, so a ragged edge would read and write out of bounds.
void transpose(int m, int n, const float *dIn, int ldIn, float *dOut, int ldOut) {
  auto probShape = make_shape(m, n);

  // Both tensors span the same (m, n) index space and the transpose lives entirely in the strides:
  // element (i, j) is at i * ldIn + j in dIn and at j * ldOut + i in dOut. Nothing downstream has
  // to permute a coordinate - the two shared-memory buffers below hold the same logical tile and
  // differ only in how it is physically addressed.
  auto strideIn = make_stride(ldIn, Int<1>{});
  auto strideOut = make_stride(Int<1>{}, ldOut);

  auto bM = Int<32>{};
  auto bN = Int<32>{};
  auto ctaTiler = make_shape(bM, bN);

  // sIn is dense and deliberately not swizzled. copyIn fills it with a 128-bit cp.async, and copy()
  // recasts its destination to uint128_t, which upcasts the swizzle by 4. A swizzle that permutes
  // floats inside a 16-byte chunk cannot survive that: upcast clamps MBase to 0 and silently drops
  // the low bits rather than failing, so the vectorized write and the scalar read below would
  // disagree about where an element lives. sIn needs no swizzle anyway - the relayout reads it
  // along rows, which is already conflict-free.
  auto smemLayoutIn = make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{}));

  // sOut is where the bank conflicts get solved. Swizzle<5, 0, 5> xors the whole 5-bit column index
  // with the row index, which de-conflicts both directions through the buffer: the relayout writes
  // one row per warp, copyOut reads one column, and either way the 32 accesses land on 32 distinct
  // banks. A dense sOut would make that column read a 32-way conflict. MBase is 0, so the
  // permutation reaches inside a 16-byte chunk - legal only because copyOut is a 32-bit atom and
  // never recasts. The period is 2^(0 + 5 + 5) = 1024 floats, exactly one tile, so the atom already
  // is the whole layout and there is nothing for tile_to_shape to repeat. Both smem layouts stay
  // static, which is what lets the kernel size its shared memory with cosize_v.
  auto swizzleAtom = composition(
      Swizzle<5, 0, 5>{},
      make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));
  auto smemLayoutOut = swizzleAtom;

  // Both copies use all 128 threads, and each one's thread layout is picked to be coalesced in the
  // global tensor it touches: copyIn indexes threads along n, where dIn is contiguous, and copyOut
  // along m, where dOut is.
  //
  // copyIn vectorizes, because a thread's four floats along n are contiguous in dIn and in sIn: one
  // 16-byte cp.async per thread, issued twice to cover the 32 rows.
  //
  // copyOut cannot. A thread's four floats along m are contiguous in dOut, but sOut's swizzle
  // scatters them, and no layout of sOut could be contiguous along both m and n anyway. So it stays
  // 32-bit and takes its coalescing from the thread layout instead: 32 threads run down m with
  // stride 1, so one instruction is exactly one 128-byte segment of dOut, repeated eight times to
  // cover the 32 columns. Vectorizing it would mean going through registers - read sOut 32 bits at
  // a time, write the registers back out as a float4 - for the same bytes at a quarter of the
  // instructions.
  auto copyIn = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
      make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));
  auto copyOut = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint32_t>, float>{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{})),
      make_layout(make_shape(Int<1>{}, Int<1>{})));

  // The relayout between the two buffers, one warp per row of the tile. It has to be static:
  // local_partition on the swizzled sOut goes through to_mixed_bits, which static_asserts a
  // power-of-two shape * stride, so a dynamic extent here does not compile at all.
  auto computeLayout =
      make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

  const dim3 dimBlock(size(copyIn));
  const dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));

  transpose_device<<<dimGrid, dimBlock>>>(
      probShape,
      ctaTiler,
      dIn,
      strideIn,
      dOut,
      strideOut,
      smemLayoutIn,
      smemLayoutOut,
      copyIn,
      copyOut,
      computeLayout);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
