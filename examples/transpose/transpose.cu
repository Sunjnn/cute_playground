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

using cute::_;
using cute::ceil_div;
using cute::composition;
using cute::Copy_Atom;
using cute::cosize_v;
using cute::cp_async_fence;
using cute::cp_async_wait;
using cute::Int;
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
using cute::uint32_t;
using cute::UniversalCopy;
using std::runtime_error;

namespace {

template <
    class ProblemShape,
    class CtaTiler,
    class StrideIn,
    class StrideOut,
    class SmemLayout,
    class TiledCopyIn,
    class TiledCopyOut>
__global__ void transpose_device(
    ProblemShape probShape,
    CtaTiler ctaTiler,
    const float *dIn,
    StrideIn strideIn,
    float *dOut,
    StrideOut strideOut,
    SmemLayout smemLayout,
    TiledCopyIn tiledCopyIn,
    TiledCopyOut tiledCopyOut) {
  const auto mIn = make_tensor(make_gmem_ptr(dIn), probShape, strideIn);
  auto mOut = make_tensor(make_gmem_ptr(dOut), probShape, strideOut);

  const auto ctaCoord = make_coord(blockIdx.x, blockIdx.y);
  const auto gIn = local_tile(mIn, ctaTiler, ctaCoord);
  auto gOut = local_tile(mOut, ctaTiler, ctaCoord);

  __shared__ float sMem[cosize_v<SmemLayout>];
  const auto sTile = make_tensor(make_smem_ptr(sMem), smemLayout);

  const auto thrCopyIn = tiledCopyIn.get_slice(threadIdx.x);
  const auto tIngIn = thrCopyIn.partition_S(gIn);
  auto tInsTile = thrCopyIn.partition_D(sTile);

  copy(tiledCopyIn, tIngIn(_, _, _), tInsTile(_, _, _));
  cp_async_fence();

  cp_async_wait<0>();
  __syncthreads();
  const auto thrCopyOut = tiledCopyOut.get_slice(threadIdx.x);
  const auto tOutsTile = thrCopyOut.partition_S(sTile);
  auto tOutgOut = thrCopyOut.partition_D(gOut);

  copy(tiledCopyOut, tOutsTile, tOutgOut);
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
  // to permute a coordinate - the shared-memory tile below is a pure pass-through, written and read
  // at the same logical (i, j).
  auto strideIn = make_stride(ldIn, Int<1>{});
  auto strideOut = make_stride(Int<1>{}, ldOut);

  auto bM = Int<32>{};
  auto bN = Int<32>{};
  auto ctaTiler = make_shape(bM, bN);

  // One buffer, one layout. copyIn writes logical (i, j) and copyOut reads logical (i, j) - the
  // transpose itself lives entirely in the global strides - so the two only round-trip through
  // shared memory if they agree on where (i, j) physically sits. Two layouts over the same array (a
  // dense one to load into, a swizzled one to store out of) need an explicit relayout pass between
  // them to move the data from one arrangement to the other; without it every row but the zeroth
  // reads back permuted.
  //
  // Swizzle<5, 0, 5> xors the whole 5-bit column index with the row index, which de-conflicts both
  // directions through the buffer: copyIn writes along a row and copyOut reads down a column, and
  // either way the 32 accesses land on 32 distinct banks. A dense buffer would make that column
  // read a 32-way conflict. The period is 2^(0 + 5 + 5) = 1024 floats, exactly one tile, so the
  // swizzle atom already is the whole layout. It stays static, which is what lets the kernel size
  // its shared memory with cosize_v.
  //
  // MBase 0 is what forces copyIn down to a 4-byte atom. The permutation reaches inside a 16-byte
  // chunk, and copy() recasts a wide atom's destination to uint128_t; upcast<4> then subtracts
  // log2(4) from MBase, goes negative, and silently truncates Swizzle<5, 0, 5> to Swizzle<3, 0, 5>
  // rather than failing. Two bits dropped means the vectorized write no longer lands where the
  // scalar read looks. A 4-byte atom never recasts, so the two stay in agreement.
  auto smemLayout =
      composition(Swizzle<5, 0, 5>{}, make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));

  // Both copies use all 128 threads, and each one's thread layout is picked to be coalesced in the
  // global tensor it touches: copyIn indexes threads along n, where dIn is contiguous, and copyOut
  // along m, where dOut is.
  //
  // copyIn cannot vectorize. A thread's four floats along n are contiguous in dIn, but MBase 0 lets
  // the swizzle permute them within their 16-byte chunk of the tile, so there is no single wide
  // store that expresses where they land. It issues one 4-byte cp.async per float instead - eight
  // per thread to cover the tile, against two for a 16-byte atom. That costs instructions, not
  // coalescing: the four floats a thread owns are the four that fill the same 128-byte line of dIn,
  // so each line is still written once, in four instructions.
  //
  // copyOut could not vectorize either way. A thread's four floats along m are contiguous in dOut,
  // but the swizzle scatters them, and no layout of the tile could be contiguous along both m and n
  // anyway. So it stays 32-bit and takes its coalescing from the thread layout instead: 32 threads
  // run down m with stride 1, so one instruction is exactly one 128-byte segment of dOut, repeated
  // eight times to cover the 32 columns. Vectorizing it would mean going through registers - read
  // the tile 32 bits at a time, write the registers back out as a float4 - for the same bytes at a
  // quarter of the instructions.
  auto copyIn = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, float>{},
      make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));
  auto copyOut = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint32_t>, float>{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{})),
      make_layout(make_shape(Int<1>{}, Int<1>{})));

  const dim3 dimBlock(size(copyIn));
  const dim3 dimGrid(size(ceil_div(m, bM)), size(ceil_div(n, bN)));

  transpose_device<<<dimGrid, dimBlock>>>(
      probShape, ctaTiler, dIn, strideIn, dOut, strideOut, smemLayout, copyIn, copyOut);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
