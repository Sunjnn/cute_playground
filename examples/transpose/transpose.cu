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
#include "cute/config.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/math.hpp"
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
using cute::get;
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

// Maps a linear CTA id onto a (tile m, tile n) coordinate, so that the CTAs resident at any
// moment cover a compact square of both tensors rather than one long strip of each.
//
// With a plain (tilesM, tilesN) grid and blockIdx.x fastest over m, the ~360 CTAs in flight read
// 128-byte fragments strided 128 * ldIn apart across the whole of dIn while writing one tight
// strip of dOut. Against a GDDR7 row buffer of roughly 1 KB every one of those reads is a page
// miss. Grouping the tiles into supertiles and walking n fastest inside a group turns each row of
// a supertile into a run of groupN * 128 contiguous bytes of dIn, and walking supertiles along n
// next extends that run, so both tensors see runs of at least a page. Neither side can be made
// fully contiguous - that is what makes this a transpose - but neither has to be scattered at
// 128-byte granularity either.
//
// Group sizes are clamped to the tile count on each axis, so a matrix narrower than Group tiles
// does not pad the grid with empty CTAs. Where a tile count is not a multiple of its group size
// the grid is rounded up to whole supertiles and the CTAs past the edge exit immediately. That
// exit is block-uniform, since coord() reads only blockIdx.x, so no CTA can return out of a
// __syncthreads the rest of it is waiting at.
template <int Group> struct CtaSwizzle {
  int tilesM;
  int tilesN;
  int groupM;
  int groupN;
  int groupsM;
  int groupsN;

  CUTE_HOST_DEVICE CtaSwizzle(int tilesM, int tilesN)
  : tilesM(tilesM)
  , tilesN(tilesN)
  , groupM(cute::min(Group, tilesM))
  , groupN(cute::min(Group, tilesN))
  , groupsM(ceil_div(tilesM, groupM))
  , groupsN(ceil_div(tilesN, groupN)) {}

  CUTE_HOST_DEVICE int size() const { return groupsM * groupsN * groupM * groupN; }

  // Axis order, fastest first: tile n, tile m, supertile n, supertile m.
  CUTE_HOST_DEVICE auto coord(int id) const {
    const auto perGroup = groupM * groupN;
    const auto within = id % perGroup;
    const auto group = id / perGroup;
    return make_coord(
        within / groupN + (group / groupsN) * groupM, within % groupN + (group % groupsN) * groupN);
  }

  CUTE_HOST_DEVICE bool contains(int tileM, int tileN) const {
    return tileM < tilesM && tileN < tilesN;
  }
};

template <
    class ProblemShape,
    class CtaTiler,
    class CtaMap,
    class StrideIn,
    class StrideOut,
    class SmemLayout,
    class TiledCopyIn,
    class TiledCopyOut>
__global__ void transpose_device(
    ProblemShape probShape,
    CtaTiler ctaTiler,
    CtaMap ctaMap,
    const float *dIn,
    StrideIn strideIn,
    float *dOut,
    StrideOut strideOut,
    SmemLayout smemLayout,
    TiledCopyIn tiledCopyIn,
    TiledCopyOut tiledCopyOut) {
  const auto mIn = make_tensor(make_gmem_ptr(dIn), probShape, strideIn);
  auto mOut = make_tensor(make_gmem_ptr(dOut), probShape, strideOut);

  const auto ctaCoord = ctaMap.coord(blockIdx.x);
  if (!ctaMap.contains(get<0>(ctaCoord), get<1>(ctaCoord))) {
    return;
  }
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

  // The stores below are plain st.global, not cp.async, so there is no group to fence and nothing
  // for a wait to release - kernel exit is what the host's synchronize waits on.
  copy(tiledCopyOut, tOutsTile, tOutgOut);
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
  // copyIn cannot vectorize. Widening its value layout to four contiguous floats along n would put
  // four adjacent floats of dIn in one thread, and those are contiguous in dIn - but MBase 0 lets
  // the swizzle permute them within their 16-byte chunk of the tile, so there is no single wide
  // store that expresses where they land. It issues one 4-byte cp.async per float instead - eight
  // per thread to cover the tile, against two for a 16-byte atom.
  //
  // What the thread layout buys in place of that is one warp-instruction per 128-byte line of dIn.
  // All 32 lanes run along n with stride 1, so lane k and lane k + 1 read adjacent floats and the
  // longest run of consecutive floats in an instruction is the full 32. The layout only spans 4 of
  // the tile's 32 rows, so the repeat count along m is 8 and a thread's eight instructions step
  // down in strides of 4. The narrower shapes each lose somewhere: with 8 lanes along n every
  // instruction spans 4 rows and 4 lines however the values are grouped, and handing one thread the
  // four floats on top of that also isolates each lane - longest run 1, and 16 sectors touched per
  // instruction against 4.
  //
  // The same layout is what keeps the shared-memory write conflict-free. Under Swizzle<5, 0, 5> the
  // write bank is n ^ m, and with n spanning all 32 lanes that xor reaches all 32 banks. Restrict n
  // to 8 and a warp carries m over 0..3 against n over 0..7, so n ^ m collapses onto 8 banks and
  // the write becomes a 4-way conflict.
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
      make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<1>{})));
  auto copyOut = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint32_t>, float>{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{})),
      make_layout(make_shape(Int<1>{}, Int<1>{})));

  // Supertiles of 8 x 8 CTAs: eight tiles along n is 8 * 32 * 4 = 1 KB of dIn per row, about one
  // GDDR7 page, which is the granularity the read side was missing at 128 bytes.
  const auto ctaMap = CtaSwizzle<8>{size(ceil_div(m, bM)), size(ceil_div(n, bN))};

  const dim3 dimBlock(size(copyIn));
  const dim3 dimGrid(ctaMap.size());

  transpose_device<<<dimGrid, dimBlock>>>(
      probShape, ctaTiler, ctaMap, dIn, strideIn, dOut, strideOut, smemLayout, copyIn, copyOut);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}
