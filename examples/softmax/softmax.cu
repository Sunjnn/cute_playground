#include "softmax.cuh"

#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/container/array_subbyte.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor.hpp" // IWYU pragma: keep
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include "cutlass/uint128.h"

using cute::_;
using cute::ceil_div;
using cute::clear;
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
using cute::make_tensor_like;
using cute::make_tiled_copy;
using cute::shape;
using cute::size;
using cute::SM80_CP_ASYNC_CACHEALWAYS;
using cute::Step;
using cute::uint128_t;
using cute::UniversalCopy;
using std::runtime_error;

namespace {

template <
    int kBlockNumOnFlight,
    class TensortCsIn,
    class TensortOutsOut,
    class TensortOutgOut,
    class TiledCopyOut,
    class BlockIndex,
    class PipeRead,
    class PipeNum,
    class GlobalSum>
__device__ void epilogue(
    TensortCsIn &tCsIn,
    TensortOutsOut &tOutsOut,
    TensortOutgOut &tOutgOut,
    const TiledCopyOut &tiledCopyOut,
    BlockIndex &blockIndex,
    PipeRead &pipeRead,
    const PipeNum &pipeNum,
    const GlobalSum &globalSum) {
  if constexpr (kBlockNumOnFlight == 0) {
    return;
  } else {
    cp_async_wait<kBlockNumOnFlight - 1>();

    __syncthreads();
    auto tCsInPipe = tCsIn(_, _, pipeRead);
    for (auto i = 0; i < size(tCsInPipe); ++i) {
      tCsInPipe[i] = expf(tCsInPipe[i]) / globalSum;
    }

    __syncthreads();
    auto tOutsOutPipe = tOutsOut(_, _, _, pipeRead);
    copy(tiledCopyOut, tOutsOutPipe, tOutgOut(_, _, _, blockIndex));

    pipeRead = (pipeRead + 1) == pipeNum ? 0 : pipeRead + 1;
    ++blockIndex;
    epilogue<kBlockNumOnFlight - 1>(
        tCsIn, tOutsOut, tOutgOut, tiledCopyOut, blockIndex, pipeRead, pipeNum, globalSum);
  }
}

template <class TiledCopyIn, class TensortIngIn, class TensortInsIn, class PipeNum>
__device__ void prefetch_block(
    const TiledCopyIn &tiledCopyIn,
    const TensortIngIn &tIngIn,
    TensortInsIn &tInsIn,
    int blockIndex,
    int blockNum,
    const PipeNum &pipeNum,
    int pipeRead) {
  auto blockIndexOnFlight = blockIndex + pipeNum;
  blockIndexOnFlight =
      blockIndexOnFlight >= blockNum ? blockIndexOnFlight - blockNum : blockIndexOnFlight;
  copy(tiledCopyIn, tIngIn(_, _, _, blockIndexOnFlight), tInsIn(_, _, _, pipeRead));
  cp_async_fence();
}

template <class TensortCrOut, class TensortReducesPipe>
__device__ float block_reduce_sum(const TensortCrOut &tCrOut, TensortReducesPipe &tReducesPipe) {
  auto thrSum = 0.0f;
  for (auto i = 0; i < size(tCrOut); ++i) {
    thrSum += tCrOut[i];
  }
  auto warpSum = thrSum;
  for (auto offset = 16; offset > 0; offset /= 2) {
    warpSum += __shfl_xor_sync(0xffffffff, warpSum, offset);
  }

  __syncthreads();
  auto warpId = threadIdx.x / 32;
  auto laneId = threadIdx.x % 32;
  // tReducesPipe must be thread 0's slice of a pipe the caller is about to refill
  if (laneId == 0) {
    tReducesPipe[warpId] = warpSum;
  }

  __syncthreads();
  auto warpNum = blockDim.x / 32;
  auto globalSum = 0.0f;
  for (auto i = 0; i < warpNum; ++i) {
    globalSum += tReducesPipe[i];
  }

  __syncthreads();
  return globalSum;
}

template <
    class ProblemShape,
    class CtaTiler,
    class StrideIn,
    class SmemLayoutIn,
    class TiledCopyIn,
    class StrideOut,
    class SmemlayoutOut,
    class TiledCopyOut,
    class ComputeLayout>
__global__ void softmax_device(
    ProblemShape probShape,
    CtaTiler ctaTiler,
    const float *dIn,
    StrideIn strideIn,
    SmemLayoutIn smemLayoutIn,
    TiledCopyIn tiledCopyIn,
    float *dOut,
    StrideOut strideOut,
    SmemlayoutOut smemLayoutOut,
    TiledCopyOut tiledCopyOut,
    ComputeLayout computeLayout) {
  auto mIn = make_tensor(make_gmem_ptr(dIn), probShape, strideIn);
  auto mOut = make_tensor(make_gmem_ptr(dOut), probShape, strideOut);

  auto ctaCoord = make_coord(blockIdx.x, _);
  auto gIn = local_tile(mIn, ctaTiler, ctaCoord);
  auto gOut = local_tile(mOut, ctaTiler, ctaCoord);

  extern __shared__ float sMem[];
  auto sIn = make_tensor(make_smem_ptr(sMem), smemLayoutIn);
  auto sOut = make_tensor(make_smem_ptr(sMem), smemLayoutOut);

  auto thrCopyIn = tiledCopyIn.get_slice(threadIdx.x);
  auto tIngIn = thrCopyIn.partition_S(gIn);
  auto tInsIn = thrCopyIn.partition_D(sIn);

  constexpr auto kPipeNum = shape<2>(smemLayoutIn);
  for (auto pipeIndex = 0; pipeIndex < kPipeNum; ++pipeIndex) {
    copy(tiledCopyIn, tIngIn(_, _, _, pipeIndex), tInsIn(_, _, _, pipeIndex));
    cp_async_fence();
  }

  auto pipeRead = 0;

  auto tCsIn = local_partition(sIn, computeLayout, threadIdx.x);
  auto tReducesIn = local_partition(sIn, computeLayout, 0);
  auto tCsOut = local_partition(sOut, computeLayout, threadIdx.x);
  auto tCrOut = make_tensor_like(tCsOut(_, _, 0));

  clear(tCrOut);

  const auto blockNum = size<2>(gIn);
  auto blockIndex = 0;
  for (; blockIndex < blockNum - 1; ++blockIndex) {
    cp_async_wait<kPipeNum - 1>();

    __syncthreads();
    auto tCsInPipe = tCsIn(_, _, pipeRead);
    for (auto i = 0; i < size(tCsInPipe); ++i) {
      tCrOut[i] += expf(tCsInPipe[i]);
    }

    __syncthreads();
    prefetch_block(tiledCopyIn, tIngIn, tInsIn, blockIndex, blockNum, kPipeNum, pipeRead);

    pipeRead = (pipeRead + 1) == kPipeNum ? 0 : pipeRead + 1;
  }

  cp_async_wait<kPipeNum - 1>();

  __syncthreads();
  auto tCsInPipe = tCsIn(_, _, pipeRead);
  for (auto i = 0; i < size(tCsInPipe); ++i) {
    tCrOut[i] += expf(tCsInPipe[i]);
  }

  auto tReducesPipe = tReducesIn(_, _, pipeRead);
  auto globalSum = block_reduce_sum(tCrOut, tReducesPipe);

  prefetch_block(tiledCopyIn, tIngIn, tInsIn, blockIndex, blockNum, kPipeNum, pipeRead);

  pipeRead = (pipeRead + 1) == kPipeNum ? 0 : pipeRead + 1;

  auto thrCopyOut = tiledCopyOut.get_slice(threadIdx.x);
  auto tOutgOut = thrCopyOut.partition_D(gOut);
  auto tOutsOut = thrCopyOut.partition_S(sOut);

  blockIndex = 0;
  for (; blockIndex < blockNum - kPipeNum; ++blockIndex) {
    cp_async_wait<kPipeNum - 1>();

    __syncthreads();
    auto tCsInPipe = tCsIn(_, _, pipeRead);
    for (auto i = 0; i < size(tCsInPipe); ++i) {
      tCsInPipe[i] = expf(tCsInPipe[i]) / globalSum;
    }

    __syncthreads();
    auto tOutsOutPipe = tOutsOut(_, _, _, pipeRead);
    copy(tiledCopyOut, tOutsOutPipe, tOutgOut(_, _, _, blockIndex));

    __syncthreads();
    prefetch_block(tiledCopyIn, tIngIn, tInsIn, blockIndex, blockNum, kPipeNum, pipeRead);

    pipeRead = (pipeRead + 1) == kPipeNum ? 0 : pipeRead + 1;
  }

  epilogue<kPipeNum>(
      tCsIn, tOutsOut, tOutgOut, tiledCopyOut, blockIndex, pipeRead, kPipeNum, globalSum);
}

} // namespace

void softmax(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  auto probShape = make_shape(m, n);

  auto strideIn = make_stride(ldIn, Int<1>{});
  auto strideOut = make_stride(ldOut, Int<1>{});

  auto bM = Int<1>{};
  auto bN = Int<2048>{};
  auto ctaTiler = make_shape(bM, bN);
  auto bP = Int<4>{};

  auto sIn = make_layout(make_shape(bM, bN, bP));
  auto sOut = make_layout(make_shape(bM, bN, bP));

  auto copyIn = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{},
      make_layout(make_shape(Int<1>{}, Int<128>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));
  auto copyOut = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, float>{},
      make_layout(make_shape(Int<1>{}, Int<128>{})),
      make_layout(make_shape(Int<1>{}, Int<4>{})));

  auto computeLayout = make_layout(make_shape(Int<1>{}, Int<128>{}));

  const dim3 dimBlock(size(computeLayout));
  const dim3 dimGrid(size(ceil_div(m, bM)));

  softmax_device<<<dimGrid, dimBlock, cosize(sIn) * sizeof(float), nullptr>>>(
      probShape,
      ctaTiler,
      dIn,
      strideIn,
      sIn,
      copyIn,
      dOut,
      strideOut,
      sOut,
      copyOut,
      computeLayout);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw runtime_error("");
  }
}
