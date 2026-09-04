#include "softmax.cuh"

#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/host_vector.h>
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
using cute::size;
using cute::SM80_CP_ASYNC_CACHEALWAYS;
using cute::Step;
using cute::uint128_t;
using cute::UniversalCopy;
using std::runtime_error;

namespace {

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

  __shared__ float sMem[cosize_v<SmemLayoutIn>];
  auto sIn = make_tensor(make_smem_ptr(sMem), smemLayoutIn);
  auto sOut = make_tensor(make_smem_ptr(sMem), smemLayoutOut);

  auto thrCopyIn = tiledCopyIn.get_slice(threadIdx.x);
  auto tIngIn = thrCopyIn.partition_S(gIn);
  auto tInsIn = thrCopyIn.partition_D(sIn);

  auto tCsIn = local_partition(sIn, computeLayout, threadIdx.x);
  auto tCsOut = local_partition(sOut, computeLayout, threadIdx.x);
  auto tCrOut = make_tensor_like(tCsOut);

  clear(tCrOut);

  auto blockNum = size<2>(gIn);
  for (auto blockCount = 0; blockCount < blockNum; ++blockCount) {
    __syncthreads();
    copy(tiledCopyIn, tIngIn(_, _, _, blockCount), tInsIn(_, _, _));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (auto i = 0; i < size(tCsIn); ++i) {
      tCrOut[i] += expf(tCsIn[i]);
    }
  }

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
  if (laneId == 0) {
    sMem[warpId] = warpSum;
  }

  __syncthreads();
  auto warpNum = blockDim.x / 32;
  auto globalSum = 0.0f;
  for (auto i = 0; i < warpNum; ++i) {
    globalSum += sMem[i];
  }

  auto thrCopyOut = tiledCopyOut.get_slice(threadIdx.x);
  auto tOutgOut = thrCopyOut.partition_D(gOut);
  auto tOutsOut = thrCopyOut.partition_S(sOut);

  for (auto blockCount = 0; blockCount < blockNum; ++blockCount) {
    __syncthreads();
    copy(tiledCopyIn, tIngIn(_, _, _, blockCount), tInsIn(_, _, _));
    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();
    for (auto i = 0; i < size(tCsIn); ++i) {
      tCsIn[i] = expf(tCsIn[i]) / globalSum;
    }

    __syncthreads();
    copy(tiledCopyOut, tOutsOut(_, _, _), tOutgOut(_, _, _, blockCount));
  }
}

} // namespace

void softmax(int m, int n, float *dIn, int ldIn, float *dOut, int ldOut) {
  auto probShape = make_shape(m, n);

  auto strideIn = make_stride(ldIn, Int<1>{});
  auto strideOut = make_stride(ldOut, Int<1>{});

  auto bM = Int<1>{};
  auto bN = Int<512>{};
  auto ctaTiler = make_shape(bM, bN);

  auto sIn = make_layout(make_shape(bM, bN));
  auto sOut = make_layout(make_shape(bM, bN));

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

  softmax_device<<<dimGrid, dimBlock>>>(
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
