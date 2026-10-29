#include "flash_attention_upstream.cuh"

#ifdef PLAYGROUND_NO_FLASH_ATTENTION

#include <stdexcept>

// When enabled, the private upstream library supplies this symbol. The real implementation lives
// in examples/flash_attention/upstream/upstream_forward.cu - one level below the examples/*/*.cu
// glob - and is compiled by cmake/FlashAttention.cmake against upstream's own pinned CUTLASS and
// Torch, so neither reaches this example target.
void flash_attention_upstream(
    int,
    int,
    int,
    const __half *,
    int,
    const __half *,
    int,
    const __half *,
    int,
    __half *,
    int,
    float,
    bool) {
  throw std::invalid_argument(
      "official FlashAttention-2 is disabled; configure with "
      "-DPLAYGROUND_ENABLE_FLASH_ATTENTION=ON (sources come from the "
      "third_party/flash-attention submodule) and CUDA-enabled LibTorch via "
      "Torch_DIR or CMAKE_PREFIX_PATH");
}

#endif
