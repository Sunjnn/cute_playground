# Opt-in official FlashAttention-2 forward kernels, not the Python extension.
# Supply a local v2.8.3.post2 checkout/archive with its pinned CUTLASS initialized;
# this module never downloads sources or initializes submodules.

option(PLAYGROUND_ENABLE_FLASH_ATTENTION "Build the official FlashAttention-2 baseline" OFF)

# When the submodule at third_party/flash-attention is present and initialized,
# use it as the default source directory. An explicit -DFLASH_ATTENTION_DIR=...
# still overrides this.
set(_fa_default_dir "${PROJECT_SOURCE_DIR}/third_party/flash-attention")
if(NOT FLASH_ATTENTION_DIR AND EXISTS "${_fa_default_dir}/csrc/flash_attn/src/flash.h")
  set(FLASH_ATTENTION_DIR
      "${_fa_default_dir}"
      CACHE PATH "Path to official FlashAttention v2.8.3.post2 sources (auto-detected submodule)")
else()
  set(FLASH_ATTENTION_DIR
      "${FLASH_ATTENTION_DIR}"
      CACHE PATH "Path to official FlashAttention v2.8.3.post2 sources with csrc/cutlass initialized")
endif()

add_library(playground_flash_attention_upstream INTERFACE)
add_library(playground::flash_attention_upstream ALIAS playground_flash_attention_upstream)

if(NOT PLAYGROUND_ENABLE_FLASH_ATTENTION)
  target_compile_definitions(playground_flash_attention_upstream
                             INTERFACE PLAYGROUND_NO_FLASH_ATTENTION)
  message(STATUS "FlashAttention-2 disabled - upstream baseline reports itself unavailable")
  return()
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "PLAYGROUND_ENABLE_FLASH_ATTENTION=ON requires a CUDA compiler and toolkit 12.8+")
endif()
if(CMAKE_CUDA_COMPILER_VERSION AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "FlashAttention-2 on sm_120 requires CUDA compiler 12.8+ "
    "(found ${CMAKE_CUDA_COMPILER_VERSION})")
endif()
if(NOT CUDAToolkit_FOUND OR NOT TARGET CUDA::cudart)
  message(FATAL_ERROR
    "PLAYGROUND_ENABLE_FLASH_ATTENTION=ON requires the CUDA toolkit and CUDA::cudart")
endif()
if(NOT CUDAToolkit_VERSION OR CUDAToolkit_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "FlashAttention-2 on sm_120 requires CUDA toolkit 12.8+ (found ${CUDAToolkit_VERSION})")
endif()

if(NOT FLASH_ATTENTION_DIR OR NOT IS_DIRECTORY "${FLASH_ATTENTION_DIR}")
  message(FATAL_ERROR
    "PLAYGROUND_ENABLE_FLASH_ATTENTION=ON cannot find official FlashAttention "
    "v2.8.3.post2 sources. Initialize the submodule with:\n"
    "  git submodule update --init --recursive\n"
    "or point -DFLASH_ATTENTION_DIR=<path> at an existing checkout. "
    "CMake never downloads sources itself.")
endif()
set(_fa_src "${FLASH_ATTENTION_DIR}/csrc/flash_attn/src")
set(_fa_cutlass "${FLASH_ATTENTION_DIR}/csrc/cutlass")
set(_fa_version_file "${FLASH_ATTENTION_DIR}/flash_attn/__init__.py")
if(NOT EXISTS "${_fa_version_file}")
  message(FATAL_ERROR "Missing FlashAttention version file: ${_fa_version_file}")
endif()
file(STRINGS "${_fa_version_file}" _fa_version_line REGEX "^__version__[ \t]*=")
if(NOT _fa_version_line MATCHES "^__version__[ \t]*=[ \t]*[\"']([^\"']+)[\"']"
   OR NOT CMAKE_MATCH_1 STREQUAL "2.8.3.post2")
  message(FATAL_ERROR
    "FLASH_ATTENTION_DIR must contain official FlashAttention version 2.8.3.post2")
endif()

set(_fa_kernel_names
    flash_fwd_hdim64_fp16_sm80.cu
    flash_fwd_hdim64_fp16_causal_sm80.cu
    flash_fwd_hdim128_fp16_sm80.cu
    flash_fwd_hdim128_fp16_causal_sm80.cu)
set(_fa_kernel_sources)
foreach(_fa_kernel IN LISTS _fa_kernel_names)
  if(NOT EXISTS "${_fa_src}/${_fa_kernel}")
    message(FATAL_ERROR "Missing official FlashAttention forward source: ${_fa_src}/${_fa_kernel}")
  endif()
  list(APPEND _fa_kernel_sources "${_fa_src}/${_fa_kernel}")
endforeach()
foreach(_fa_header flash.h flash_fwd_launch_template.h)
  if(NOT EXISTS "${_fa_src}/${_fa_header}")
    message(FATAL_ERROR "Missing official FlashAttention forward header: ${_fa_src}/${_fa_header}")
  endif()
endforeach()
foreach(_fa_header cutlass/version.h cutlass/numeric_types.h cute/tensor.hpp)
  if(NOT EXISTS "${_fa_cutlass}/include/${_fa_header}")
    message(FATAL_ERROR
      "FlashAttention's own csrc/cutlass submodule is not initialized; "
      "missing ${_fa_cutlass}/include/${_fa_header}. Run:\n"
      "  git submodule update --init --recursive\n"
      "This must be FlashAttention's pinned CUTLASS, not the playground's "
      "third_party/cutlass.")
  endif()
endforeach()
file(READ "${_fa_cutlass}/include/cutlass/version.h" _fa_cutlass_version)
foreach(_fa_component MAJOR MINOR PATCH)
  if(_fa_component STREQUAL "MAJOR")
    set(_fa_expected_component 4)
  else()
    set(_fa_expected_component 0)
  endif()
  if(NOT _fa_cutlass_version MATCHES
     "#[ \t]*define[ \t]+CUTLASS_${_fa_component}[ \t]+([0-9]+)"
     OR NOT CMAKE_MATCH_1 STREQUAL "${_fa_expected_component}")
    message(FATAL_ERROR "FlashAttention v2.8.3.post2 requires its own pinned CUTLASS version 4.0.0")
  endif()
endforeach()

# Do not accidentally inspect an enclosing playground Git repository for an
# archive. Only a checkout's own .git metadata can establish its provenance.
if(EXISTS "${FLASH_ATTENTION_DIR}/.git" OR EXISTS "${_fa_cutlass}/.git")
  find_package(Git REQUIRED)
endif()
if(EXISTS "${FLASH_ATTENTION_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${FLASH_ATTENTION_DIR}" rev-parse --verify HEAD
    RESULT_VARIABLE _fa_git_result
    OUTPUT_VARIABLE _fa_head
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT _fa_git_result EQUAL 0)
    message(FATAL_ERROR "Cannot read Git HEAD for FLASH_ATTENTION_DIR")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${FLASH_ATTENTION_DIR}" rev-parse
            --verify "refs/tags/v2.8.3.post2^{commit}"
    RESULT_VARIABLE _fa_tag_result
    OUTPUT_VARIABLE _fa_tag_head
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_fa_tag_result EQUAL 0)
    if(NOT _fa_head STREQUAL _fa_tag_head)
      message(FATAL_ERROR
        "FlashAttention Git HEAD is not the local v2.8.3.post2 tag; use that exact release")
    endif()
  else()
    message(WARNING
      "FlashAttention tag v2.8.3.post2 is not available locally; "
      "only the declared source version was verified. No tag is fetched.")
  endif()
else()
  message(WARNING
    "FlashAttention sources have no Git metadata; "
    "verified version 2.8.3.post2 but cannot verify the release tag.")
endif()
if(EXISTS "${_fa_cutlass}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${_fa_cutlass}" rev-parse --verify HEAD
    RESULT_VARIABLE _fa_git_result
    OUTPUT_VARIABLE _fa_cutlass_head
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT _fa_git_result EQUAL 0 OR
     NOT _fa_cutlass_head STREQUAL "dc4817921edda44a549197ff3a9dcf5df0636e7b")
    message(FATAL_ERROR
      "FlashAttention's CUTLASS Git HEAD must be "
      "dc4817921edda44a549197ff3a9dcf5df0636e7b (version 4.0.0)")
  endif()
else()
  message(WARNING
    "FlashAttention's CUTLASS has no Git metadata; verified version 4.0.0 "
    "but cannot verify commit dc4817921edda44a549197ff3a9dcf5df0636e7b.")
endif()

# Torch supplies the small ATen/c10 host-side facilities used by flash.h and
# the launch templates. Do not build flash_api.cpp, pybind11, or Python bindings.
# QUIET + our own diagnostic: the stock "Could not find a package configuration
# file provided by Torch" says nothing about why an attention example needs
# PyTorch or how to get a CUDA-enabled one.
find_package(Torch CONFIG QUIET)
if(NOT Torch_FOUND)
  message(FATAL_ERROR
    "PLAYGROUND_ENABLE_FLASH_ATTENTION=ON needs CUDA-enabled LibTorch, because "
    "upstream FlashAttention-2 headers hard-depend on ATen/C10 (Philox state and "
    "CUDA error checks) even with dropout disabled.\n"
    "Install one of:\n"
    "  - CUDA PyTorch: pip install torch --index-url "
    "https://download.pytorch.org/whl/cu128, then configure with "
    "-DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; "
    "print(torch.utils.cmake_prefix_path)')\n"
    "  - standalone LibTorch (CUDA build) from pytorch.org/get-started/locally, "
    "then configure with -DTorch_DIR=<libtorch>/share/cmake/Torch\n"
    "A CPU-only Torch package will be rejected later (no c10_cuda). To build "
    "without Torch, drop -DPLAYGROUND_ENABLE_FLASH_ATTENTION=ON; the upstream "
    "baseline row then reports n/a and everything else still builds.")
endif()
if(NOT TORCH_INCLUDE_DIRS OR NOT TORCH_LIBRARIES)
  message(FATAL_ERROR "Torch must provide TORCH_INCLUDE_DIRS and TORCH_LIBRARIES")
endif()
foreach(_fa_header ATen/cuda/CUDAGeneratorImpl.h c10/cuda/CUDAException.h
                   ATen/cuda/detail/UnpackRaw.cuh)
  set(_fa_header_found FALSE)
  foreach(_fa_include IN LISTS TORCH_INCLUDE_DIRS)
    if(EXISTS "${_fa_include}/${_fa_header}")
      set(_fa_header_found TRUE)
      break()
    endif()
  endforeach()
  if(NOT _fa_header_found)
    message(FATAL_ERROR "The CUDA-enabled Torch package must provide ${_fa_header}")
  endif()
endforeach()
if(TARGET c10_cuda)
  set(_fa_c10_cuda c10_cuda)
else()
  # Never accept an unrelated system c10_cuda for a CPU-only Torch package.
  unset(FLASH_ATTENTION_C10_CUDA_LIBRARY CACHE)
  unset(FLASH_ATTENTION_C10_CUDA_LIBRARY)
  if(TORCH_INSTALL_PREFIX)
    find_library(FLASH_ATTENTION_C10_CUDA_LIBRARY NAMES c10_cuda
                 PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
  endif()
  if(NOT FLASH_ATTENTION_C10_CUDA_LIBRARY)
    message(FATAL_ERROR
      "FlashAttention requires a CUDA-enabled Torch package with c10_cuda "
      "(CPU-only Torch is unsupported)")
  endif()
  set(_fa_c10_cuda "${FLASH_ATTENTION_C10_CUDA_LIBRARY}")
  mark_as_advanced(FLASH_ATTENTION_C10_CUDA_LIBRARY)
endif()

add_library(playground_flash_attention_upstream_kernels STATIC
            ${_fa_kernel_sources}
            "${PROJECT_SOURCE_DIR}/examples/flash_attention/upstream/upstream_forward.cu")
set_target_properties(playground_flash_attention_upstream_kernels PROPERTIES
                      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                      CUDA_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
# Inherit the root's CUDA architectures (native/sm_120); these filenames name
# upstream's kernel family, not a request to override the target architecture.
target_include_directories(playground_flash_attention_upstream_kernels PRIVATE
                           "${PROJECT_SOURCE_DIR}/examples/flash_attention"
                           "${_fa_src}" "${_fa_cutlass}/include")
target_include_directories(playground_flash_attention_upstream_kernels SYSTEM PRIVATE
                           ${TORCH_INCLUDE_DIRS})
# Link only c10/c10_cuda instead of every LibTorch library. The forward path
# needs Torch headers plus c10_cuda's error checks; torch_cuda.dll is never
# called. On Windows this keeps the runtime CUDA dependency to cudart only.
option(PLAYGROUND_FA_LINK_MINIMAL_TORCH
       "Link only c10/c10_cuda for the upstream FA2 kernels (not all of LibTorch)" OFF)
if(PLAYGROUND_FA_LINK_MINIMAL_TORCH)
  if(TARGET c10)
    set(_fa_c10 c10)
  else()
    find_library(FLASH_ATTENTION_C10_LIBRARY NAMES c10
                 PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
    if(NOT FLASH_ATTENTION_C10_LIBRARY)
      message(FATAL_ERROR "Cannot find the c10 library for PLAYGROUND_FA_LINK_MINIMAL_TORCH")
    endif()
    set(_fa_c10 "${FLASH_ATTENTION_C10_LIBRARY}")
    mark_as_advanced(FLASH_ATTENTION_C10_LIBRARY)
  endif()
  target_link_libraries(playground_flash_attention_upstream_kernels PRIVATE
                        ${_fa_c10_cuda} ${_fa_c10} CUDA::cudart)
else()
  target_link_libraries(playground_flash_attention_upstream_kernels PRIVATE
                        ${TORCH_LIBRARIES} ${_fa_c10_cuda} CUDA::cudart)
endif()

# Apply exported Torch ABI definitions to both nvcc passes, and forward any
# host-only options with -Xcompiler. TORCH_CXX_FLAGS may be absent on platforms
# without Linux's dual ABI; an absent value is empty. These settings stay private.
separate_arguments(_fa_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
foreach(_fa_flag IN LISTS _fa_torch_flags)
  if(_fa_flag MATCHES "^[-/]D(.+)")
    target_compile_definitions(playground_flash_attention_upstream_kernels
                               PRIVATE "${CMAKE_MATCH_1}")
  else()
    target_compile_options(playground_flash_attention_upstream_kernels PRIVATE
                           "$<$<COMPILE_LANGUAGE:CXX>:${_fa_flag}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_fa_flag}>")
  endif()
endforeach()
set(_fa_cuda_flags -O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
                   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
                   -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__)
target_compile_options(playground_flash_attention_upstream_kernels PRIVATE
                       "$<$<COMPILE_LANGUAGE:CUDA>:${_fa_cuda_flags}>")
# MSVC reports __cplusplus as 199711L unless /Zc:__cplusplus is enabled. CUTLASS
# gates its C++17 platform traits on that macro, so the nvcc host pass otherwise
# cannot find cutlass::platform::is_unsigned_v (and friends).
if(MSVC)
  target_compile_options(playground_flash_attention_upstream_kernels PRIVATE
                         "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>")
endif()
# CUTLASS 4.0's cuda_host_adapter.hpp marks a template that calls a host-only
# virtual as __host__ __device__; CMake's --Werror cross-execution-space-call
# turns that diagnostic into a hard error. Suppress only that diagnostic.
target_compile_options(playground_flash_attention_upstream_kernels PRIVATE
                       "$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe;--diag_suppress=20011>")
# Only dense, full/causal fp16 D=64/128 forward attention is dispatched by the
# bridge. Pruning these branches is safe for that deliberately narrow contract.
target_compile_definitions(playground_flash_attention_upstream_kernels PRIVATE
                           FLASHATTENTION_DISABLE_DROPOUT
                           FLASHATTENTION_DISABLE_ALIBI
                           FLASHATTENTION_DISABLE_SOFTCAP
                           FLASHATTENTION_DISABLE_LOCAL
                           FLASHATTENTION_DISABLE_UNEVEN_K)
target_link_libraries(playground_flash_attention_upstream INTERFACE
                      playground_flash_attention_upstream_kernels)
message(STATUS "FlashAttention-2 enabled: official v2.8.3.post2, "
               "private CUTLASS 4.0.0, fp16 D=64/128 full/causal forward")
