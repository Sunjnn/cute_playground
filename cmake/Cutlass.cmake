# CuTe/CUTLASS headers come from the git submodule at third_party/cutlass.
# Only the headers are used; CUTLASS' own CMake project (which requires a
# CUDA toolkit) is never added as a subdirectory.

set(CUTLASS_DIR
    "${PROJECT_SOURCE_DIR}/third_party/cutlass"
    CACHE PATH "Path to the CUTLASS submodule")

if(NOT EXISTS "${CUTLASS_DIR}/include/cute/layout.hpp")
  message(
    FATAL_ERROR
      "CUTLASS submodule not found at ${CUTLASS_DIR}.\n"
      "Run: git submodule update --init --recursive")
endif()

add_library(cutlass_cute INTERFACE)
add_library(cutlass::cute ALIAS cutlass_cute)

target_include_directories(
  cutlass_cute INTERFACE ${CUTLASS_DIR}/include
                         ${CUTLASS_DIR}/tools/util/include)

target_compile_features(cutlass_cute INTERFACE cxx_std_17)

# Host-only .cpp examples still need the real CUDA headers (CuTe includes
# cuda_runtime_api.h / vector_types.h unconditionally).
if(CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(cutlass_cute INTERFACE
                             ${CUDAToolkit_INCLUDE_DIRS})
  # CUDA 13 moved thrust/cub/cuda::std under <toolkit>/include/cccl, which
  # nvcc adds implicitly and CMake therefore strips from the generated
  # commands. Pass it as a flag so compile_commands.json (clangd /
  # clang-tidy) also sees the directory.
  # CUDAToolkit_INCLUDE_DIRS is a list; build the path from its first entry so
  # the result is a single well-formed -isystem flag rather than a stray token.
  list(GET CUDAToolkit_INCLUDE_DIRS 0 cutlass_cuda_inc)
  target_compile_options(
    cutlass_cute
    INTERFACE $<$<COMPILE_LANGUAGE:CUDA,CXX>:-isystem${cutlass_cuda_inc}/cccl>)
endif()
