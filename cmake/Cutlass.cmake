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
endif()
