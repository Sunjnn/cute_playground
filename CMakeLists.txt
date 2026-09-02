cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

project(cute_playground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# CUDA is optional: examples in *.cu are only built when a CUDA toolkit is
# available on the machine.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Pick the arch default before enable_language(), which would otherwise
  # initialize the variable to the compiler's default architecture.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      # 'native' needs CMake 3.24; query the driver instead ("12.0" -> 120).
      execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        RESULT_VARIABLE smiResult
        OUTPUT_VARIABLE computeCap
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(smiResult EQUAL 0 AND computeCap MATCHES "^([0-9]+\\.[0-9]+)")
        string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_0}")
      else()
        set(CMAKE_CUDA_ARCHITECTURES 120)
      endif()
    endif()
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} "
                 "(archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "CUDA toolkit not found - skipping .cu examples")
endif()

include(Cutlass)

option(BUILD_EXAMPLES "Build CuTe learning examples" ON)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
