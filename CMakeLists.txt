cmake_minimum_required(VERSION 3.24)

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
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
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
