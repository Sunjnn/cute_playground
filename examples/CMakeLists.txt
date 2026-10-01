# All examples require a CUDA toolkit: CuTe headers include CUDA runtime
# headers unconditionally, so even host-only .cpp examples need them.
# Without nvcc this directory contributes no targets.

if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "No CUDA toolkit - skipping all examples")
  return()
endif()

# One executable per example subdirectory. A single example may span several
# translation units (e.g. softmax/ = main.cpp + softmax.cu + softmax.cuh), so
# every .cpp (host) and .cu (CUDA) file in the folder is compiled into one
# target named after the folder. Headers are found via the folder include dir.

# Discover example folders from their sources and group the sources per folder.
file(
  GLOB example_srcs
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS */*.cpp */*.cu)

set(example_dirs "")
foreach(src ${example_srcs})
  get_filename_component(dir ${src} DIRECTORY)
  list(APPEND example_dirs ${dir})
  list(APPEND example_srcs_${dir} ${src})
endforeach()
list(REMOVE_DUPLICATES example_dirs)

foreach(dir ${example_dirs})
  add_executable(${dir} ${example_srcs_${dir}})
  # playground::cudnn and playground::cutensor are linked everywhere, not just where they are used,
  # to keep this loop uniform. Each is an INTERFACE target that either supplies its library's
  # include path and library or defines PLAYGROUND_NO_<LIB>, so it costs nothing when the
  # dependency is absent. CUDA::cublas is unconditional: cmake/Cutlass.cmake runs
  # find_package(CUDAToolkit REQUIRED) whenever a CUDA compiler was found, and without one this
  # directory returns early above.
  target_link_libraries(
    ${dir} PRIVATE cutlass::cute playground::cudnn playground::cutensor CUDA::cublas)
  target_include_directories(${dir} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  target_compile_options(
    ${dir} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                    --expt-extended-lambda>)
endforeach()

# cuTENSOR 2.7's host-side plan creation overflows the linker's default 1 MiB main-thread stack on
# sm_120 and dies with STATUS_STACK_BUFFER_OVERRUN inside cutensorCreatePlan (verified on an RTX
# 5060, CUDA 13.3). transpose builds one plan per call and rmsnorm_cutensor builds five, so both
# need the larger stack; only these two call cuTENSOR. 8 MiB held for every shape in the transpose
# README sweep up to 16384 x 16384.
#
# /STACK is link.exe's, hence the MSVC guard: the flag is passed straight to the host linker, and
# ld.bfd rejects it as a missing input file. Elsewhere the main-thread stack comes from ulimit -s,
# which is 8 MiB by default anyway.
if(MSVC)
  foreach(cutensor_user transpose rmsnorm)
    target_link_options(
      ${cutensor_user}
      PRIVATE $<$<LINK_LANG_AND_ID:CUDA,NVIDIA>:-Xlinker;/STACK:8388608>)
  endforeach()
endif()

