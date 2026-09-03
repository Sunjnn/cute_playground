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
  target_link_libraries(${dir} PRIVATE cutlass::cute)
  target_include_directories(${dir} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  target_compile_options(
    ${dir} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                    --expt-extended-lambda>)
endforeach()
