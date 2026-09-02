# All examples require a CUDA toolkit: CuTe headers include CUDA runtime
# headers unconditionally, so even host-only .cpp examples need them.
# Without nvcc this directory contributes no targets.

if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "No CUDA toolkit - skipping all examples")
  return()
endif()

# One executable per example file.
#   *.cpp -> host-only examples (compiled by the host compiler)
#   *.cu  -> CUDA examples (compiled by nvcc)

file(
  GLOB host_examples
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS *.cpp)
foreach(src ${host_examples})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE cutlass::cute)
endforeach()

file(
  GLOB cuda_examples
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS *.cu)
foreach(src ${cuda_examples})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE cutlass::cute)
  target_compile_options(
    ${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
                    --expt-extended-lambda>)
endforeach()
