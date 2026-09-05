# cuDNN is optional and only examples/softmax/cudnn_softmax.cu uses it. When the library cannot be
# found, playground::cudnn carries PLAYGROUND_NO_CUDNN instead of the real include paths and
# library, which turns softmax_cudnn() into a stub that reports itself unavailable in the benchmark
# table. Building the examples therefore never requires cuDNN.
#
# Set CUDNN_ROOT to an installation that is not next to the CUDA toolkit - an unpacked .deb, a pip
# wheel, a vendor drop - and the search below picks it up.

set(CUDNN_ROOT
    ""
    CACHE PATH "Root of a cuDNN installation, containing include/ and lib/")

# pip-wheel-style installs nest everything under a CUDA-versioned layer:
# include/13.3/cudnn.h, lib/13.3/x64/cudnn.lib. Collect those subdirectories as
# additional hint roots so both flat and versioned layouts are found.
set(CUDNN_INCLUDE_HINTS)
set(CUDNN_LIB_HINTS)
if(CUDNN_ROOT)
  file(GLOB CUDNN_INCLUDE_HINTS "${CUDNN_ROOT}/include/*")
  file(GLOB CUDNN_LIB_HINTS "${CUDNN_ROOT}/lib/*" "${CUDNN_ROOT}/lib/*/x64")
endif()

find_path(
  CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDNN_ROOT} ${CUDNN_INCLUDE_HINTS} ENV CUDNN_ROOT
  PATH_SUFFIXES include)
find_library(
  CUDNN_LIBRARY
  NAMES cudnn
  HINTS ${CUDNN_ROOT} ${CUDNN_LIB_HINTS} ENV CUDNN_ROOT
  PATH_SUFFIXES lib lib64)

add_library(playground_cudnn INTERFACE)
add_library(playground::cudnn ALIAS playground_cudnn)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
  message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
  target_include_directories(playground_cudnn INTERFACE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(playground_cudnn INTERFACE ${CUDNN_LIBRARY})
else()
  message(
    STATUS
      "cuDNN not found - softmax_cudnn reports itself unavailable (set -DCUDNN_ROOT=<dir> to enable it)"
  )
  target_compile_definitions(playground_cudnn INTERFACE PLAYGROUND_NO_CUDNN)
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
