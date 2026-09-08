# cuTENSOR is optional and only examples/transpose/cutensor_transpose.cu uses it. Unlike cuBLAS and
# cuDNN it does not ship with the CUDA toolkit, so when it cannot be found playground::cutensor
# carries PLAYGROUND_NO_CUTENSOR instead of the real include path and library, which turns
# transpose_cutensor() into a stub that reports itself unavailable in the benchmark table. Building
# the examples therefore never requires cuTENSOR.
#
# Set CUTENSOR_ROOT to an installation that is not next to the CUDA toolkit - an unpacked .deb, a
# pip wheel (cutensor-cu12 / cutensor-cu13), a vendor drop - and the search below picks it up.

set(CUTENSOR_ROOT
    ""
    CACHE PATH "Root of a cuTENSOR installation, containing include/ and lib/")

# Wheels and vendor drops nest the payload one level down - cutensor/include, cutensor/lib - and
# pip-wheel-style installs may add a CUDA-versioned layer on top of that. Collect those
# subdirectories as additional hint roots so flat, nested and versioned layouts are all found.
set(CUTENSOR_INCLUDE_HINTS)
set(CUTENSOR_LIB_HINTS)
if(CUTENSOR_ROOT)
  file(GLOB CUTENSOR_NESTED "${CUTENSOR_ROOT}/*/")
  file(GLOB CUTENSOR_INCLUDE_HINTS "${CUTENSOR_ROOT}/include/*"
       "${CUTENSOR_NESTED}include")
  file(GLOB CUTENSOR_LIB_HINTS "${CUTENSOR_ROOT}/lib/*" "${CUTENSOR_NESTED}lib"
       "${CUTENSOR_NESTED}lib/*/x64")
endif()

find_path(
  CUTENSOR_INCLUDE_DIR cutensor.h
  HINTS ${CUTENSOR_ROOT} ${CUTENSOR_INCLUDE_HINTS} ENV CUTENSOR_ROOT
  PATH_SUFFIXES include)
find_library(
  CUTENSOR_LIBRARY
  NAMES cutensor
  HINTS ${CUTENSOR_ROOT} ${CUTENSOR_LIB_HINTS} ENV CUTENSOR_ROOT
  PATH_SUFFIXES lib lib64)

# Wheels ship only the versioned soname (libcutensor.so.2), which find_library does not match
# because it looks for libcutensor.so. Fall back to a glob over the same hint roots.
if(CUTENSOR_ROOT AND NOT CUTENSOR_LIBRARY)
  file(
    GLOB CUTENSOR_SONAMES
    LIST_DIRECTORIES false
    "${CUTENSOR_ROOT}/lib*/libcutensor.so*" "${CUTENSOR_ROOT}/*/lib*/libcutensor.so*")
  list(SORT CUTENSOR_SONAMES)
  list(LENGTH CUTENSOR_SONAMES cutensorSonameCount)
  if(cutensorSonameCount GREATER 0)
    list(GET CUTENSOR_SONAMES 0 CUTENSOR_LIBRARY)
  endif()
endif()

add_library(playground_cutensor INTERFACE)
add_library(playground::cutensor ALIAS playground_cutensor)

if(CUTENSOR_INCLUDE_DIR AND CUTENSOR_LIBRARY)
  message(STATUS "cuTENSOR found: ${CUTENSOR_LIBRARY}")
  target_include_directories(playground_cutensor
                             INTERFACE ${CUTENSOR_INCLUDE_DIR})
  target_link_libraries(playground_cutensor INTERFACE ${CUTENSOR_LIBRARY})
else()
  message(
    STATUS
      "cuTENSOR not found - transpose_cutensor reports itself unavailable (set -DCUTENSOR_ROOT=<dir> to enable it)"
  )
  target_compile_definitions(playground_cutensor
                             INTERFACE PLAYGROUND_NO_CUTENSOR)
endif()

mark_as_advanced(CUTENSOR_INCLUDE_DIR CUTENSOR_LIBRARY)
