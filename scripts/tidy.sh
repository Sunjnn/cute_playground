#!/bin/bash
set -euo pipefail

# clang-tidy lints translation units (*.cpp, *.cu) via the compile database.
# Headers (*.cuh/*.hpp/*.h) have no database entry of their own; they are
# checked transitively through the TUs that include them.

if [ ! -f build/compile_commands.json ]; then
    echo "compile_commands.json not found, generating..."
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
fi

# The database holds raw nvcc command lines, but clang-tidy drives the *clang*
# frontend, which rejects nvcc-only flags (-arch=native, --expt-*, -x cu, ...).
# Build a sanitized copy with those stripped and `-x cu` -> `-x cuda`; this
# mirrors the Remove list in .clangd so both tools see the same flags.
db_dir="$(mktemp -d)"
trap 'rm -rf "$db_dir"' EXIT
sed -E \
    -e 's/ -forward-unknown-to-host-compiler//g' \
    -e 's/ --expt-relaxed-constexpr//g' \
    -e 's/ --expt-extended-lambda//g' \
    -e 's/ --generate-code=[^" ]*//g' \
    -e 's/ -arch=[^" ]*//g' \
    -e 's/ -Xcompiler=[^" ]*//g' \
    -e 's/ -x cu / -x cuda /g' \
    build/compile_commands.json >"$db_dir/compile_commands.json"

# clang's CUDA support lags the toolkit, and with CUDA 13.3 clang 21 dies on two
# system-header problems that abort every TU. Neither is ours to fix, and nvcc
# never sees any of this:
#   1. clang's __clang_cuda_runtime_wrapper.h still includes
#      texture_fetch_functions.h, which CUDA 13 removed - an empty stub satisfies
#      it, since no example uses textures.
#   2. crt/math_functions.hpp uses _NV_RSQRT_SPECIFIER before crt/math_functions.h
#      defines it, in clang's include order only. Define it to what CUDA itself
#      defines on glibc >= 2.42.
compat_dir="$db_dir/clang-cuda-compat"
mkdir -p "$compat_dir"
: >"$compat_dir/texture_fetch_functions.h"

EXTRA_ARGS=(
    --extra-arg="-I$compat_dir"
    '--extra-arg=-D_NV_RSQRT_SPECIFIER=noexcept(true)'
    # Stripping -arch=native above would leave clang linting for its sm_52
    # default instead of the sm_120 target documented in CLAUDE.md.
    "--extra-arg=--cuda-gpu-arch=sm_${TIDY_CUDA_ARCH:-120}"
)
if [ "$(uname)" = "Darwin" ]; then
    EXTRA_ARGS+=("--extra-arg=--sysroot=$(xcrun --show-sdk-path)")
fi

find examples \
    -type f \( -name "*.cpp" -o -name "*.cu" \) \
    -exec clang-tidy -p "$db_dir" "${EXTRA_ARGS[@]}" {} +

echo "clang-tidy done."
