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
    -e 's/ -x cu / -x cuda /g' \
    build/compile_commands.json >"$db_dir/compile_commands.json"

EXTRA_ARGS=""
if [ "$(uname)" = "Darwin" ]; then
    EXTRA_ARGS="--extra-arg=--sysroot=$(xcrun --show-sdk-path)"
fi

find examples \
    -type f \( -name "*.cpp" -o -name "*.cu" \) \
    -exec clang-tidy -p "$db_dir" $EXTRA_ARGS {} +

echo "clang-tidy done."
