#!/bin/bash
set -euo pipefail

if [ ! -f build/compile_commands.json ]; then
    echo "compile_commands.json not found, generating..."
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
fi

EXTRA_ARGS=""
if [ "$(uname)" = "Darwin" ]; then
    EXTRA_ARGS="--extra-arg=--sysroot=$(xcrun --show-sdk-path)"
fi

find examples \
    -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) \
    -exec clang-tidy -p build $EXTRA_ARGS {} +

echo "clang-tidy done."
