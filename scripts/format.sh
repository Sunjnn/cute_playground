#!/bin/bash
set -euo pipefail

# Format every C/C++/CUDA source and header under examples/. The find is
# recursive, so example subfolders (e.g. examples/softmax/) are covered.
# clang-format infers CUDA from the .cu/.cuh extensions and needs no compile
# database.
find examples \
    -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) \
    -exec clang-format -i {} +

echo "clang-format done."
