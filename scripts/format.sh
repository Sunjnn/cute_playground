#!/bin/bash
set -euo pipefail

find examples \
    -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" \) \
    -exec clang-format -i {} +

echo "clang-format done."
