# cute_playground

Learning playground for [CUTLASS CuTe](https://github.com/NVIDIA/cutlass).
Each example is a standalone program under `examples/`; dropping in a new
`.cpp` or `.cu` file automatically creates a build target.

> Agent/LLM guidance (naming conventions, lint rules, dependency details)
> lives in [CLAUDE.md](CLAUDE.md).

## Requirements

- CMake ≥ 3.24 and a C++17 host compiler
- **CUDA Toolkit 12.8+** (required for all examples — CuTe headers include
  CUDA runtime headers unconditionally)
- Target GPU: **NVIDIA RTX 5060** (Blackwell, sm_120). `CMAKE_CUDA_ARCHITECTURES`
  defaults to `native`; override with `-DCMAKE_CUDA_ARCHITECTURES=120` if needed.

Without a CUDA toolkit, configuration still succeeds but no example targets
are generated.

## Build & Run

```bash
git submodule update --init --recursive   # fetch CUTLASS (first time only)
cmake -B build && cmake --build build
./build/examples/<example_name>
```

## Project Layout

```
cmake/Cutlass.cmake     # wires up the cutlass::cute INTERFACE target
examples/               # one executable per .cpp / .cu file
third_party/cutlass/    # CUTLASS git submodule (headers only)
scripts/                # format.sh (clang-format), tidy.sh (clang-tidy)
```

## Lint & Format

```bash
bash scripts/format.sh
bash scripts/tidy.sh    # requires build/ to exist for compile_commands.json
```
