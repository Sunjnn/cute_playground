# CLAUDE.md

Playground for learning CUTLASS CuTe. Each example is a standalone program in
`examples/`; add a file and it is built automatically.

## Target Hardware

Examples are developed and validated on an **NVIDIA GeForce RTX 5060**
(Blackwell, compute capability **12.0 / sm_120**).

- Requires **CUDA Toolkit 12.8+** (the first release with sm_120 support).
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which resolves to `120` on
  that machine. Override with `-DCMAKE_CUDA_ARCHITECTURES=120` when
  cross-compiling or when `native` detection is unavailable.
- Code may rely on Blackwell-specific features (e.g. TMA, tcgen05 MMA); it is
  not expected to run on older architectures without adaptation.

## Build & Run

```bash
git submodule update --init --recursive  # first time only
cmake -B build && cmake --build build    # configure + build
./build/examples/01_layout_basics        # run an example
```

All examples require a CUDA toolkit (nvcc). Without one, `examples/`
contributes no build targets and configure still succeeds.

- `examples/*.cpp` — host-only examples (compiled by the host compiler, but
  still need CUDA headers because CuTe includes them unconditionally).
- `examples/*.cu` — CUDA examples (compiled by nvcc).

## Dependencies

- **CUTLASS** — git submodule at `third_party/cutlass`. Only its headers are
  used, exposed as the `cutlass::cute` INTERFACE target
  (`third_party/cutlass/include`, `third_party/cutlass/tools/util/include`).
  Override the location with `-DCUTLASS_DIR=...` if needed.
- **CUDA toolkit** — detected via `check_language(CUDA)`; when present,
  `CUDAToolkit_INCLUDE_DIRS` is added so `.cpp` examples find the real CUDA
  headers. See [Target Hardware](#target-hardware) for architecture flags.

## Lint & Format

```bash
bash scripts/format.sh    # clang-format (in-place) on examples/
bash scripts/tidy.sh      # clang-tidy (requires build/ for compile_commands.json)
```

## Naming Conventions (enforced by clang-tidy)

| Element                        | Style        | Example           |
|-------------------------------|-------------|-------------------|
| Functions / methods           | `lower_case` | `do_thing()`      |
| Local variables / parameters  | `camelBack`  | `myVar`           |
| Classes / structs / enums     | `CamelCase`  | `MyClass`         |
| Member variables              | `camelBack_` | `memberVar_`      |
| Global variables              | `gCamelCase` | `gMaxSize`        |
| Static vars & consts          | `sCamelCase` | `sMaxSize`        |
| Constexpr variables           | `kCamelCase` | `kMaxSize`        |
| Enum values                   | `UPPER_CASE` | `COLOR_RED`       |
| Namespaces                    | `lower_case` | `mylib`           |
| Macros                        | `UPPER_CASE` | `MY_MACRO`        |

## Code Style

- C++17, clang-format LLVM-based: 2-space indent, 100-col limit, Attach braces.
- clang-tidy check groups: `bugprone-*`, `cppcoreguidelines-*`, `misc-*`,
  `modernize-*`, `performance-*`, `readability-*`. See `.clang-tidy` for the
  disabled checks and thresholds.
- Header guards use `#ifndef`/`#define` (not `#pragma once`).
- Never use blanket `using namespace` directives at file scope, except
  `using namespace cute;` inside example `main()` functions.
