# flash_attention

Single-head forward attention, following the harness and implementation layout of
`examples/fused_softmax`:

```text
Q: (m, k), K: (n, k), V: (n, k), O: (m, k), all row-major fp16
O = softmax(scale * Q K^T + mask) V
```

`scale` defaults to `1/sqrt(k)`. The default mask is causal: query `i` sees key
`j` iff `j <= i + n - m`. This bottom-right alignment also works when `m < n`;
causal mode requires `m <= n` so that every query sees at least one key.
`--mask=none` computes dense attention. GPU implementations additionally require
`abs(scale) <= FLT_MAX / (2*k*65504^2)`: a conservative bound for finite fp16 Q/K,
with rounding margin. Merely checking that scale is finite would still allow
FP32 logits to overflow and make stable softmax evaluate `inf - inf`.

The example contains a plain CUDA correctness floor, a cuBLAS library
composition, an educational fused CuTe implementation, and an optional **official
Dao-AILab FlashAttention-2 baseline**. Every available implementation is checked
against an independent double-precision host reference using the **same quantized
fp16 inputs** before timing. `flash_attention` is the educational kernel;
`flash_attention_upstream` calls genuine upstream kernels, not PyTorch SDPA or a
second homegrown implementation. Batch/head dimensions, GQA, dropout, arbitrary
masks/bias, public log-sum-exp output and backward are not included.

## The implementations

- **`flash_attention` — this repo's fused CuTe kernel.** One CTA owns 32 query
  rows and streams the keys/values in 64-row tiles. The SM80 `16x8x16` fp16/fp32
  MMA configuration and swizzled Q/K shared layouts come from
  `fused_softmax_mainloop.cuh`; the instructions also run on the project's
  sm_120 target. QK produces only a tile of scores, not a global score matrix,
  and those scores never leave registers: the online max/sum reduction runs on
  the accumulator fragment itself (four-lane shuffles, then a four-entry flush
  through shared memory for the N-atom split), and only the unnormalized fp16
  probabilities are staged for the second MMA. A second MMA multiplies these by
  V, keeping the output numerator in FP32 registers across key tiles. O is
  normalized and stored once at the end. Q is staged once per CTA, while K and V
  are single buffers that a `cp.async` prefetch refills for tile `t + 1` while
  tile `t`'s softmax and PV run. Fully future causal tiles are skipped
  CTA-uniformly; partial tiles are masked row-by-row.
- **`flash_attention_naive` — plain CUDA correctness floor.** A CTA owns one
  query row, computes the visible scores using SIMT arithmetic, performs stable
  softmax in FP32, and multiplies by V without tensor cores. It does not allocate
  a global attention matrix, but its whole score row must fit in shared memory.
  It is deliberately not the speedup baseline.
- **`cublas_unfused_attention` — the library baseline.** A first
  `cublasGemmEx` computes scaled FP32 scores; a separate masked softmax kernel
  writes normalized fp16 P; a second GEMM computes PV with FP32 accumulation and
  fp16 output. It allocates global FP32 S and fp16 P, i.e. `6*m*n` scratch bytes.
  This is the cost of separating the two GEMMs and softmax rather than fusing
  them.
- **`flash_attention_upstream` — official FlashAttention-2 nonsplit forward.**
  Built from [Dao-AILab/flash-attention
  `v2.8.3.post2`](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2),
  using its four FP16 D64/D128 causal/noncausal forward translation units. The
  adapter passes the existing GPU pointers directly to
  `flash::run_mha_fwd_<cutlass::half_t, D, causal>` on the default stream. There is
  no Python dispatch, Torch tensor conversion or host/device input copy. Upstream
  requires an internal FP32 LSE output of `m` elements; its asynchronous allocation
  and release are included in each wrapper call, like the cuBLAS scratch. This is
  the real official kernel, but with `num_splits=1`: it is **not** the full
  Python-facing dispatcher's automatically tuned split-KV inference path.

### Online softmax

The fused kernel retains a maximum `a`, denominator `l` and **unnormalized**
output numerator `u` for every query row, initially `(-inf, 0, 0)`. For a new
scaled, masked score tile `s`:

```text
a_new = max(a, row_max(s))
alpha = exp(a - a_new)
p     = exp(s - a_new)
l     = alpha * l + row_sum(p)
u     = alpha * u + p V_tile
a     = a_new

After the final tile: O = u / l
```

Rescaling **both** `l` and every feature of `u` is essential when a later tile
raises the maximum. The empty initial state uses `alpha=0`; masked scores
produce exactly zero probability without evaluating `exp(-inf - -inf)`.
Unlike `fused_softmax_recompute`, this visits each KV tile once and never writes
a global S or P. Each visit still loops over head-dimension tiles for QK; Q lives
in shared memory, not in registers.

### Layout and precision choices

QK's output fragment is **not** PV's input fragment, but no score tile is ever
written to shared memory: the online maximum, exponentiation and row sum are
evaluated on the accumulator fragment, and the resulting unnormalized fp16
probabilities are written straight into a row-major P tile that PV consumes as
MMA operand A. Physical row-major `V(key, feature)` is exposed as the logical
MMA B operand `B(feature, key)` using its actual transposed layout, stride
`(1, k)`. Shared-to-register PV copies are generic CuTe copies, rather than
reusing the K-major QK `ldmatrix` load atom for a differently oriented operand.
Every output fragment element's row is obtained through the MMA's
identity-tensor partition for rescaling and final normalization.

Q is staged once per CTA because its head dimension is invariant across key
tiles; K and V are single buffers, and P is a third. A `cp.async` prefetch
refills K and V for tile `t + 1` immediately after tile `t` has consumed them, so
the transfer overlaps the softmax and PV in between. There is no multi-stage ring
and no buffer aliasing. Dynamic shared memory is 24 KiB at `k=64` and 44 KiB at
`k=128`, plus 384 bytes of row statistics and 1 KiB of per-tile reduction
partials.

Both matrix multiplications and the online statistics/output numerator use
FP32 accumulation. The fused kernel casts **unnormalized** p to fp16 for PV but
computes `l` from FP32 p; the baseline instead casts **normalized** P to fp16.
These round at different points and are not expected to agree bit-for-bit.
The host reference does not make either probability approximation. Verification
uses `max(|got-want| / max(|want|, 1)) <= 1e-3`, allowing fp16 rounding without
unstable relative errors near zero. Nonfinite output fails verification.

## Shape constraints

| implementation | constraint |
|---|---|
| `flash_attention` | `m % 32 = 0`, `n % 64 = 0`, `k = 64` or `128` |
| `flash_attention` input strides | Q/K/V leading dimensions divisible by 8 and row pointers 16-byte aligned; all leading dimensions at least `k` |
| `flash_attention_naive` | `k + n <= 12272`, including query/score rows and warp reductions within 48 KiB shared memory |
| `cublas_unfused_attention` | positive dimensions and valid row-major leading dimensions |
| `flash_attention_upstream` | enabled dependency; `k = 64/128`; positive fp32-safe scale; Q/K/V/O pointers 16-byte aligned and leading dimensions divisible by 8; sequence tails supported |
| harness | positive sizes/iterations; fp32-safe scale as above; causal requires `m <= n`; `m*n`, `m*k`, `n*k` fit in int32 |

CuTe has no ragged-sequence or head-dimension tail path in this specialization;
unsupported shapes show `n/a` with an explanation rather than silently losing
elements. The baseline implementations can still check such shapes. Actual
verification failures and CUDA runtime errors make the executable return a
nonzero status; failing implementations are not timed.

## Run

The project targets **NVIDIA RTX 5060 (sm_120)** and requires **CUDA Toolkit
12.8+**. CMake discovers the new example directory automatically; no new target
registration is needed.

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target flash_attention
./build/examples/flash_attention                       # 2048 x 2048 x 128, causal
./build/examples/flash_attention --m=256 --n=512 --k=64 --mask=none
./build/examples/flash_attention --m=128 --n=256 --k=128 --mask=causal
```

### Enable the official FlashAttention-2 baseline

The optional baseline is disabled by default. The row remains visible as `n/a`
with an enablement message, and the original three implementations still build.
CMake never downloads source or installs packages. Explicitly enabling it with
missing/incompatible prerequisites is a configuration error, not a silent fallback.

The official source lives as `third_party/flash-attention`, a git submodule at
**`v2.8.3.post2`**. After cloning this repository, initialize all submodules:

```bash
git submodule update --init --recursive
```

This also initialises the pinned CUTLASS unter `csrc/cutlass` (commit
`dc4817921edda44a549197ff3a9dcf5df0636e7b`, version 4.0.0). If the submodule is
present and initialized, CMake automatically uses it as `FLASH_ATTENTION_DIR`;
explicit `-DFLASH_ATTENTION_DIR=/other/path` overrides it.

Only one other prerequisite remains: a **CUDA-enabled LibTorch/PyTorch C++
installation**:

```bash
cmake -B build/fa2 -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPLAYGROUND_ENABLE_FLASH_ATTENTION=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/fa2 --target flash_attention -j2
./build/fa2/examples/flash_attention --m=128 --n=256 --k=128
```

Use a fresh build directory such as `build/fa2`; do not reuse a build cache from
another machine/platform. Alternatively, set
`-DTorch_DIR=/path/to/libtorch/share/cmake/Torch` for a standalone
CUDA LibTorch distribution; Python is not required at runtime. Match LibTorch's
CUDA/toolchain/ABI requirements to the selected toolkit. A CPU-only Torch package
is insufficient. Installing the `flash-attn` Python wheel alone does not supply
the pinned source checkout needed for this native build.

Stable FA2 includes ATen Philox headers and calls C10 CUDA error-check functions
even in this no-dropout configuration. These are real build/link dependencies;
this adapter does not patch them out or pretend the upstream source is Torch-free.
Only four regular forward TUs and the private bridge
(`examples/flash_attention/upstream/upstream_forward.cu`, below the example glob
and compiled only into the private library) are compiled: no
`flash_api.cpp`, backward, BF16, other dimensions or split-KV TUs.

`cmake/FlashAttention.cmake` checks the source version and the upstream CUTLASS
4.0.0 headers, pinned at `dc4817921edda44a549197ff3a9dcf5df0636e7b`. Git revisions
are checked when checkout metadata is available. All upstream CUTLASS/Torch
headers, ABI flags and options stay private to the separate library; they do not
replace this project's CUTLASS or affect other examples. The public adapter
header contains only the existing CUDA-half/raw-pointer interface.

Official FA2 reduces raw scores before applying scale. Zero or negative scale
can invalidate masked/tail softmax (`-inf * 0` or a reversed stability maximum),
so this adapter reports `n/a` for `scale <= 0`; the other implementations keep
their existing support. D128 on sm_120 uses upstream's shared-memory opt-in;
validate the actual target GPU's launch and resource limits before benchmarking.

Options: `--m`, `--n`, `--k`, `--iterations`, `--mask=causal|none`, `--scale`,
`--verify=true|false`, `--input=random|zero-qk|constant-v|late-max`, `--help`.
Use the documented `--name=value` syntax. Empty values, malformed numeric
suffixes, unknown options and misspelled booleans are rejected; `--verify` also
accepts `0`/`1`. The seed is fixed. Three warmups precede 20 timed iterations by default, with
default-stream synchronization for equal treatment of the wrappers.
`--verify=false` skips the slow CPU double reference for performance sweeps;
its resulting timings are **not** evidence of correctness.

The output table follows neighboring examples: `implementation`, `verify`,
`max_rel_diff`, `time_us`, `TFLOP/s`, `GB/s`, `vs_cublas`, `vs_fa2`. TFLOP/s is the
**dense-equivalent** `4*m*n*k` QK-plus-PV count; causal kernels can skip work, so
it is not measured tensor-core utilization. GB/s prices only the irreducible
`2*m*k`-byte O write, **not** Q/K/V reads or actual DRAM traffic. `vs_cublas` is
cuBLAS time divided by implementation time; `vs_fa2` is official FA2 time divided
by implementation time. Above 1 means faster than the corresponding baseline.
`vs_fa2` is `n/a` when the official row is disabled, unsupported or failed.

Timings include the complete wrapper invocation and default-stream synchronization,
not only the device kernel. Required asynchronous scratch allocation/release is
included for cuBLAS and upstream FA2, with pool pages retained after warmup. No
implementation performs input transfers or Python dispatch inside the timed loop.

## Verification cases

On a CUDA machine, run small square and rectangular cases for both masks and
head dimensions, then check online rescaling and masking explicitly. For the
enabled official comparison, use `./build/fa2/examples/flash_attention` in place
of the default executable below and require `PASS` for that row, not `n/a`.
The zero-scale case intentionally skips only the official row:

```bash
for k in 64 128; do
  for mask in none causal; do
    ./build/examples/flash_attention --m=128 --n=128 --k=$k --mask=$mask
    ./build/examples/flash_attention --m=96 --n=320 --k=$k --mask=$mask
    ./build/examples/flash_attention --m=32 --n=64 --k=$k --mask=$mask
  done
done

# Uniform attention: mean of V, or each query's visible causal prefix mean.
./build/examples/flash_attention --m=128 --n=256 --k=64 --input=zero-qk --mask=none
./build/examples/flash_attention --m=128 --n=256 --k=128 --input=zero-qk --mask=causal

# Attention must preserve constant V, regardless of the score distribution.
./build/examples/flash_attention --m=128 --n=256 --k=128 --input=constant-v

# Later tile raises the maximum; omitting numerator rescaling must fail.
./build/examples/flash_attention --m=128 --n=256 --k=64 --input=late-max --scale=1 --mask=none
./build/examples/flash_attention --m=128 --n=256 --k=128 --input=late-max --scale=1 --mask=causal

# Zero scaling and an unsupported CuTe tail (baselines still run).
./build/examples/flash_attention --m=96 --n=320 --k=64 --scale=0
./build/examples/flash_attention --m=33 --n=65 --k=64 --mask=none

compute-sanitizer --tool memcheck ./build/examples/flash_attention --m=96 --n=320 --k=128
compute-sanitizer --tool racecheck ./build/examples/flash_attention --m=96 --n=320 --k=128
```

**Validation status:** CUDA compilation, GPU correctness, sanitizer checks and
performance measurements have not been performed on the current macOS editing
machine, which has no `nvcc` or NVIDIA GPU tooling. No benchmark numbers or
speedup claims are reported here. Run the commands above on the target CUDA
machine before relying on this example's numerical or performance results.

Local checks cover formatting/whitespace, disabled CMake configuration, explicit
missing-CUDA errors, mock dependency/target isolation, and host-only bridge
parameter/cleanup and table/stub checks. The latter use mock headers and launchers:
they do **not** compile official CUDA kernels or validate real LibTorch linkage.
