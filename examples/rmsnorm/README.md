# rmsnorm

Row-wise RMS normalization of an `m x n` row-major float matrix, seven
implementations benchmarked head-to-head on an **NVIDIA RTX 5060** (Blackwell,
sm_120). Three kernels are implemented in this repo — two CuTe-free plain CUDA
(`rmsnorm_fused`, `rmsnorm_naive`) and one CuTe (`rmsnorm`) — the other four
are library compositions they are measured against.

Each row is one token's hidden state and each column one channel:

```
dOut[i, j] = dIn[i, j] * rsqrt(mean_j(dIn[i, j]^2) + eps) * dWeight[j]
```

`dWeight` is a single vector of `n` floats shared by every row, and `eps` is
`1e-5` (passed to every implementation by the harness, so they cannot drift
apart on it).

## The implementations

- **`rmsnorm_fused` — this repo's single-pass kernel.** One row per CTA with
  the whole row cached in registers (32 floats per thread as 8 × `float4`), so
  the input is read once and the output written once — 2× DRAM traffic instead
  of the two-pass kernels' 3×. Unlike softmax, RMSNorm needs nothing from a
  first walk that a second one could not have produced — there is no row
  maximum to learn before exponentiating — so the row can be cached and
  normalized in one pass. One warp-shuffle reduction plus one float per warp in
  shared memory, no smem tile and no copy at all. `dWeight` is *not* cached
  alongside the row: that would double the register footprint past what fits,
  and every CTA asks for the same `n` floats, so after the first few rows the
  vector is L2-resident. Row length is capped at 32768 floats by the register
  file. This is the shape a production fused RMSNorm (PyTorch, Apex, vLLM)
  takes, and the row to beat.
- **`rmsnorm` — this repo's CuTe kernel.** One CTA per row, the row walked
  twice in 1×512 tiles staged through shared memory with `cp.async`. The first
  walk folds each thread's elements into one register and reduces the sum of
  squares with warp shuffles; the second walk reloads each tile together with
  the matching tile of `dWeight`, scales it in place and stores it as 16-byte
  vectors. 3× the traffic, and the shared-memory tile is what a pipelined
  version would build on — see [Next steps](#next-steps).
- **`rmsnorm_naive` — this repo's baseline, and the reference the `vs_naive`
  column is normalized against.** One row per CTA, 256 threads walking their
  columns straight through global memory twice, no shared-memory tile, no
  `cp.async` and no vector wider than the compiler happens to find. It moves
  the same 3× the row's bytes as `rmsnorm`, so the comparison between those two
  measures how the bytes are fetched rather than how many there are.
- `rmsnorm_cub` — library composition: `cub::DeviceSegmentedReduce::Sum` over a
  transform iterator yielding `x*x/n` for the row statistic, then one thrust
  `for_each_n` to apply the scale. Two kernels, 3× traffic, no scratch beyond
  `m` floats. The closest a library-only version gets to a fused kernel.
- `rmsnorm_cublas` — library composition: cuBLAS has no squared reduction and
  `sgemv` cannot square on the way in, so the row statistic takes a thrust pass
  that materializes `x²` into a second `m x n` buffer, one `cublasSgemv`
  against a vector of ones (`alpha = 1/n` makes the result the mean directly),
  and a thrust pass to scale. Five walks over `m x n` and `m * n * 4` bytes of
  scratch — 128 MB at the default shape.
- `rmsnorm_cudnn` — library composition, and a trick: SPATIAL batch
  normalization reduces over every mode except `C`, so describing the tile as
  `N=1, C=m, H=1, W=n` turns that reduction into a reduction over one row's
  columns. With `estimatedMean` forced to zero, `estimatedVariance` set to
  `mean_j(x²)`, scale 1 and bias 0, `cudnnBatchNormalizationForwardInference`
  computes RMSNorm without the weight. BN's scale and bias are per-`C` — i.e.
  per-row under this description — so `dWeight[j]` cannot ride along and is
  applied afterwards by a `cudnnOpTensor` multiply with `B` described as
  `(1, 1, 1, n)` and broadcast. cuDNN supplies neither the statistic (legacy
  cuDNN has no row-wise reduction, so it comes from the same CUB call
  `rmsnorm_cub` uses) nor a fused normalize-and-scale (BN writes to an `m x n`
  scratch that OpTensor reads back): five walks over `m x n`.
- `rmsnorm_cutensor` — library composition, five steps because cuTENSOR's
  elementwise operations take two operands and its unary operators include
  neither a square nor a reciprocal square root: `x*x` (binary), reduce over
  `j` with `alpha = 1/n` and `beta = 1` against a vector of `eps` (reduction),
  `sqrt` (unary), `x*w` (binary), `* rcp(std)` (binary). Seven walks over
  `m x n` — square reads `x` and writes the scratch, reduce reads it, weight
  reads `x` and overwrites it, scale reads it and writes the output — and one
  `m x n` scratch buffer shared by two non-overlapping lifetimes. It cannot do
  better: with two operands per elementwise op and no square or reciprocal
  square root in the unary set, `x*x`, `sqrt` and `rcp` each cost a pass of
  their own. **Never compiled, never run** — see [Caveats](#caveats).

**Shape constraints.** `rmsnorm` needs `n` a multiple of 512 and both leading
dimensions multiples of 4: its CTA tile is not predicated and its copies are 16
bytes wide, so a ragged or unaligned row would read and write out of bounds.
`rmsnorm_fused` needs `n`, `ldIn` and `ldOut` multiples of 4 and
`512 <= n <= 32768`, because it reads `float4` and holds one row in registers.
Both throw rather than run on a shape they cannot handle; the other five accept
any positive shape.

## Results

Not measured yet. `cute_dev` — the container this folder was built in — has
CUDA 13.3 but no GPU (`nvcc` reports *Cannot find valid GPU for '-arch=native'*),
so every implementation here is compile-verified and nothing has been executed.

```
rmsnorm of 8192 x 4096 float, eps 1e-05, 20 timed iterations
implementation       verify  max_rel_diff    time_us     GB/s   vs_naive
rmsnorm_naive        PASS         ...          ...      ...      1.00x
rmsnorm_cub          PASS         ...          ...      ...      ...
rmsnorm_cublas       PASS         ...          ...      ...      ...
rmsnorm_cudnn        PASS         ...          ...      ...      ...
rmsnorm_cutensor     n/a       (built without cuTENSOR)
rmsnorm              PASS         ...          ...      ...      ...
rmsnorm_fused        PASS         ...          ...      ...      ...
```

## Reading the numbers

- **verify** — the output is compared against a row-wise reference accumulated
  in double; the reference's sum of squares is accumulated in double too.
- **max_rel_diff** — the largest `|got - want| / max(|want|, 1)` over every
  element. The denominator floors at one because an RMSNorm output is zero
  wherever its input is and a relative error against a zero reference is
  meaningless; elements of magnitude at least one are judged relatively, the
  rest absolutely. `PASS` means at most `1e-4`.
- **time_us** — mean over 20 timed iterations, 3 warmup iterations first. An
  implementation that fails verification is not timed at all: measuring how
  fast it is at being wrong would only reward a kernel that stores nothing.
- **GB/s** — priced at two reads of the input and one write of the output,
  which is what a two-pass RMSNorm moves and the least one that reloads can
  cost. `dWeight` is not counted: it is the same `n` floats for all `m` rows, so
  after the first CTAs it comes from L2, and charging it as DRAM traffic would
  overstate everyone by a factor that depends on `m` rather than on the kernel.
  `rmsnorm_fused` moves 2/3 of this, so read its GB/s against a ceiling 1.5×
  higher.
- **vs_naive** — `rmsnorm_naive`'s time over this row's time.

## Caveats

- **`rmsnorm_cutensor` has never been compiled.** The container this folder was
  built in has no libcutensor (`cmake/Cutensor.cmake` reports NOTFOUND, so
  `PLAYGROUND_NO_CUTENSOR` is defined and only the stub is built), and the
  cuTENSOR 2.x signatures were reconstructed from `cutensorCreatePermutation` in
  `examples/transpose` — the one operation in that API this repo has exercised.
  `cutensorCreateReduction` and the three `*Execute` argument lists in particular
  need checking against `cutensor.h`. The RTX 5060 host does have cuTENSOR 2.7
  (`-DCUTENSOR_ROOT="C:/Program Files/NVIDIA cuTENSOR/v2.7"`, per
  `examples/transpose/README.md`), so that is where this branch gets its first
  compile — expect to fix signatures there.
- **`rmsnorm_cutensor` builds five plans per call,** and cuTENSOR 2.7's host-side
  plan creation overflows the linker's default 1 MiB main-thread stack on sm_120
  (`STATUS_STACK_BUFFER_OVERRUN` inside `cutensorCreatePlan`). `examples/CMakeLists.txt`
  therefore gives the `rmsnorm` target the same `/STACK:8388608` it gives
  `transpose`, which only builds one plan.
- **`rmsnorm_cudnn` compiles but has not run.** Two things could still reject at
  runtime: `cudnnOpTensor`'s broadcast of a size-1 `C` dimension, and SPATIAL
  batch norm with `C = m = 8192` channels of width 1. If either fails the row
  prints the cuDNN error string instead of a time.
- The harness sets the default memory pool's release threshold to `UINT64_MAX`,
  as `examples/softmax` does: four of the baselines allocate scratch per call,
  and without it `cudaFreeAsync` would hand the pages back to the driver and put
  a real allocation inside the timed region.

## Run

```bash
cmake -B build && cmake --build build
./build/examples/rmsnorm [--m=N] [--n=N] [--iterations=N]
```

A shape sweep is worth more than the default shape here, because the two
single-pass kernels are capped from opposite ends: `rmsnorm_fused` stops
working above `n = 32768` (register file), while `rmsnorm` and `rmsnorm_naive`
keep going but pay 3× traffic at every size.

## Next steps

- **Pipeline the CuTe kernel.** `rmsnorm` waits for each tile's `cp.async`
  before computing it. A multi-stage ring — a prologue that prefetches the first
  tiles, a steady state that issues tile *i + k*'s load while tile *i* is
  computed, an epilogue that drains — is what `softmax_multistage` does, and it
  would hide the second walk's latency without changing its traffic.
- **Give `rmsnorm_fused` a CuTe counterpart.** It is plain CUDA because there is
  nothing to stage, but its tiling is exactly what `local_partition` over a
  register-resident tile describes, and expressing it in CuTe would make the
  2×-traffic kernel comparable to the 3× one in the same layout algebra.
- **Small `n`.** At `n <= 1024` a whole row fits in one warp, so the block
  reduction and its `__syncthreads` disappear; at large `m` that trades
  occupancy for latency. Neither kernel here specializes for it.
