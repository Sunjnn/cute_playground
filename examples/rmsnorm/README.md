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
- `rmsnorm_cutensor` — library composition, five steps because the operator
  set has no square and cuTENSOR's elementwise binary cannot broadcast: `x*x`
  (binary MUL), reduce over `j` with `alpha = 1/n` and `beta = 1` against a
  vector of `eps` (reduction), `sqrt` (permutation with `CUTENSOR_OP_SQRT`),
  `x*w` (trinary `MUL(x, w) + 0*C` with `w` broadcast by mode omission), and
  `* rcp(std)` (trinary again, `rcp` riding as `opB`). Seven walks over
  `m x n` — square reads `x` and writes the scratch, reduce reads it, weight
  reads `x` and overwrites it, scale reads it and writes the output — and one
  `m x n` scratch buffer shared by two non-overlapping lifetimes. It cannot do
  better: `x*x` and `sqrt` each cost a pass of their own, and only `rcp`
  rides along, as the trinary's `opB`.

**Shape constraints.** `rmsnorm` needs `n` a multiple of 512 and both leading
dimensions multiples of 4: its CTA tile is not predicated and its copies are 16
bytes wide, so a ragged or unaligned row would read and write out of bounds.
`rmsnorm_fused` needs `n`, `ldIn` and `ldOut` multiples of 4 and
`512 <= n <= 32768`, because it reads `float4` and holds one row in registers.
Both throw rather than run on a shape they cannot handle; the other five accept
any positive shape.

## Results

Measured on an RTX 5060 (sm_120), CUDA 13.3, cuTENSOR 2.7, cuDNN 9.24.

```
rmsnorm of 8192 x 4096 float, eps 1e-05, 20 timed iterations
implementation       verify    max_rel_diff    time_us     GB/s   vs_naive
rmsnorm_naive        PASS         2.427e-07      735.5    547.5      1.00x
rmsnorm_cub          PASS         2.986e-07     1032.3    390.1      0.71x
rmsnorm_cublas       PASS         2.792e-07     1720.6    234.0      0.43x
rmsnorm_cudnn        PASS         2.986e-07     1763.0    228.4      0.42x
rmsnorm_cutensor     PASS         5.886e-07     2578.7    156.1      0.29x
rmsnorm              PASS         2.788e-07      713.9    564.0      1.03x
rmsnorm_fused        PASS         2.353e-07      710.1    567.0      1.04x
```

Shape sweep (GB/s, 20 timed iterations, every row verified `PASS`; shapes
respect both kernels' constraints — `n` a multiple of 512 and `n <= 32768`):

| shape (m x n) | naive  | cub    | cublas | cudnn  | cutensor | rmsnorm | fused  | rmsnorm vs | fused vs |
|---------------|--------|--------|--------|--------|----------|---------|--------|------------|----------|
| 512 x 512     | 158.7  | 96.0   | 49.0   | 50.5   | 29.4     | 178.0   | 184.0  | 1.12x      | 1.16x    |
| 1024 x 1024   | 221.9  | 191.0  | 169.1  | 113.7  | 54.2     | 589.8   | 643.9  | 2.66x      | 2.90x    |
| 2048 x 2048   | 603.5  | 373.4  | 235.2  | 218.2  | 137.2    | 586.0   | 658.2  | 0.97x      | 1.09x    |
| 4096 x 4096   | 541.7  | 387.8  | 219.9  | 214.2  | 149.0    | 498.8   | 531.8  | 0.92x      | 0.98x    |
| 8192 x 4096   | 537.7  | 387.1  | 235.1  | 216.4  | 154.2    | 549.2   | 561.8  | 1.02x      | 1.04x    |
| 16384 x 4096  | 555.4  | 382.4  | 232.5  | 227.7  | 145.5    | 545.8   | 566.0  | 0.98x      | 1.02x    |
| 32768 x 4096  | 561.3  | 392.9  | 232.6  | 229.5  | 120.9    | 558.9   | 565.5  | 1.00x      | 1.01x    |
| 4096 x 8192   | 559.7  | 394.9  | 229.3  | 226.9  | 155.5    | 491.0   | 567.2  | 0.88x      | 1.01x    |
| 4096 x 16384  | 473.6  | 395.0  | 226.5  | 225.3  | 151.8    | 374.9   | 562.6  | 0.79x      | 1.19x    |
| 2048 x 32768  | 388.8  | 398.0  | 228.0  | 223.0  | 152.0    | 379.2   | 559.6  | 0.98x      | 1.44x    |
| 32 x 8192     | 181.5  | 97.9   | 51.0   | 42.5   | 30.7     | 174.8   | 214.2  | 0.96x      | 1.18x    |
| 128 x 32768   | 369.2  | 446.1  | 150.1  | 210.7  | 149.0    | 479.8   | 627.2  | 1.30x      | 1.70x    |
| 512 x 32768   | 363.8  | 376.1  | 231.6  | 220.0  | 151.2    | 361.9   | 563.4  | 0.99x      | 1.55x    |
| 8192 x 512    | 503.6  | 314.6  | 216.1  | 205.7  | 143.9    | 619.7   | 445.7  | 1.23x      | 0.88x    |
| 32768 x 512   | 555.3  | 306.1  | 226.5  | 184.7  | 113.6    | 568.0   | 570.5  | 1.02x      | 1.03x    |

The numbers worth reading:

- **`rmsnorm_fused` is the row to beat, and beats everything but itself.**
  It wins over `rmsnorm_naive` at 14 of 15 shapes and widens the gap exactly
  where its 2x-traffic advantage is paid in full: few long rows
  (128 x 32768: 1.70x, 512 x 32768: 1.55x, 2048 x 32768: 1.44x), where naive's
  third walk over the row is pure DRAM. The one loss, 8192 x 512 (0.88x), is
  its smallest supported row, where the register cache barely amortizes.
- **`rmsnorm` (CuTe, two passes) ties `rmsnorm_naive`** at ~1.0x across the
  sweep — the same 3x traffic and the same bytes, so the cp.async tile neither
  gains nor loses against the plain walk — and drops below it on the widest
  rows (4096 x 16384: 0.79x), where its 1x512 tiles give one row of 16384
  sixteen round trips through shared memory. The two outliers above 1.2x
  (1024 x 1024, 8192 x 512) sit on naive's two soft spots: the 1024 x 1024
  dip to 221.9 GB/s (launch-bound small problem), and n = 512, its smallest
  tile.
- **The four libraries floor at their walk counts.** `cub` (two walks) is the
  best at ~390 GB/s; `cublas` and `cudnn` (five) hold ~220-230; `cutensor`
  (seven) trails at ~120-155. The extra walks cost more than any library's
  tuned kernel earns back.

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

- **`rmsnorm_cutensor` was written against reconstructed signatures.** The
  container this folder was built in had no libcutensor, so the cuTENSOR 2.x
  argument lists were inferred from `cutensorCreatePermutation` in
  `examples/transpose`. Compiling against cuTENSOR 2.7 fixed three things: the
  binary elementwise operator argument had been left `CUTENSOR_OP_IDENTITY`
  where it had to be `CUTENSOR_OP_MUL`, `cutensorCreateReduction` needed the
  `descD`/`modeD` pair, and the unary `sqrt` step had used an API that does not
  exist (a unary op is applied via `cutensorCreatePermutation`). One further
  redesign fell out of testing the real library: cuTENSOR 2.7's elementwise
  binary rejects broadcasting — by mode omission *and* by zero stride, both
  `CUTENSOR_STATUS_NOT_SUPPORTED` — so the weight and scale steps use the
  trinary elementwise, which does broadcast a mode missing from an operand.
- **`rmsnorm_cutensor` builds five plans per call,** and cuTENSOR 2.7's host-side
  plan creation overflows the linker's default 1 MiB main-thread stack on sm_120
  (`STATUS_STACK_BUFFER_OVERRUN` inside `cutensorCreatePlan`). `examples/CMakeLists.txt`
  therefore gives the `rmsnorm` target the same `/STACK:8388608` it gives
  `transpose`, which only builds one plan.
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
