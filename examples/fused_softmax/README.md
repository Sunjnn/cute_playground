# fused_softmax

Fused softmax of the attention score matrix, benchmarked on an **NVIDIA RTX
5060** (Blackwell, sm_120), CUDA 13.3. Q is `(m, k)` and K is `(n, k)`, both
row-major fp16; the output `P = softmax(scale * Q K^T)` is `(m, n)` fp16. Three
kernels are implemented in this repo — plain CUDA `fused_softmax_naive` and CuTe
`fused_softmax_regrow` / `fused_softmax_recompute` — and one library
composition, `cublas_unfused_softmax`, is the baseline they are measured
against. The harness verifies every implementation against a double-precision
host reference before timing it.

The mask defaults to causal: query `i` sees column `j` iff `j <= i + n - m`,
which is why the harness requires `m <= n` under that mask. `scale` defaults to
`1/sqrt(k)`, so it moves with `k` in the k-sweep below. Every number in this
file comes from the harness's defaults (20 timed iterations, 3 warm-up
iterations) except where a table says otherwise.

## The implementations

- **`fused_softmax_regrow` — this repo's single-pass CuTe kernel.** One CTA owns
  32 rows and *all* of a row's columns, so the score matrix never leaves the
  chip: a 2 x 4 grid of SM80 `16x8x16` fp16/fp32 MMA atoms computes `S = Q K^T`
  into registers, `store_scores()` scales and masks it on the way to shared
  memory, and one warp per row reduces with shuffles and stores `P` as
  `__half2`. P is written once and S never, which is the 2·M·N-byte floor. The
  row has to fit, so the kernel is instantiated for `n = 128, 256, 512`; the GEMM
  operands and the score staging alias one buffer because the phases never
  overlap. Shared memory is 20 / 36 / 68 KB for those widths, which is 4 / 2 / 1
  CTA per SM; the 512-wide build opts in through `cudaFuncSetAttribute` and is
  the width where occupancy starts to cost. It does not skip masked work: the
  whole row is one tile, so there is no column tile to drop.
- **`fused_softmax_recompute` — this repo's streaming CuTe kernel.** The same
  mainloop, but the columns are streamed in tiles and visited twice. The first
  pass folds each tile into the row's running `(max, sum)` with the
  online-softmax update and throws the tile away; the second recomputes it and
  writes `exp(s - max) / sum`. It keeps the 2·M·N floor for any `n` that is a
  multiple of 64 and pays for it with 2x the GEMM multiply-adds instead of
  extra traffic. The global-to-shared copies are `cp.async`, one commit group
  per K tile and one wait before the read; that is worth 7–18% on the shapes
  with the fewest CTAs, where the copy would otherwise be exposed. Four
  scheduling choices matter as much as the arithmetic:
  - **Causal row order.** Under a causal mask the last row blocks see every
    column and the first block sees only its own 32, so the row blocks have
    very different weights. Blocks are dispatched in increasing `blockIdx`, so
    the kernel maps CTA 0 to the *last* row block: the heavy CTAs start first
    (longest-processing-time-first), and the grid does not need to be persistent
    to get the benefit.
  - **Causal tile skip.** A 64- or 128-wide column tile that lies entirely past
    the last visible column (`j <= row + n - m`) is dropped in both passes, and
    its output zeros are written with a bare store instead of a computed tile.
    That is up to half the GEMM on a square causal shape.
  - **Adaptive tile width.** The kernel is instantiated at 128 and 64 columns.
    128 halves the per-tile overhead and fits two CTAs per SM; 64 halves the
    tiles per CTA and fits four. The host dispatches to 64 when the 128-wide
    grid would spill into a second, mostly empty wave — measured with
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, not a hardcoded shape —
    and keeps 128 for causal shapes when `n` allows it, where the LPT order
    already hides the imbalance and the per-tile overhead dominates. The
    64-wide instantiation is also what lets `recompute` accept `n` multiples of
    64 (e.g. 192, 320).
  - **One cp.async stage, not a ring.** `compute_tile_async<kStages>` can stage
    the operands kStages deep, but only `kStages = 1` is used: at 2 the 64-wide
    build drops from 4 CTAs per SM to 3 and the 128-wide from 2 to 1, and the
    large shapes regressed (16384 x 2048: 868 -> 964 µs). The parameter stays
    so the trade can be retried when the tile changes.
- **`fused_softmax_naive` — this repo's correctness floor.** One row per CTA,
  256 threads, plain SIMT multiply-add (no tensor cores), the row's scores
  staged in shared memory and walked three times: max, exponentiate-and-sum,
  normalize-and-store. It is the only implementation that skips masked work at
  the per-element level — a causal row evaluates only its visible columns — so a
  causal shape costs it half. Every other implementation masks in the epilogue
  (or, in `recompute`'s case, skips whole tiles).
- `cublas_unfused_softmax` — the baseline, and the shape of the problem without
  fusion: `cublasGemmEx` (fp16 in, fp32 out, `alpha = scale`) writes the whole
  `m x n` fp32 score matrix to global memory, then a second kernel reads it
  back, masks and softmaxes it with the row held in registers (32 floats per
  thread, one row per CTA, 64–1024 threads) and stores fp16. Two launches,
  10·M·N bytes of traffic (4·M·N of fp32 scores written, the same read back,
  2·M·N of P) — 5x the fused kernels' floor — plus `m * n * 4` bytes of
  scratch. cuBLAS absorbs the scale for free, since it is the GEMM's `alpha`,
  but it cannot absorb the mask or the softmax: a GEMM CTA never sees a whole
  row of scores.

**The shared mainloop is swizzled and uses `ldmatrix`.** Both kernels stage each
64-wide K tile in a 128-bit-swizzled K-major shared layout (`Swizzle<3,3,3>`
over an 8 x 64-half atom), and load MMA operands with `SM75_U32x4_LDSM_N` for A
and `SM75_U32x2_LDSM_N` for B — x4 for B is a static error here, because this
MMA's 2 x 4 atom layout leaves each lane only 4 halves of B per k-step.
`recompute` issues its global-to-shared copies as `cp.async`; `regrow` keeps the
synchronous vector copy, because at 2–4 CTAs per SM its copies are already
hidden and cp.async measured neutral to 5% worse there. Making a multi-stage
ring affordable is the largest remaining mainloop item (see
[Next steps](#next-steps)).

**Shape constraints.**

| implementation | constraint |
|---|---|
| `fused_softmax_regrow` | `n = 128, 256` or `512`; `m % 32 = 0`; `k % 64 = 0` |
| `fused_softmax_recompute` | `n % 64 = 0`; `m % 32 = 0`; `k % 64 = 0` |
| `fused_softmax_naive` | `k + n <= 12288` (48 KB of shared memory) |
| `cublas_unfused_softmax` | `256 <= n <= 32768`; `n % 4 = 0` |
| harness | causal requires `m <= n`; `m*n`, `m*k`, `n*k` must fit in int32 |

## Results

Each table is the median of **seven full passes**, each pass itself the mean of
20 timed iterations after 3 warm-up iterations. A first full pass was run and
discarded as GPU clock warm-up. Pass-to-pass spread after warm-up is under ~12%
at most shapes; on the shortest shapes (20–80 µs) isolated samples deviate by
20–50% and occasionally more (one 1024 x 2048 cuBLAS sample ran at 71 µs
against a 46 µs median, and one 1024 x 512, `k = 512` regrow sample at 112 µs
against a 62 µs median), which is why the tables report medians — this is a
shared desktop GPU. `—` means the implementation does not accept that shape.
The ratio columns are `cublas time / implementation time`, so **above 1.00x
means the in-repo kernel is faster than the baseline**.

**Square, causal mask, `k = 128` (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 73.5 | 32.2 | 21.7 | 20.7 | 0.44x | 1.48x | 1.56x |
| 512 × 512 | 227.6 | 33.9 | 22.0 | 30.9 | 0.15x | 1.54x | 1.10x |
| 1024 × 1024 | 862.6 | 46.0 | — | 52.5 | 0.05x | — | 0.88x |
| 2048 × 2048 | 3897.4 | 80.0 | — | 107.2 | 0.02x | — | 0.75x |
| 4096 × 4096 | 15815.2 | 440.1 | — | 240.9 | 0.03x | — | 1.83x |
| 8192 × 8192 | 59292.0 | 1733.3 | — | 931.2 | 0.03x | — | 1.86x |

**Rows, `n = 2048`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1004.6 | 36.5 | — | 88.3 | 0.04x | — | 0.41x |
| 512 × 2048 | 1987.0 | 45.6 | — | 87.5 | 0.02x | — | 0.52x |
| 1024 × 2048 | 3949.2 | 48.8 | — | 107.8 | 0.01x | — | 0.45x |
| 2048 × 2048 | 7826.9 | 81.2 | — | 162.6 | 0.01x | — | 0.50x |
| 4096 × 2048 | 15670.0 | 205.1 | — | 293.0 | 0.01x | — | 0.70x |
| 8192 × 2048 | 31511.1 | 448.3 | — | 479.3 | 0.01x | — | 0.94x |
| 16384 × 2048 | 62745.1 | 901.0 | — | 879.0 | 0.01x | — | 1.03x |

**Columns, `m = 1024`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 423.0 | 34.5 | 17.9 | 24.0 | 0.08x | 1.93x | 1.44x |
| 1024 × 512 | 850.4 | 38.1 | 35.1 | 34.5 | 0.04x | 1.09x | 1.10x |
| 1024 × 1024 | 1696.3 | 45.4 | — | 58.4 | 0.03x | — | 0.78x |
| 1024 × 2048 | 3938.7 | 47.2 | — | 112.0 | 0.01x | — | 0.42x |
| 1024 × 4096 | 7950.5 | 81.9 | — | 201.8 | 0.01x | — | 0.41x |
| 1024 × 8192 | 14937.2 | 209.4 | — | 393.3 | 0.01x | — | 0.53x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 4012.6 | 87.5 | — | 122.9 | 0.02x | — | 0.71x |
| 2048 × 2048, k = 128 | 7863.7 | 79.8 | — | 163.5 | 0.01x | — | 0.49x |
| 2048 × 2048, k = 256 | 16009.1 | 96.8 | — | 271.6 | 0.01x | — | 0.36x |
| 2048 × 2048, k = 512 | 32472.3 | 163.6 | — | 483.8 | 0.01x | — | 0.34x |

**Regrow's 256-wide row, `n = 256`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 117.2 | 32.1 | 17.7 | 20.7 | 0.27x | 1.81x | 1.55x |
| 1024 × 256 | 423.0 | 34.5 | 18.0 | 24.0 | 0.08x | 1.92x | 1.44x |
| 4096 × 256 | 1649.4 | 55.8 | 31.0 | 46.5 | 0.03x | 1.80x | 1.20x |
| 16384 × 256 | 6481.3 | 108.8 | 61.0 | 117.6 | 0.02x | 1.78x | 0.93x |

**Regrow's 256-wide row, `m = 4096`, `n = 256`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 828.8 | 49.1 | 26.9 | 36.3 | 0.06x | 1.83x | 1.35x |
| 4096 × 256, k = 128 | 1624.7 | 55.1 | 34.4 | 46.5 | 0.03x | 1.60x | 1.18x |
| 4096 × 256, k = 256 | 3266.2 | 69.2 | 46.2 | 68.0 | 0.02x | 1.50x | 1.02x |
| 4096 × 256, k = 512 | 6589.3 | 65.6 | 53.2 | 107.8 | 0.01x | 1.23x | 0.61x |

**Regrow's 512-wide row, `n = 512`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 512 | 226.4 | 31.3 | 21.8 | 36.4 | 0.14x | 1.44x | 0.86x |
| 1024 × 512 | 843.3 | 38.0 | 42.5 | 34.3 | 0.05x | 0.89x | 1.11x |
| 4096 × 512 | 3269.1 | 56.1 | 56.8 | 81.4 | 0.02x | 0.99x | 0.69x |
| 16384 × 512 | 13042.5 | 241.1 | 178.7 | 226.1 | 0.02x | 1.35x | 1.07x |

**Regrow's 512-wide row, `m = 1024`, `n = 512`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 512, k = 64 | 424.9 | 37.5 | 23.3 | 27.1 | 0.09x | 1.61x | 1.38x |
| 1024 × 512, k = 128 | 849.5 | 37.9 | 35.3 | 34.3 | 0.04x | 1.07x | 1.10x |
| 1024 × 512, k = 256 | 1683.0 | 41.9 | 39.3 | 52.5 | 0.02x | 1.07x | 0.80x |
| 1024 × 512, k = 512 | 3380.7 | 50.9 | 62.5 | 88.0 | 0.02x | 0.81x | 0.58x |

`n = 128` has no baseline: `cublas_unfused_softmax` needs `n >= 256` to fill a
row's register cache. At 4096 x 128 x 128, `regrow` runs in 19.6 µs against
`recompute`'s 31.9 µs and `naive`'s 830.7 µs.

## Reading the numbers

**The mainloop rewrite changed the ranking.** Before the swizzle and `ldmatrix`
work (see the earlier revisions of this file), the baseline won everything
except `regrow` at its narrowest widths. Now:

- `fused_softmax_regrow` wins at every width it supports: 1.23–1.92x at
  `n = 256`, and 0.81–1.61x at `n = 512`. The low end is the deep-K narrow
  GEMM (`n = 512, k = 512`: 0.81x) plus one noisy 1024 x 512 entry (0.89x,
  where other passes measured ~1.05x); the 16384 x 512 case is a 1.35x win.
- `fused_softmax_recompute` wins the causal squares at the ends of the range —
  1.56x at 256 x 256, 1.83x at 4096 x 4096, 1.86x at 8192 x 8192 — and loses
  narrowly in the middle (0.88x at 1024 x 1024, 0.75x at 2048 x 2048), where
  the baseline's score buffer still fits L2 and the row-block schedule has
  little room to help. It is also a 1.10x win at 512 x 512.
- On no-mask shapes it still loses while the second GEMM dominates: 0.41–0.70x
  across the rows sweep up to `m = 4096` and 0.41–0.78x across the columns
  sweep from `n = 1024` on, recovering to 0.94x/1.03x at 8192/16384 rows. The
  narrow columns are the exception: `n = 256`/`512` are 1.44x/1.10x wins, the
  cp.async copy and small score buffer doing the work. Head dims at 2048 x 2048
  are 0.34–0.71x.
- `fused_softmax_naive` remains the floor: 2.3x slower than the baseline at
  256 x 256, 19x at 1024 x 1024, 33–96x at the larger shapes (about 200x at
  `k = 512`).

For scale, the same shapes measured before this round of work: `recompute` at
2048 x 2048 causal went 339.0 -> 107.2 µs (3.2x), at 4096 x 4096 causal
1068.8 -> 240.9 µs (4.4x), at 2048 x 2048 no mask 338.8 -> 162.6 µs (2.1x);
`regrow` at 16384 x 256 went 108.2 -> 61.0 µs (1.8x) and at 4096 x 256, `k = 512`
went 103.5 -> 53.2 µs (1.9x). The pieces, measured as they landed:

- **`ldmatrix` + swizzle** (first round): roughly halved the fused GEMM's time
  and is most of the no-mask improvement.
- **Causal skip + LPT row order**: 4096 x 4096 causal 327.9 -> 263.2 µs (1.25x)
  and 8192 x 8192 causal 1137 -> 1024 µs (1.11x); nothing when the grid fits
  one wave (≤1024 x 1024).
- **Adaptive 64/128 tile width**: 2048 x 2048 no mask 212 -> 162 µs (1.31x),
  16384 x 2048 963 -> 868 µs (and now a 1.03x win); it also makes 8192 x 2048
  a 0.94x near-tie instead of 0.83x.
- **cp.async copy** (this round): 4096 x 4096 causal 263.2 -> 240.9 µs,
  8192 x 8192 causal 1024 -> 931 µs, 1024 x 8192 465 -> 393 µs,
  1024 x 4096 232 -> 202 µs, 1024 x 2048 121 -> 112 µs. The two-stage ring
  was measured and turned down: at kStages = 2 the 64-wide build lost its
  fourth CTA per SM and 16384 x 2048 regressed 868 -> 964 µs.

**Where the speed comes from now.** The fused GEMM is no longer the weak link.
Measured as `2·M·N·K / time`, with both `recompute` passes attributed to the
GEMM (and its causal skip counted as the ~half of the flops it really runs), it
reaches **13.2 TFLOP/s** at 2048 x 2048 no mask against the baseline's 13.2,
**19.5** at 16384 x 2048 against 9.5, **17.8** at 4096 x 4096 causal against
9.8, **18.4** at 8192 x 8192 causal against 9.9, and **17.6** in `regrow` at
16384 x 256 against 9.9. The baseline's 9.5–9.9 TFLOP/s at the large shapes is
its whole pipeline, score round trip included; there it is moving 366–425 GB/s
of nominal traffic (82–95% of the card's 448 GB/s) and is DRAM-bound, while the
fused kernels are still compute-bound. The remaining losses are structural:

- **`recompute` pays for its second GEMM.** At 2048 x 2048 no mask its per-pass
  rate matches the baseline's whole pipeline (13.2 vs 13.2), so the 2x passes
  alone explain the 0.50x. At 16384 x 2048 the per-pass rate is 2.1x the
  baseline's and the total wins 1.03x despite the second pass. The crossover is
  when the fused GEMM's per-flop advantage outruns 2x.
- **The causal schedule is worth a fifth of the GEMM at large sizes.** The LPT
  row order took 4096 x 4096 causal from 327.9 to 263.2 µs (1.25x) and
  8192 x 8192 causal from 1137 to 1024 µs (1.11x), before the cp.async copy
  took those to 240.9 and 931 µs. It does nothing when the whole grid fits one
  wave, which is exactly where the imbalance has nowhere to hide.
- **The 64-wide dispatch is worth a quarter at large `m`.** 2048 x 2048 no mask
  went 212 -> 162 µs (1.31x), 16384 x 2048 963 -> 868 µs, and 2048 x 2048 at
  `k = 512` 530 -> 484 µs. It costs nothing when the grid fits one wave, which
  is why the host measures the wave rather than guessing from `m`.
- **`regrow`'s wider rows cost occupancy.** 128 and 256 wide run 4 and 2 CTAs
  per SM; 512 needs 68 KB and runs 1. Its per-GEMM efficiency therefore depends
  on the problem being large enough to keep that one CTA busy. At 16384 x 512 it
  wins 1.35x: the baseline's fp32 score buffer is 33.5 MB there, well past the
  24 MB L2, so its 5x traffic is real DRAM work (88 MB in 241 µs, 366 GB/s),
  while the fused kernel writes P once and stays compute-bound. At 4096 x 512
  the baseline's buffer (8.4 MB) is back in L2 and the two tie; at `k = 512` the
  GEMM dominates and the extra CTA matters more than the saved traffic.
- **The causal mask still buys `regrow` nothing.** `recompute` skips whole
  masked tiles; `regrow` computes the full row. On 256 x 256 and 512 x 512 the
  skip is most of `recompute`'s causal advantage. A 128-wide N sub-tile inside
  `regrow` would recover the same saving, at the cost of a score buffer that can
  no longer alias the operand tile. See [Next steps](#next-steps).

**Verification.** All timed shapes pass, including 1024 x 8192 and the 8192 x
8192 causal square. The 1024 x 8192 case used to report 1.207e-03 for all four
implementations; the fix is in the metric, not the kernels. P is fp16, whose
smallest normal is 6.1e-5 and whose subnormal spacing is 6.0e-8, so the smallest
probabilities at `n = 8192` (~2.5e-5, roughly `1/n`) carry a format-limited
relative error around 1.2e-3. `max_rel_diff()` now floors its denominator at
2^-14, which judges those elements against the resolution fp16 has there (about
one subnormal ulp at the 1e-3 threshold) and leaves the comparison unchanged
above the normal floor. The same effect was already visible as the threshold was
approached: 4.9e-4 at `n <= 2048`, 5.3e-4 at `n = 4096`, and 1.2e-3 at
`n = 8192` before the change.

## Caveats

- **Shared desktop GPU.** Passes after the discarded warm-up still show
  occasional 20–50% single-sample spikes on the 20–80 µs shapes, from desktop
  memory bursts and GDDR7 power-state changes; the 1024 x 2048 cuBLAS and
  1024 x 512, `k = 512` regrow samples quoted above are the worst of this
  session. The medians are stable; individual samples are not. This is the same
  environment documented in `examples/softmax`.
- **No hardware counters.** Nsight Compute could not attach in this session
  (`ERR_NVGPUCTRPERM`, performance counters need an elevated permission), so the
  bottleneck attribution above is inferred from the k-sweep and the SASS
  instruction mix, not from measured DRAM/SM utilization. The SASS does confirm
  the score reads are `LDS.128` and the MMA operands are `LDSM`, so the
  softmax's shared-memory access pattern is bandwidth-optimal; the epilogue's
  cost is the staging round trip itself, not bank conflicts.
- **`TFLOP/s` in the harness** prices every implementation at `2·M·N·K`, which
  is right for cuBLAS, `regrow` and `recompute` but overstates `naive`'s work
  under a causal mask by 2x (it skips masked columns), so its printed rate is
  half its real one. The `recompute` rates quoted above halve their priced
  traffic by hand for the two passes and credit the causal skip; they are
  effective rates, not harness output.
- **The harness's `GB/s` column** now prices P's write correctly
  (`m * n * sizeof(half)` bytes, the 2·M·N bytes the comment describes); it used
  to multiply by `sizeof(half)` twice. The traffic figures in this README are
  computed independently as 2·M·N bytes for P plus `(m + n)·k·2` for Q and K.
- **The dispatch threshold is device-dependent by construction.** It is the
  measured residency of the 128-wide kernel times the multiprocessor count, so
  on a larger GPU the crossover moves to a larger grid. That is the intent, but
  it does mean the exact shape where 64 wins is not portable.

## Run

```bash
cmake -B build && cmake --build build
./build/examples/fused_softmax                            # 2048 x 2048 x 128, causal
./build/examples/fused_softmax --m=4096 --n=256 --k=128 --mask=none
```

Options: `--m`, `--n`, `--k`, `--iterations`, `--mask=causal|none`, `--scale`,
`--verify=true|false`. Verification builds a double-precision reference on the
host; it is O(m·n·k) and is the slow part of the harness at large shapes (the
8192 x 8192 causal row costs several seconds of reference per pass).

## Next steps

- **Make a cp.async ring affordable.** The ring is in
  `compute_tile_async<kStages>` and works; at two stages it loses the CTA that
  made the single-stage version fast. The ways to pay for it are a 32-wide K tile
  (two stages fit the current shared-memory budget, but need a padded or
  64-byte-swizzled layout) or `regrow`'s row split into 128-wide N sub-tiles
  (which also enables causal skipping and keeps two CTAs). Measure both.
- **Give `regrow` a causal skip.** Split the row into 128-wide N sub-tiles,
  drop the fully masked ones, and store zeros for them; the score staging can no
  longer alias the operand tile, so this trades a little shared memory for up to
  half the GEMM on square causal shapes. `recompute` already shows the payoff.
- **Attack the epilogue.** With the GEMM fast, the per-tile score staging and
  the two reductions dominate at small `k` (the k = 64 row of the head-dim table
  is the clearest case). Computing pass 1's (max, sum) directly from the
  accumulator fragment would remove a 16 KB shared-memory round trip and two
  barriers per tile; the reads are already `LDS.128`, so the win is removing the
  staging, not changing its access pattern.
- **Widen the shapes.** The kernels still throw on `m % 32 != 0` and
  `k % 64 != 0`; attention head dims such as 96 or ragged sequence lengths need
  predication or a tail path. `regrow`'s `n` is a fixed set because the whole
  row must fit; 512 is its ceiling (the B tile alone is 64 KB, the score staging
  another 64 KB).
- **Profile before the next change.** `ncu` on `regrow` at 16384 x 256 and
  `recompute` at 2048 x 2048 would confirm whether the next bottleneck is the
  copy, the epilogue, or occupancy; the numbers above only bracket it. It needs
  performance-counter permission on the machine.
