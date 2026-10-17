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
  with the fewest CTAs, where the copy would otherwise be exposed. Five
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
  - **Last-tile reuse.** Pass 1 leaves its last active tile's scores in shared
    memory, and the row statistics are final by then, so pass 2 normalizes that
    tile in place instead of recomputing it. One GEMM and one copy per CTA are
    saved: a third of the second pass at `n = 256`, and all of it at `n = 128`
    (4096 x 128 x 128: 36.5 -> 28.3 µs). At large `n` it converges to the
    1/nTiles share (3% at 1024 x 8192).
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
20–50% and occasionally more (one 1024 x 2048 cuBLAS sample ran at 105 µs
against a 47 µs median, and one 2048 x 2048, `k = 64` recompute sample at
295 µs against a 123 µs median), which is why the tables report medians — this
is a shared desktop GPU. `—` means the implementation does not accept that
shape. The ratio columns are `cublas time / implementation time`, so **above
1.00x means the in-repo kernel is faster than the baseline**.

**Square, causal mask, `k = 128` (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 73.6 | 32.4 | 17.5 | 18.7 | 0.44x | 1.85x | 1.73x |
| 512 × 512 | 227.8 | 34.5 | 21.1 | 33.1 | 0.15x | 1.64x | 1.04x |
| 1024 × 1024 | 848.5 | 45.5 | — | 48.6 | 0.05x | — | 0.94x |
| 2048 × 2048 | 3881.7 | 80.6 | — | 96.9 | 0.02x | — | 0.83x |
| 4096 × 4096 | 15745.4 | 435.8 | — | 236.4 | 0.03x | — | 1.84x |
| 8192 × 8192 | 58868.5 | 1723.3 | — | 927.3 | 0.03x | — | 1.86x |

**Rows, `n = 2048`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1009.4 | 36.9 | — | 86.4 | 0.04x | — | 0.43x |
| 512 × 2048 | 1981.1 | 45.3 | — | 87.0 | 0.02x | — | 0.52x |
| 1024 × 2048 | 3953.1 | 47.1 | — | 105.0 | 0.01x | — | 0.45x |
| 2048 × 2048 | 7817.2 | 84.9 | — | 167.7 | 0.01x | — | 0.51x |
| 4096 × 2048 | 15624.2 | 207.3 | — | 285.3 | 0.01x | — | 0.73x |
| 8192 × 2048 | 31336.8 | 447.2 | — | 468.7 | 0.01x | — | 0.95x |
| 16384 × 2048 | 62692.6 | 894.3 | — | 859.5 | 0.01x | — | 1.04x |

**Columns, `m = 1024`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 423.2 | 34.4 | 17.7 | 21.9 | 0.08x | 1.94x | 1.57x |
| 1024 × 512 | 844.0 | 38.9 | 36.3 | 32.2 | 0.05x | 1.07x | 1.21x |
| 1024 × 1024 | 1683.3 | 44.9 | — | 56.1 | 0.03x | — | 0.80x |
| 1024 × 2048 | 3917.8 | 55.7 | — | 104.8 | 0.01x | — | 0.53x |
| 1024 × 4096 | 7929.9 | 82.5 | — | 195.5 | 0.01x | — | 0.42x |
| 1024 × 8192 | 14906.6 | 207.6 | — | 380.4 | 0.01x | — | 0.55x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 3908.8 | 98.0 | — | 124.8 | 0.03x | — | 0.79x |
| 2048 × 2048, k = 128 | 7825.3 | 80.3 | — | 167.0 | 0.01x | — | 0.48x |
| 2048 × 2048, k = 256 | 16089.2 | 96.0 | — | 265.2 | 0.01x | — | 0.36x |
| 2048 × 2048, k = 512 | 32413.4 | 155.7 | — | 462.2 | 0.00x | — | 0.34x |

**Regrow's 256-wide row, `n = 256`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 117.1 | 32.2 | 20.1 | 18.7 | 0.27x | 1.60x | 1.72x |
| 1024 × 256 | 423.1 | 34.2 | 18.1 | 22.3 | 0.08x | 1.89x | 1.53x |
| 4096 × 256 | 1618.7 | 55.5 | 35.2 | 42.5 | 0.03x | 1.58x | 1.31x |
| 16384 × 256 | 6459.6 | 109.3 | 63.5 | 107.7 | 0.02x | 1.72x | 1.01x |

**Regrow's 256-wide row, `m = 4096`, `n = 256`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 824.7 | 49.3 | 23.3 | 34.1 | 0.06x | 2.12x | 1.45x |
| 4096 × 256, k = 128 | 1634.3 | 55.1 | 32.0 | 42.5 | 0.03x | 1.72x | 1.30x |
| 4096 × 256, k = 256 | 3239.2 | 58.6 | 47.2 | 66.5 | 0.02x | 1.24x | 0.88x |
| 4096 × 256, k = 512 | 6579.9 | 65.8 | 53.9 | 94.0 | 0.01x | 1.22x | 0.70x |

**Regrow's 512-wide row, `n = 512`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 512 | 225.6 | 32.5 | 21.3 | 34.5 | 0.14x | 1.53x | 0.94x |
| 1024 × 512 | 855.5 | 38.5 | 33.0 | 32.1 | 0.05x | 1.17x | 1.20x |
| 4096 × 512 | 3284.0 | 56.2 | 56.9 | 77.6 | 0.02x | 0.99x | 0.72x |
| 16384 × 512 | 13074.8 | 241.9 | 178.7 | 214.6 | 0.02x | 1.35x | 1.13x |

**Regrow's 512-wide row, `m = 1024`, `n = 512`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 512, k = 64 | 425.5 | 37.6 | 23.5 | 28.4 | 0.09x | 1.60x | 1.32x |
| 1024 × 512, k = 128 | 843.0 | 38.2 | 35.8 | 32.2 | 0.05x | 1.07x | 1.19x |
| 1024 × 512, k = 256 | 1702.8 | 42.1 | 39.2 | 47.6 | 0.02x | 1.07x | 0.88x |
| 1024 × 512, k = 512 | 3378.0 | 56.1 | 61.4 | 84.5 | 0.02x | 0.91x | 0.66x |

`n = 128` has no baseline: `cublas_unfused_softmax` needs `n >= 256` to fill a
row's register cache. At 4096 x 128 x 128, `regrow` runs in 20.1 µs against
`recompute`'s 28.3 µs and `naive`'s 829.6 µs.

## Reading the numbers

**The mainloop rewrite changed the ranking.** Before the swizzle and `ldmatrix`
work (see the earlier revisions of this file), the baseline won everything
except `regrow` at its narrowest widths. Now:

- `fused_softmax_regrow` wins at every width it supports: 1.22–2.12x at
  `n = 256`, and 0.91–1.60x at `n = 512`. The low end is the deep-K narrow
  GEMM (`n = 512, k = 512`: 0.91x) and a tie at 4096 x 512 (0.99x); the
  16384 x 512 case is a 1.35x win.
- `fused_softmax_recompute` wins the causal squares at the ends of the range —
  1.73x at 256 x 256, 1.84x at 4096 x 4096, 1.86x at 8192 x 8192 — is a 1.04x
  win at 512 x 512, and loses narrowly in the middle (0.94x at 1024 x 1024,
  0.83x at 2048 x 2048), where the baseline's score buffer still fits L2 and
  the row-block schedule has little room to help.
- On no-mask shapes it still loses while the second GEMM dominates: 0.43–0.73x
  across the rows sweep up to `m = 4096` and 0.42–0.80x across the columns
  sweep from `n = 1024` on, recovering to 0.95x/1.04x at 8192/16384 rows. The
  narrow columns are the exception: `n = 256`/`512` are 1.57x/1.21x wins, the
  cp.async copy and small score buffer doing the work. Head dims at 2048 x 2048
  are 0.34–0.79x.
- `fused_softmax_naive` remains the floor: 2.3x slower than the baseline at
  256 x 256, 19x at 1024 x 1024, 33–92x at the larger shapes (about 200x at
  `k = 512`).

For scale, the same shapes measured before this round of work: `recompute` at
2048 x 2048 causal went 339.0 -> 96.9 µs (3.5x), at 4096 x 4096 causal
1068.8 -> 236.4 µs (4.5x), at 2048 x 2048 no mask 338.8 -> 167.7 µs (2.0x);
`regrow` at 16384 x 256 went 108.2 -> 63.5 µs (1.7x) and at 4096 x 256, `k = 512`
went 103.5 -> 53.9 µs (1.9x). The pieces, measured as they landed:

- **`ldmatrix` + swizzle** (first round): roughly halved the fused GEMM's time
  and is most of the no-mask improvement.
- **Causal skip + LPT row order**: 4096 x 4096 causal 327.9 -> 263.2 µs (1.25x)
  and 8192 x 8192 causal 1137 -> 1024 µs (1.11x); nothing when the grid fits
  one wave (≤1024 x 1024).
- **Adaptive 64/128 tile width**: 2048 x 2048 no mask 212 -> 162 µs (1.31x),
  16384 x 2048 963 -> 868 µs (and now a 1.04x win); it also makes 8192 x 2048
  a 0.95x near-tie instead of 0.83x.
- **cp.async copy + last-tile reuse** (this round): 4096 x 4096 causal
  263.2 -> 236.4 µs, 8192 x 8192 causal 1024 -> 927 µs, 1024 x 8192
  465 -> 380 µs, 1024 x 4096 232 -> 196 µs, 1024 x 2048 121 -> 105 µs. The
  two-stage ring was measured and turned down: at kStages = 2 the 64-wide build
  lost its fourth CTA per SM and 16384 x 2048 regressed 868 -> 964 µs.
- **Last-tile reuse** alone is 20% at `n = 128` (4096 x 128 x 128: 33.8 ->
  26.9 µs, the entire second GEMM), 6–10% at `n = 256/512` (1024 x 256:
  23.4 -> 21.9, 1024 x 512: 34.5 -> 32.2), and ~4% on the causal squares
  (2048 x 2048: 107 -> 97); at 64 tiles it is the 1/64 share.

**Where the speed comes from now.** The fused GEMM is no longer the weak link.
Measured as `2·M·N·K / time`, with both `recompute` passes attributed to the
GEMM (and its causal skip counted as the ~half of the flops it really runs), it
reaches **12.8 TFLOP/s** at 2048 x 2048 no mask against the baseline's 12.7,
**20.0** at 16384 x 2048 against 9.6, **18.2** at 4096 x 4096 causal against
9.9, **18.5** at 8192 x 8192 causal against 10.0, and **16.9** in `regrow` at
16384 x 256 against 9.8. The baseline's 8.9–10.0 TFLOP/s at the large shapes is
its whole pipeline, score round trip included; there it is moving 365–423 GB/s
of nominal traffic (81–94% of the card's 448 GB/s) and is DRAM-bound, while the
fused kernels are still compute-bound. The remaining losses are structural:

- **`recompute` pays for its second GEMM.** At 2048 x 2048 no mask its per-pass
  rate matches the baseline's whole pipeline (12.8 vs 12.7), so the 2x passes
  alone explain the 0.51x. At 16384 x 2048 the per-pass rate is 2.1x the
  baseline's and the total wins 1.04x despite the second pass. The crossover is
  when the fused GEMM's per-flop advantage outruns 2x.
- **The causal schedule is worth a fifth of the GEMM at large sizes.** The LPT
  row order took 4096 x 4096 causal from 327.9 to 263.2 µs (1.25x) and
  8192 x 8192 causal from 1137 to 1024 µs (1.11x), before the cp.async copy and
  last-tile reuse took those to 236.4 and 927 µs. It does nothing when the
  whole grid fits one wave, which is exactly where the imbalance has nowhere to
  hide.
- **The 64-wide dispatch is worth a quarter at large `m`.** 2048 x 2048 no mask
  went 212 -> 162 µs (1.31x), 16384 x 2048 963 -> 868 µs, and 2048 x 2048 at
  `k = 512` 530 -> 462 µs. It costs nothing when the grid fits one wave, which
  is why the host measures the wave rather than guessing from `m`.
- **`regrow`'s wider rows cost occupancy.** 128 and 256 wide run 4 and 2 CTAs
  per SM; 512 needs 68 KB and runs 1. Its per-GEMM efficiency therefore depends
  on the problem being large enough to keep that one CTA busy. At 16384 x 512 it
  wins 1.35x: the baseline's fp32 score buffer is 33.5 MB there, well past the
  24 MB L2, so its 5x traffic is real DRAM work (88 MB in 242 µs, 365 GB/s),
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
  2048 x 2048, `k = 64` recompute samples quoted above are the worst of this
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
  accumulator fragment would remove the 16 KB score staging and the warp-per-row
  read from pass 1 entirely; last-tile reuse means only the final tile still
  needs to be staged. The reads are already `LDS.128`, so the win is removing the
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
