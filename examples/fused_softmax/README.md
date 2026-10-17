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
  pass folds each tile into the row's running `(max, sum)` and throws the tile
  away; the second recomputes it and writes `exp(s - max) / sum`. It keeps the
  2·M·N floor for any `n` that is a multiple of 64 and pays for it with 2x the
  GEMM multiply-adds instead of extra traffic. The global-to-shared copies are
  `cp.async`, one commit group per K tile and one wait before the read; that is
  worth 7–18% on the shapes with the fewest CTAs, where the copy would otherwise
  be exposed. The first pass also folds each tile straight out of the
  accumulator — per-thread partials, a four-lane shuffle, and one shared partial
  per (row, N atom), double-buffered by tile parity — so **no score staging is
  written or read** except for the one tile pass 2 reuses; at `k = 64` it took
  the ratio against the baseline from 0.79x to 0.92x, a 24% total win.
  Five scheduling choices matter as much as the arithmetic:
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

Each table is the median of **ten full passes**, each pass itself the mean of
20 timed iterations after 3 warm-up iterations. A first full pass was run and
discarded as GPU clock warm-up. Pass-to-pass spread after warm-up is under ~12%
at most shapes; the shortest shapes are noisier and occasionally bimodal (one
512 x 512 causal recompute run measured 29 µs and another 40 µs, with a 37 µs
ten-pass median), which is why the tables report medians — this is a shared
desktop GPU. `—` means the implementation does not accept that shape. The
ratio columns are `cublas time / implementation time`, so **above 1.00x means
the in-repo kernel is faster than the baseline**.

**Square, causal mask, `k = 128` (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 73.0 | 32.0 | 21.8 | 18.4 | 0.44x | 1.47x | 1.73x |
| 512 × 512 | 227.7 | 34.0 | 22.0 | 36.9 | 0.15x | 1.54x | 0.92x |
| 1024 × 1024 | 848.6 | 45.4 | — | 44.4 | 0.05x | — | 1.02x |
| 2048 × 2048 | 3904.7 | 79.8 | — | 87.7 | 0.02x | — | 0.91x |
| 4096 × 4096 | 15722.7 | 430.5 | — | 208.1 | 0.03x | — | 2.07x |
| 8192 × 8192 | 58666.6 | 1713.7 | — | 829.2 | 0.03x | — | 2.07x |

**Rows, `n = 2048`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1005.0 | 36.4 | — | 76.1 | 0.04x | — | 0.48x |
| 512 × 2048 | 1984.2 | 45.0 | — | 75.2 | 0.02x | — | 0.60x |
| 1024 × 2048 | 3931.5 | 58.0 | — | 96.8 | 0.01x | — | 0.60x |
| 2048 × 2048 | 7814.5 | 80.2 | — | 146.2 | 0.01x | — | 0.55x |
| 4096 × 2048 | 15606.6 | 204.4 | — | 235.2 | 0.01x | — | 0.87x |
| 8192 × 2048 | 31259.1 | 443.1 | — | 387.1 | 0.01x | — | 1.14x |
| 16384 × 2048 | 62620.6 | 894.4 | — | 720.9 | 0.01x | — | 1.24x |

**Columns, `m = 1024`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 422.9 | 34.5 | 18.0 | 19.1 | 0.08x | 1.92x | 1.81x |
| 1024 × 512 | 843.0 | 38.0 | 33.1 | 30.6 | 0.05x | 1.14x | 1.24x |
| 1024 × 1024 | 1684.6 | 45.4 | — | 51.2 | 0.03x | — | 0.89x |
| 1024 × 2048 | 3936.7 | 61.5 | — | 94.5 | 0.02x | — | 0.65x |
| 1024 × 4096 | 7894.1 | 80.6 | — | 178.6 | 0.01x | — | 0.45x |
| 1024 × 8192 | 14888.2 | 203.3 | — | 347.6 | 0.01x | — | 0.58x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 3886.3 | 86.8 | — | 94.8 | 0.02x | — | 0.92x |
| 2048 × 2048, k = 128 | 7822.2 | 81.0 | — | 146.6 | 0.01x | — | 0.55x |
| 2048 × 2048, k = 256 | 15882.1 | 95.7 | — | 246.9 | 0.01x | — | 0.39x |
| 2048 × 2048, k = 512 | 32333.8 | 152.4 | — | 446.0 | 0.00x | — | 0.34x |

**Regrow's 256-wide row, `n = 256`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 117.0 | 32.5 | 18.6 | 17.2 | 0.28x | 1.74x | 1.88x |
| 1024 × 256 | 422.4 | 34.5 | 18.0 | 19.2 | 0.08x | 1.91x | 1.79x |
| 4096 × 256 | 1623.7 | 54.0 | 31.3 | 36.2 | 0.03x | 1.73x | 1.49x |
| 16384 × 256 | 6459.9 | 109.8 | 61.1 | 90.4 | 0.02x | 1.80x | 1.21x |

**Regrow's 256-wide row, `m = 4096`, `n = 256`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 825.7 | 49.8 | 23.4 | 29.0 | 0.06x | 2.13x | 1.72x |
| 4096 × 256, k = 128 | 1621.5 | 55.6 | 30.2 | 36.2 | 0.03x | 1.84x | 1.54x |
| 4096 × 256, k = 256 | 3242.4 | 62.6 | 41.6 | 61.2 | 0.02x | 1.50x | 1.02x |
| 4096 × 256, k = 512 | 6587.8 | 65.8 | 53.2 | 88.6 | 0.01x | 1.24x | 0.74x |

**Regrow's 512-wide row, `n = 512`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 512 | 224.9 | 32.1 | 21.4 | 37.2 | 0.14x | 1.50x | 0.86x |
| 1024 × 512 | 842.1 | 37.8 | 36.0 | 29.9 | 0.04x | 1.05x | 1.26x |
| 4096 × 512 | 3268.9 | 56.1 | 56.8 | 64.7 | 0.02x | 0.99x | 0.87x |
| 16384 × 512 | 13013.7 | 240.1 | 178.7 | 179.1 | 0.02x | 1.34x | 1.34x |

**Regrow's 512-wide row, `m = 1024`, `n = 512`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 512, k = 64 | 425.0 | 36.8 | 24.5 | 23.4 | 0.09x | 1.50x | 1.57x |
| 1024 × 512, k = 128 | 841.9 | 37.9 | 35.9 | 30.0 | 0.05x | 1.06x | 1.26x |
| 1024 × 512, k = 256 | 1678.8 | 41.9 | 39.2 | 44.5 | 0.02x | 1.07x | 0.94x |
| 1024 × 512, k = 512 | 3366.2 | 43.8 | 59.8 | 73.1 | 0.01x | 0.73x | 0.60x |

`n = 128` has no baseline: `cublas_unfused_softmax` needs `n >= 256` to fill a
row's register cache. At 4096 x 128 x 128, `regrow` runs in 20.0 µs against
`recompute`'s 23.4 µs and `naive`'s 826.5 µs (recompute was 28.3 µs before the
register-based statistics, and 36.5 µs before the last-tile reuse).

## Reading the numbers

**The mainloop and epilogue rewrites changed the ranking.** Before the swizzle
and `ldmatrix` work (see the earlier revisions of this file), the baseline won
everything except `regrow` at its narrowest widths. Now:

- `fused_softmax_regrow` wins at every width it supports: 1.24–2.13x at
  `n = 256`, and 0.73–1.57x at `n = 512`. The low end is the deep-K narrow
  GEMM (`n = 512, k = 512`: 0.73x) and a tie at 4096 x 512 (0.99x); the
  16384 x 512 case is a 1.34x win.
- `fused_softmax_recompute` wins the causal squares at both ends of the range —
  1.73x at 256 x 256, 2.07x at 4096 x 4096, 2.07x at 8192 x 8192 — and ties
  1024 x 1024 (1.02x). It still loses narrowly at 512 x 512 (0.92x, a bimodal
  shape this session) and 2048 x 2048 (0.91x), where the baseline's score
  buffer still fits L2 and the row-block schedule has little room to help.
- On no-mask shapes it has mostly caught up: 1.14x at 8192 x 2048, 1.24x at
  16384 x 2048, 0.87x at 4096 x 2048, 0.48–0.65x across the no-mask rows up to
  2048 rows, 0.45–0.58x across the wide columns from `n = 4096` on, and
  0.34–0.92x across head dims. The narrow columns are wins: `n = 256`/`512` are 1.81x/1.24x, and
  `k = 64` at 2048 x 2048 is 0.92x — the register-based epilogue is what closed
  that gap.
- `fused_softmax_naive` remains the floor: 2.3x slower than the baseline at
  256 x 256, 19x at 1024 x 1024, 33–97x at the larger shapes (about 210x at
  `k = 512`).

For scale, the same shapes measured before this round of work: `recompute` at
2048 x 2048 causal went 339.0 -> 87.7 µs (3.9x), at 4096 x 4096 causal
1068.8 -> 208.1 µs (5.1x), at 2048 x 2048 no mask 338.8 -> 146.2 µs (2.3x);
`regrow` at 16384 x 256 went 108.2 -> 61.1 µs (1.8x) and at 4096 x 256, `k = 512`
went 103.5 -> 53.2 µs (1.9x). The pieces, measured as they landed:

- **`ldmatrix` + swizzle** (first round): roughly halved the fused GEMM's time
  and is most of the no-mask improvement.
- **Causal skip + LPT row order**: 4096 x 4096 causal 327.9 -> 263.2 µs (1.25x)
  and 8192 x 8192 causal 1137 -> 1024 µs (1.11x); nothing when the grid fits
  one wave (≤1024 x 1024).
- **Adaptive 64/128 tile width**: 2048 x 2048 no mask 212 -> 162 µs (1.31x),
  16384 x 2048 963 -> 868 µs; 8192 x 2048, 0.83x before this work, is 1.14x
  now that the epilogue work has landed too.
- **cp.async copy + last-tile reuse**: 4096 x 4096 causal 263.2 -> 236.4 µs,
  8192 x 8192 causal 1024 -> 927 µs, 1024 x 8192 465 -> 380 µs, 1024 x 4096
  232 -> 196 µs, 1024 x 2048 121 -> 105 µs. The two-stage ring was measured and
  turned down: at kStages = 2 the 64-wide build lost its fourth CTA per SM and
  16384 x 2048 regressed 868 -> 964 µs.
- **Register-based pass-1 statistics** (this round): 2048 x 2048 no mask
  167.7 -> 146.2 µs, 4096 x 4096 causal 236.4 -> 208.1, 8192 x 8192 causal
  927 -> 829, 1024 x 8192 380 -> 348, 16384 x 2048 860 -> 721 (now 1.24x),
  2048 x 2048, `k = 64` 125 -> 95 µs (now 0.92x), and 4096 x 128 x 128
  28.3 -> 23.4. A first attempt with three barriers per tile made 512 x 512
  causal about 10% slower; double-buffering the partial arrays removed the extra
  barrier and the clean-mode time returned to ~33 µs (the ten-pass median is
  higher because that shape was bimodal this session).

**Where the speed comes from now.** The fused GEMM is no longer the weak link.
Measured as `2·M·N·K / time`, with both `recompute` passes attributed to the
GEMM (and its causal skip counted as the ~half of the flops it really runs), it
reaches **14.7 TFLOP/s** at 2048 x 2048 no mask against the baseline's 13.4,
**23.8** at 16384 x 2048 against 9.6, **20.6** at 4096 x 4096 causal against
10.0, **20.7** at 8192 x 8192 causal against 10.0, and **17.6** in `regrow` at
16384 x 256 against 9.8. The baseline's 8.9–10.0 TFLOP/s at the large shapes is
its whole pipeline, score round trip included; there it is moving 367–421 GB/s
of nominal traffic (82–94% of the card's 448 GB/s) and is DRAM-bound, while the
fused kernels are still compute-bound. The remaining losses are structural:

- **`recompute` pays for its second GEMM.** At 2048 x 2048 no mask its per-pass
  rate is now 1.1x the baseline's whole pipeline (14.7 vs 13.4), so the 2x
  passes alone explain the 0.55x. At 16384 x 2048 the per-pass rate is 2.5x the
  baseline's and the total wins 1.24x despite the second pass. The crossover is
  when the fused GEMM's per-flop advantage outruns 2x.
- **The causal schedule is worth a fifth of the GEMM at large sizes.** The LPT
  row order took 4096 x 4096 causal from 327.9 to 263.2 µs (1.25x) and
  8192 x 8192 causal from 1137 to 1024 µs (1.11x), before cp.async, last-tile
  reuse and the register epilogue took those to 208 and 829 µs. It does nothing
  when the whole grid fits one wave, which is exactly where the imbalance has
  nowhere to hide.
- **The 64-wide dispatch is worth a quarter at large `m`.** 2048 x 2048 no mask
  went 212 -> 162 µs (1.31x), 16384 x 2048 963 -> 868 µs, and 2048 x 2048 at
  `k = 512` 530 -> 446 µs. It costs nothing when the grid fits one wave, which
  is why the host measures the wave rather than guessing from `m`.
- **`regrow`'s wider rows cost occupancy.** 128 and 256 wide run 4 and 2 CTAs
  per SM; 512 needs 68 KB and runs 1. Its per-GEMM efficiency therefore depends
  on the problem being large enough to keep that one CTA busy. At 16384 x 512 it
  wins 1.34x: the baseline's fp32 score buffer is 33.5 MB there, well past the
  24 MB L2, so its 5x traffic is real DRAM work (88 MB in 240 µs, 367 GB/s),
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
  occasional 20–50% single-sample spikes on the 20–80 µs shapes, and 512 x 512
  causal was bimodal this session (29–40 µs for the same kernel); the ten-pass
  medians are the slow mode there. This is the same environment documented in
  `examples/softmax`.
- **No hardware counters.** Nsight Compute could not attach in this session
  (`ERR_NVGPUCTRPERM`, performance counters need an elevated permission), so the
  bottleneck attribution above is inferred from the k-sweep and the SASS
  instruction mix, not from measured DRAM/SM utilization. The SASS does confirm
  the score reads in the staged path are `LDS.128` and the MMA operands are
  `LDSM`; the register epilogue removed the pass-1 score reads entirely.
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

- **Fuse pass 2's staging too.** Pass 2 still writes each tile's scores to shared
  memory (after the GEMM) and reads them back to normalize; the last-tile reuse
  only avoids the recompute. Writing P directly from the accumulator would
  remove that round trip at the cost of less-coalesced half2 stores (the
  fragment gives 16-byte row segments where the staged path gives 128-byte), so
  measure it rather than assume.
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
- **Widen the shapes.** The kernels still throw on `m % 32 != 0` and
  `k % 64 != 0`; attention head dims such as 96 or ragged sequence lengths need
  predication or a tail path. `regrow`'s `n` is a fixed set because the whole
  row must fit; 512 is its ceiling (the B tile alone is 64 KB, the score staging
  another 64 KB).
- **Profile before the next change.** `ncu` on `regrow` at 16384 x 256 and
  `recompute` at 2048 x 2048 would confirm whether the next bottleneck is the
  copy, the epilogue, or occupancy; the numbers above only bracket it. It needs
  performance-counter permission on the machine.
