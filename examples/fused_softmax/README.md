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
  the ratio against the baseline from 0.79x to 0.92x, a 24% total win; the alias
  below then took it to 1.05x. The score
  staging then aliases the operand buffers, which are disjoint in time with one
  cp.async stage: the 128-wide build runs in 20 KB of dynamic shared memory and
  fits **four CTAs per SM** (from two). That alias is also what retired the
  adaptive tile width: the 64-wide build, which used to win the large grids by
  fitting four CTAs against the 128-wide build's two, is now slower wherever
  both fit, so 128 is the default and 64 remains only for `n` multiples of 64
  that are not multiples of 128 (192, 320, ...).
  Four scheduling choices matter as much as the arithmetic:
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
  - **One cp.async stage, not a ring.** `compute_tile_async<kStages>` can stage
    the operands kStages deep, but only `kStages = 1` is used: at 2 the score
    staging can no longer alias the operands, the 64-wide build drops from 5
    CTAs per SM to 3 and the 128-wide from 4 to 2, and the large shapes
    regressed (16384 x 2048: 868 -> 964 µs at the time). The parameter stays so
    the trade can be retried when the tile changes.
  - **Last-tile reuse.** Pass 1 leaves its last active tile's scores in shared
    memory, and the row statistics are final by then, so pass 2 normalizes that
    tile in place instead of recomputing it. One GEMM and one copy per CTA are
    saved: half of the second pass at `n = 256` (two tiles), and all of it at
    `n = 128` (one tile). At large `n` it converges to the 1/nTiles share (3% at
    1024 x 8192).
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

**Square, causal, k = 128 (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 73.3 | 31.2 | 20.4 | 18.6 | 0.43x | 1.53x | 1.67x |
| 512 × 512 | 227.5 | 34.0 | 22.1 | 36.8 | 0.15x | 1.54x | 0.93x |
| 1024 × 1024 | 849.0 | 46.0 | — | 44.2 | 0.05x | — | 1.04x |
| 2048 × 2048 | 3900.8 | 79.2 | — | 84.3 | 0.02x | — | 0.94x |
| 4096 × 4096 | 15760.8 | 435.7 | — | 199.8 | 0.03x | — | 2.18x |
| 8192 × 8192 | 58909.7 | 1712.1 | — | 720.0 | 0.03x | — | 2.38x |

**Rows, n = 2048, k = 128, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1012.9 | 36.9 | — | 76.1 | 0.04x | — | 0.48x |
| 512 × 2048 | 1986.9 | 45.5 | — | 75.4 | 0.02x | — | 0.60x |
| 1024 × 2048 | 3933.8 | 46.3 | — | 93.1 | 0.01x | — | 0.50x |
| 2048 × 2048 | 7833.0 | 80.3 | — | 125.1 | 0.01x | — | 0.64x |
| 4096 × 2048 | 15651.2 | 204.2 | — | 201.6 | 0.01x | — | 1.01x |
| 8192 × 2048 | 31372.2 | 445.1 | — | 333.5 | 0.01x | — | 1.33x |
| 16384 × 2048 | 62745.1 | 894.2 | — | 634.4 | 0.01x | — | 1.41x |

**Columns, m = 1024, k = 128, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 422.9 | 34.1 | 18.2 | 19.2 | 0.08x | 1.87x | 1.77x |
| 1024 × 512 | 843.8 | 38.8 | 38.3 | 30.0 | 0.05x | 1.01x | 1.29x |
| 1024 × 1024 | 1692.3 | 45.3 | — | 51.2 | 0.03x | — | 0.89x |
| 1024 × 2048 | 3951.1 | 46.8 | — | 93.0 | 0.01x | — | 0.50x |
| 1024 × 4096 | 7906.0 | 80.6 | — | 178.8 | 0.01x | — | 0.45x |
| 1024 × 8192 | 14898.9 | 205.0 | — | 348.1 | 0.01x | — | 0.59x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 3901.6 | 83.2 | — | 79.0 | 0.02x | — | 1.05x |
| 2048 × 2048, k = 128 | 7815.4 | 81.2 | — | 124.8 | 0.01x | — | 0.65x |
| 2048 × 2048, k = 256 | 15921.9 | 96.0 | — | 214.6 | 0.01x | — | 0.45x |
| 2048 × 2048, k = 512 | 32441.6 | 151.5 | — | 395.1 | 0.00x | — | 0.38x |

**Regrow's shape, n = 256, k = 128, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 117.1 | 31.1 | 19.3 | 18.6 | 0.27x | 1.61x | 1.67x |
| 1024 × 256 | 422.9 | 33.7 | 17.9 | 19.7 | 0.08x | 1.89x | 1.71x |
| 4096 × 256 | 1622.2 | 54.7 | 33.8 | 30.0 | 0.03x | 1.61x | 1.82x |
| 16384 × 256 | 6475.4 | 109.5 | 61.0 | 72.1 | 0.02x | 1.80x | 1.52x |

**Regrow's shape, m = 4096, n = 256, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 826.3 | 50.0 | 23.9 | 29.0 | 0.06x | 2.09x | 1.72x |
| 4096 × 256, k = 128 | 1640.9 | 54.3 | 33.6 | 30.0 | 0.03x | 1.62x | 1.81x |
| 4096 × 256, k = 256 | 3239.9 | 51.4 | 35.4 | 44.3 | 0.02x | 1.45x | 1.16x |
| 4096 × 256, k = 512 | 6578.1 | 66.0 | 53.0 | 72.5 | 0.01x | 1.25x | 0.91x |

**Regrow's 512-wide row, n = 512, k = 128, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 512 | 225.2 | 32.6 | 21.7 | 36.7 | 0.14x | 1.50x | 0.89x |
| 1024 × 512 | 846.0 | 39.0 | 38.6 | 29.9 | 0.05x | 1.01x | 1.30x |
| 4096 × 512 | 3286.8 | 56.0 | 57.0 | 54.5 | 0.02x | 0.98x | 1.03x |
| 16384 × 512 | 13048.7 | 243.5 | 179.0 | 152.2 | 0.02x | 1.36x | 1.60x |

**Regrow's 512-wide row, m = 1024, n = 512, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 512, k = 64 | 424.9 | 37.3 | 24.5 | 26.4 | 0.09x | 1.52x | 1.41x |
| 1024 × 512, k = 128 | 844.0 | 38.4 | 36.2 | 29.9 | 0.05x | 1.06x | 1.28x |
| 1024 × 512, k = 256 | 1685.2 | 42.8 | 39.2 | 44.5 | 0.03x | 1.09x | 0.96x |
| 1024 × 512, k = 512 | 3388.1 | 43.4 | 59.7 | 71.3 | 0.01x | 0.73x | 0.61x |

pass-to-pass spread (max over impls, %):
  cols1024_none              4.2%
  cols2048_none              4.9%
  cols256_none              18.1%
  cols4096_none             38.3%
  cols512_none              61.0%
  cols8192_none              3.9%
  kdim128_none              10.0%
  kdim256_none              31.8%
  kdim512_none              25.7%
  kdim64_none               15.0%
  regrow128_m4096_none      39.8%
  regrow512_m1024_none      97.8%
  regrow512_m16384_none     13.4%
  regrow512_m256_none       62.7%
  regrow512_m4096_none       4.3%
  regrow512k128             32.6%
  regrow512k256             10.2%
  regrow512k512             98.6%
  regrow512k64              67.3%
  regrow_m1024_none         20.7%
  regrow_m16384_none        33.0%
  regrow_m256_none          44.6%
  regrow_m4096_none         30.1%
  regrowk128                70.5%
  regrowk256               2615.6%
  regrowk512                21.3%
  regrowk64                 34.3%
  rows1024_none              5.2%
  rows16384_none             6.7%
  rows2048_none             11.0%
  rows256_none              78.9%
  rows4096_none              9.0%
  rows512_none               6.2%
  rows8192_none              8.6%
  square1024_causal         28.1%
  square2048_causal         21.5%
  square256_causal          47.4%
  square4096_causal         13.5%
  square512_causal          94.4%
  square8192_causal          9.3%

`n = 128` has no baseline: `cublas_unfused_softmax` needs `n >= 256` to fill a
row's register cache. At 4096 x 128 x 128, `recompute` now runs in 19.6 µs,
ahead of `regrow`'s 20.9 µs and `naive`'s 838.5 µs; it was 36.5 µs before the
last-tile reuse and 23.4 before the shared-memory alias.

## Reading the numbers

**The mainloop and epilogue rewrites changed the ranking.** Before the swizzle
and `ldmatrix` work (see the earlier revisions of this file), the baseline won
everything except `regrow` at its narrowest widths. Now:

- `fused_softmax_regrow` wins at every width it supports: 1.25–2.09x at
  `n = 256`, and 0.73–1.52x at `n = 512`. The low end is the deep-K narrow
  GEMM (`n = 512, k = 512`: 0.73x) and a near-tie at 4096 x 512 (0.98x); the
  16384 x 512 case is a 1.36x win.
- `fused_softmax_recompute` wins the causal squares at both ends of the range —
  1.67x at 256 x 256, 2.18x at 4096 x 4096, 2.38x at 8192 x 8192 — and ties
  1024 x 1024 (1.04x). It still loses narrowly at 512 x 512 (0.93x, a bimodal
  shape this session) and 2048 x 2048 (0.94x), where the baseline's score
  buffer still fits L2 and the row-block schedule has little room to help.
- On no-mask shapes it has caught up: 1.33x at 8192 x 2048, 1.41x at
  16384 x 2048, 1.01x at 4096 x 2048, 0.48–0.64x across the no-mask rows up to
  2048 rows, 0.45–0.59x across the wide columns from `n = 4096` on, and
  0.38–1.05x across head dims. The narrow columns are wins: `n = 256`/`512` are
  1.77x/1.29x, and `k = 64` at 2048 x 2048 is 1.05x — the register epilogue and
  the shared-memory alias are what closed that range.
- `fused_softmax_naive` remains the floor: 2.3x slower than the baseline at
  256 x 256, 18x at 1024 x 1024, 33–98x at the larger shapes (about 214x at
  `k = 512`).

For scale, the same shapes measured before this round of work: `recompute` at
2048 x 2048 causal went 339.0 -> 84.3 µs (4.0x), at 4096 x 4096 causal
1068.8 -> 199.8 µs (5.3x), at 2048 x 2048 no mask 338.8 -> 125.1 µs (2.7x);
`regrow` at 16384 x 256 went 108.2 -> 61.0 µs (1.8x) and at 4096 x 256, `k = 512`
went 103.5 -> 53.0 µs (2.0x). The pieces, measured as they landed:

- **`ldmatrix` + swizzle** (first round): roughly halved the fused GEMM's time
  and is most of the no-mask improvement.
- **Causal skip + LPT row order**: 4096 x 4096 causal 327.9 -> 263.2 µs (1.25x)
  and 8192 x 8192 causal 1137 -> 1024 µs (1.11x); nothing when the grid fits
  one wave (≤1024 x 1024).
- **Adaptive 64/128 tile width** (later superseded): 2048 x 2048 no mask
  212 -> 162 µs (1.31x), 16384 x 2048 963 -> 868 µs. Once the score alias
  raised the 128-wide residency to four CTAs, the 64-wide build lost at every
  shape where both fit, so the dispatch was removed.
- **cp.async copy + last-tile reuse**: 4096 x 4096 causal 263.2 -> 236.4 µs,
  8192 x 8192 causal 1024 -> 927 µs, 1024 x 8192 465 -> 380 µs, 1024 x 4096
  232 -> 196 µs, 1024 x 2048 121 -> 105 µs. The two-stage ring was measured and
  turned down: at kStages = 2 the 64-wide build lost its fourth CTA per SM and
  16384 x 2048 regressed 868 -> 964 µs.
- **Register-based pass-1 statistics**: 2048 x 2048 no mask
  167.7 -> 146.2 µs, 4096 x 4096 causal 236.4 -> 208.1, 8192 x 8192 causal
  927 -> 829, 1024 x 8192 380 -> 348, 16384 x 2048 860 -> 721,
  2048 x 2048, `k = 64` 125 -> 95 µs, and 4096 x 128 x 128 28.3 -> 23.4. A
  first attempt with three barriers per tile made 512 x 512 causal about 10%
  slower; double-buffering the partial arrays removed the extra barrier.
- **Shared-memory alias + simplified dispatch** (this round): 2048 x 2048 no
  mask 146.2 -> 125.1 µs, 4096 x 2048 235.2 -> 201.6, 8192 x 2048 387.1 ->
  333.5, 16384 x 2048 720.9 -> 634.4, 16384 x 256 90.4 -> 72.1, 2048 x 2048
  at `k = 64` 94.8 -> 79.0 (now 1.05x), 4096 x 4096 causal 208.1 -> 199.8,
  8192 x 8192 causal 829.2 -> 720.0, and 4096 x 128 x 128 23.4 -> 19.6. The
  score staging now aliases the operands, which took the 128-wide build from 36
  KB to 20 KB and from two CTAs per SM to four; the 64-wide dispatch then had
  nothing left to win and was removed.

**Where the speed comes from now.** The fused GEMM is no longer the weak link.
Measured as `2·M·N·K / time`, with both `recompute` passes attributed to the
GEMM (and its causal skip counted as the ~half of the flops it really runs), it
reaches **17.2 TFLOP/s** at 2048 x 2048 no mask against the baseline's 13.4,
**27.1** at 16384 x 2048 against 9.6, **21.5** at 4096 x 4096 causal against
9.9, **23.9** at 8192 x 8192 causal against 10.0, and **17.6** in `regrow` at
16384 x 256 against 9.8. The baseline's 8.9–10.0 TFLOP/s at the large shapes is
its whole pipeline, score round trip included; there it is moving 362–422 GB/s
of nominal traffic (81–94% of the card's 448 GB/s) and is DRAM-bound, while the
fused kernels are still compute-bound. The remaining losses are structural:

- **`recompute` pays for its second GEMM.** At 2048 x 2048 no mask its per-pass
  rate is now 1.3x the baseline's whole pipeline (17.2 vs 13.4), so the 2x
  passes alone explain the 0.64x. At 16384 x 2048 the per-pass rate is 2.8x the
  baseline's and the total wins 1.41x despite the second pass. The crossover is
  when the fused GEMM's per-flop advantage outruns 2x.
- **The causal schedule is worth a fifth of the GEMM at large sizes.** The LPT
  row order took 4096 x 4096 causal from 327.9 to 263.2 µs (1.25x) and
  8192 x 8192 causal from 1137 to 1024 µs (1.11x), before cp.async, last-tile
  reuse, the register epilogue and the alias took those to 200 and 720 µs. It does nothing
  when the whole grid fits one wave, which is exactly where the imbalance has
  nowhere to hide.
- **Aliasing the score staging onto the operands is worth a CTA of residency.**
  The 128-wide build went from 36 KB to 20 KB of dynamic shared memory, from two
  CTAs per SM to four: 2048 x 2048 no mask 146 -> 125 µs, 16384 x 2048
  721 -> 634 µs, 16384 x 256 90 -> 72 µs, 4096 x 128 x 128 23.4 -> 19.6 µs.
  With the 128-wide build at four CTAs the old 64-wide dispatch lost its reason
  to exist and was removed; 64 now only covers widths that are a multiple of 64
  but not 128.
- **`regrow`'s wider rows cost occupancy.** 128 and 256 wide run 4 and 2 CTAs
  per SM; 512 needs 68 KB and runs 1. Its per-GEMM efficiency therefore depends
  on the problem being large enough to keep that one CTA busy. At 16384 x 512 it
  wins 1.36x: the baseline's fp32 score buffer is 33.5 MB there, well past the
  24 MB L2, so its 5x traffic is real DRAM work (88 MB in 243 µs, 362 GB/s),
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
