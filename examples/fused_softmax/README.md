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
  mainloop, but the columns are streamed in 128-wide tiles and visited twice.
  The first pass folds each tile into the row's running `(max, sum)` with the
  online-softmax update and throws the tile away; the second recomputes it and
  writes `exp(s - max) / sum`. It keeps the 2·M·N floor for any `n` that is a
  multiple of 128 and pays for it with 2x the GEMM multiply-adds instead of
  extra traffic. Under a causal mask it now skips every 128-wide tile that lies
  entirely past the last visible column (`j <= row + n - m`) in both passes, and
  writes the zeros those tiles would have produced with a bare store. That is up
  to half the GEMM on a square causal shape, and it is why the causal rows below
  are close to twice the no-mask rows.
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

**The shared mainloop is swizzled and uses `ldmatrix`, but is not pipelined.**
Both kernels stage each 64-wide K tile in a 128-bit-swizzled K-major shared
layout (`Swizzle<3,3,3>` over an 8 x 64-half atom), and load MMA operands with
`SM75_U32x4_LDSM_N` for A and `SM75_U32x2_LDSM_N` for B — x4 for B is a static
error here, because this MMA's 2 x 4 atom layout leaves each lane only 4 halves
of B per k-step. That one change, on top of the elementwise shared-to-register
copies this file recorded before, roughly halved the fused GEMM's time. The
global-to-shared copies are still synchronous `UniversalCopy` barriers; a
`cp.async` ring is the largest remaining piece of GEMM headroom (see
[Next steps](#next-steps)), and the shared-memory cost of a multi-stage ring is
why it is not in yet.

**Shape constraints.**

| implementation | constraint |
|---|---|
| `fused_softmax_regrow` | `n = 128, 256` or `512`; `m % 32 = 0`; `k % 64 = 0` |
| `fused_softmax_recompute` | `n % 128 = 0`; `m % 32 = 0`; `k % 64 = 0` |
| `fused_softmax_naive` | `k + n <= 12288` (48 KB of shared memory) |
| `cublas_unfused_softmax` | `256 <= n <= 32768`; `n % 4 = 0` |
| harness | causal requires `m <= n`; `m*n`, `m*k`, `n*k` must fit in int32 |

## Results

Each table is the median of **four full passes**, each pass itself the mean of
20 timed iterations after 3 warm-up iterations. A first full pass was run and
discarded as GPU clock warm-up. Pass-to-pass spread after warm-up is under ~12%
at most shapes; on the shortest shapes (20–80 µs) isolated samples deviate by
20–50% and occasionally more (one 1024 x 512, k = 256 cuBLAS sample ran at
85 µs against a 42 µs median), which is why the tables report medians — this is
a shared desktop GPU. `—` means the implementation does not accept that shape.
The ratio columns are `cublas time / implementation time`, so **above 1.00x
means the in-repo kernel is faster than the baseline**.

**Square, causal mask, `k = 128` (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 72.5 | 32.0 | 20.1 | 21.4 | 0.44x | 1.59x | 1.49x |
| 512 × 512 | 227.3 | 34.1 | 20.7 | 30.1 | 0.15x | 1.65x | 1.13x |
| 1024 × 1024 | 847.0 | 44.9 | — | 52.7 | 0.05x | — | 0.85x |
| 2048 × 2048 | 3882.6 | 85.1 | — | 112.3 | 0.02x | — | 0.76x |
| 4096 × 4096 | 15695.3 | 429.9 | — | 327.9 | 0.03x | — | 1.31x |

**Rows, `n = 2048`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1005.5 | 38.6 | — | 88.4 | 0.04x | — | 0.44x |
| 512 × 2048 | 1977.5 | 46.4 | — | 88.1 | 0.02x | — | 0.53x |
| 1024 × 2048 | 3932.2 | 47.1 | — | 121.5 | 0.01x | — | 0.39x |
| 2048 × 2048 | 7883.0 | 81.1 | — | 211.7 | 0.01x | — | 0.38x |
| 4096 × 2048 | 15595.8 | 204.9 | — | 315.2 | 0.01x | — | 0.65x |
| 8192 × 2048 | 31228.8 | 440.9 | — | 529.5 | 0.01x | — | 0.83x |
| 16384 × 2048 | 62431.2 | 894.6 | — | 963.1 | 0.01x | — | 0.93x |

**Columns, `m = 1024`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 423.8 | 33.1 | 18.2 | 29.6 | 0.08x | 1.81x | 1.12x |
| 1024 × 512 | 844.8 | 38.3 | 38.0 | 38.2 | 0.05x | 1.01x | 1.00x |
| 1024 × 1024 | 1677.9 | 45.8 | — | 66.5 | 0.03x | — | 0.69x |
| 1024 × 2048 | 3927.8 | 46.1 | — | 122.0 | 0.01x | — | 0.38x |
| 1024 × 4096 | 7885.2 | 81.3 | — | 230.9 | 0.01x | — | 0.35x |
| 1024 × 8192 | 14852.5 | 207.5 | — | 447.9 | 0.01x | — | 0.46x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 3939.8 | 82.3 | — | 141.3 | 0.02x | — | 0.58x |
| 2048 × 2048, k = 128 | 7791.9 | 81.6 | — | 205.8 | 0.01x | — | 0.40x |
| 2048 × 2048, k = 256 | 15855.1 | 101.2 | — | 314.3 | 0.01x | — | 0.32x |
| 2048 × 2048, k = 512 | 32587.1 | 152.4 | — | 530.3 | 0.00x | — | 0.29x |

**Regrow's 256-wide row, `n = 256`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 117.2 | 34.2 | 17.5 | 21.4 | 0.29x | 1.96x | 1.60x |
| 1024 × 256 | 422.3 | 35.9 | 18.0 | 28.9 | 0.09x | 1.99x | 1.24x |
| 4096 × 256 | 1619.2 | 54.9 | 33.5 | 49.5 | 0.03x | 1.64x | 1.11x |
| 16384 × 256 | 6443.1 | 109.4 | 60.8 | 134.2 | 0.02x | 1.80x | 0.81x |

**Regrow's 256-wide row, `m = 4096`, `n = 256`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 823.1 | 49.8 | 22.7 | 35.8 | 0.06x | 2.19x | 1.39x |
| 4096 × 256, k = 128 | 1618.8 | 55.6 | 33.7 | 49.8 | 0.03x | 1.65x | 1.12x |
| 4096 × 256, k = 256 | 3232.1 | 49.1 | 35.2 | 70.8 | 0.02x | 1.39x | 0.69x |
| 4096 × 256, k = 512 | 6597.8 | 65.4 | 53.0 | 113.3 | 0.01x | 1.23x | 0.58x |

**Regrow's 512-wide row, `n = 512`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 512 | 224.2 | 32.9 | 20.1 | 30.9 | 0.15x | 1.64x | 1.06x |
| 1024 × 512 | 842.0 | 37.3 | 35.5 | 38.2 | 0.04x | 1.05x | 0.98x |
| 4096 × 512 | 3306.6 | 56.4 | 57.0 | 88.0 | 0.02x | 0.99x | 0.64x |
| 16384 × 512 | 13025.5 | 241.4 | 178.0 | 252.5 | 0.02x | 1.36x | 0.96x |

**Regrow's 512-wide row, `m = 1024`, `n = 512`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 512, k = 64 | 425.1 | 38.0 | 23.2 | 29.8 | 0.09x | 1.64x | 1.28x |
| 1024 × 512, k = 128 | 850.8 | 38.8 | 33.0 | 38.5 | 0.05x | 1.17x | 1.01x |
| 1024 × 512, k = 256 | 1685.1 | 50.5 | 40.5 | 56.1 | 0.03x | 1.25x | 0.90x |
| 1024 × 512, k = 512 | 3367.1 | 44.7 | 59.8 | 89.6 | 0.01x | 0.75x | 0.50x |

`n = 128` has no baseline: `cublas_unfused_softmax` needs `n >= 256` to fill a
row's register cache. At 4096 x 128 x 128, `regrow` runs in 19.6 µs against
`recompute`'s 31.9 µs and `naive`'s 830.7 µs.

## Reading the numbers

**The mainloop rewrite changed the ranking.** Before the swizzle and `ldmatrix`
work (see the previous commit's README), the baseline won everything except
`regrow` at its narrowest widths. Now:

- `fused_softmax_regrow` wins at every width it supports: 1.23–2.19x at
  `n = 256`, and 0.75–1.65x at `n = 512`. Its losses in that range are only the
  deep-K, narrow-GEMM cases (`n = 512, k = 512`: 0.75x) and a tie at
  4096 x 512 x 128 (0.99x).
- `fused_softmax_recompute` wins the causal squares it can skip through —
  1.49x at 256 x 256, 1.13x at 512 x 512, 1.31x at 4096 x 4096 — and is near
  parity at 16384 x 2048 (0.93x). It still loses by 2–3x on no-mask wide
  shapes (0.38–0.46 at `n = 2048`–`8192`, `m >= 2048`; 0.29–0.58 across head
  dims at 2048 x 2048), because there the second GEMM is pure cost and the
  baseline's score buffer is small enough to sit in L2.
- `fused_softmax_naive` remains the floor: 2.3x slower than the baseline at
  256 x 256, 19x at 1024 x 1024, and 36–73x at the larger shapes (214x at
  `k = 512`).

For scale, the same shapes measured before the mainloop work: `recompute` at
2048 x 2048 causal went 339.0 -> 112.3 µs (3.0x), at 4096 x 4096 causal
1068.8 -> 327.9 µs (3.3x), at 2048 x 2048 no mask 338.8 -> 211.7 µs (1.6x);
`regrow` at 16384 x 256 went 108.2 -> 60.8 µs (1.8x) and at 4096 x 256, `k = 512`
went 103.5 -> 53.0 µs (1.9x). Causal skipping accounts for roughly 1.8x of the
causal `recompute` gains; the swizzled `ldmatrix` operand path accounts for the
rest, with the larger effect at larger `k` (more operand loads amortized over
the same barriered copy).

**Where the speed comes from now.** The fused GEMM is no longer the weak link.
Measured as `2·M·N·K / time`, with both `recompute` passes attributed to the
GEMM, it reaches **10.1 TFLOP/s** at 2048 x 2048 no mask against the baseline's
13.2, **13.1** at 4096 x 4096 causal against 10.0, **17.7** in `regrow` at
16384 x 256 against 9.8, and **17.8** in `recompute` at 16384 x 2048 against
9.6. The baseline's 9.6–13.2 TFLOP/s is its whole pipeline, score round trip
included; at the large shapes it is moving 365–422 GB/s of nominal traffic
(81–94% of the card's 448 GB/s) and is DRAM-bound, while the fused kernels are
still compute-bound. The remaining losses are therefore structural, not
arithmetic:

- **`recompute` pays for its second GEMM.** At 16384 x 2048 its per-pass rate is
  1.85x the baseline's, yet the total is 0.93x — the second pass is the whole
  gap. At 2048 x 2048 the per-pass rate itself is 1.3x behind (small problem,
  launch and tail overheads, and the baseline's scratch fits in the 24 MB L2),
  so the total is 0.38x.
- **`regrow`'s wider rows cost occupancy.** 128 and 256 wide run 4 and 2 CTAs
  per SM; 512 needs 68 KB and runs 1. Its per-GEMM efficiency therefore depends
  on the problem being large enough to keep that one CTA busy. At 16384 x 512 it
  wins 1.36x: the baseline's fp32 score buffer is 33.5 MB there, well past the
  24 MB L2, so its 5x traffic is real DRAM work (88 MB in 241.4 µs, 365 GB/s),
  while the fused kernel writes P once and stays compute-bound. At 4096 x 512 the
  baseline's buffer (8.4 MB) is back in L2 and the two tie; at `k = 512` the GEMM
  dominates and the extra CTA matters more than the saved traffic.
- **The causal mask still buys `regrow` nothing.** `recompute` skips whole
  masked tiles; `regrow` computes the full row. On 256 x 256 and 512 x 512 the
  skip is most of `recompute`'s 1.49x/1.13x (and `regrow` still wins those
  shapes on the strength of its single pass). A 128-wide N sub-tile inside
  `regrow` would recover the same saving, at the cost of a score buffer that can
  no longer alias the operand tile. See [Next steps](#next-steps).

**Verification.** All timed shapes pass, including 1024 x 8192, which used to
report 1.207e-03 for all four implementations. The fix is in the metric, not the
kernels: P is fp16, whose smallest normal is 6.1e-5 and whose subnormal spacing
is 6.0e-8, so the smallest probabilities at `n = 8192` (~2.5e-5, roughly `1/n`)
carry a format-limited relative error around 1.2e-3. `max_rel_diff()` now floors
its denominator at 2^-14, which judges those elements against the resolution
fp16 has there (about one subnormal ulp at the 1e-3 threshold) and leaves the
comparison unchanged above the normal floor. The same effect was already visible
as the threshold was approached: 4.9e-4 at `n <= 2048`, 5.3e-4 at `n = 4096`, and
1.2e-3 at `n = 8192` before the change.

## Caveats

- **Shared desktop GPU.** Passes after the discarded warm-up still show
  occasional 20–50% single-sample spikes on the 20–80 µs shapes, from desktop
  memory bursts and GDDR7 power-state changes; the 1024 x 512 `k = 256` cuBLAS
  sample quoted above is the worst of this session. The medians are stable;
  individual samples are not. This is the same environment documented in
  `examples/softmax`.
- **`TFLOP/s` in the harness** prices every implementation at `2·M·N·K`, which
  is right for cuBLAS, `regrow` and `recompute` but overstates `naive`'s work
  under a causal mask by 2x (it skips masked columns), so its printed rate is
  half its real one. The `recompute` per-pass rates quoted above halve its
  priced traffic by hand for the same reason: its two passes are attributed to
  the GEMM.
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
host; it is O(m·n·k) and is the slow part of the harness at large shapes.

## Next steps

- **Pipeline the shared mainloop.** A `cp.async` ring is now the largest
  remaining GEMM lever. It costs shared memory: two 64-wide stages take
  `recompute` from 36 KB to 56 KB (2 -> 1 CTA per SM) and `regrow` from 36 to
  72 KB at `n = 256`. The ways to pay for it are a 32-wide K tile (two stages fit
  the current 36 KB, but need a padded or 64-byte-swizzled layout), or
  `regrow`'s row split into 128-wide N sub-tiles (which also enables causal
  skipping and keeps two CTAs). Measure before committing to one.
- **Give `regrow` a causal skip.** Split the row into 128-wide N sub-tiles,
  drop the fully masked ones, and store zeros for them; the score staging can no
  longer alias the operand tile, so this trades a little shared memory for up to
  half the GEMM on square causal shapes. `recompute` already shows the payoff.
- **Fix the softmax phase's bank conflicts.** `softmax_rows()` has each lane
  read `kBlockN / 32` contiguous floats, so a warp's 32-byte-strided lanes collide
  in shared memory (8-way at `n = 256`). Now that the GEMM is fast, this is a
  measurable share at the narrow widths. A swizzled score layout, or an
  interleaved lane mapping plus a shuffle transpose for the `__half2` store,
  addresses it.
- **Widen the shapes.** The kernels still throw on `m % 32 != 0`, `k % 64 != 0`
  and `n % 128 != 0`; attention head dims such as 96 or ragged sequence lengths
  need predication or a tail path. `regrow`'s `n` is a fixed set because the
  whole row must fit; 512 is its ceiling (the B tile alone is 64 KB, the score
  staging another 64 KB).
- **Profile before the next change.** `ncu` on `regrow` at 16384 x 256 and
  `recompute` at 2048 x 2048 would say whether the next bottleneck is the
  synchronous copy, the epilogue, or occupancy; the numbers above only bracket
  it.
