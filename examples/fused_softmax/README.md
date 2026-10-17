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
file comes from the harness's defaults (causal, 20 timed iterations, 3 warm-up
iterations) except where a table says otherwise.

## The implementations

- **`fused_softmax_regrow` — this repo's single-pass CuTe kernel.** One CTA owns
  32 rows and *all* of a row's 256 columns, so the score matrix never leaves the
  chip: a 2 x 4 grid of SM80 `16x8x16` fp16/fp32 MMA atoms computes `S = Q K^T`
  into registers, `store_scores()` scales and masks it on the way to shared
  memory, and one warp per row reduces with shuffles and stores `P` as
  `__half2`. P is written once and S never, which is the 2·M·N-byte floor. The
  price is that the row has to fit: `n` is fixed at 256 by the 36 KB shared
  memory tile (the GEMM operands and the score staging alias one buffer,
  because the phases never overlap).
- **`fused_softmax_recompute` — this repo's streaming CuTe kernel.** The same
  mainloop, but the columns are streamed in 128-wide tiles and visited twice.
  The first pass folds each tile into the row's running `(max, sum)` with the
  online-softmax update and throws the tile away; the second recomputes it and
  writes `exp(s - max) / sum`. It keeps the 2·M·N floor for any `n` that is a
  multiple of 128 and pays for it with 2x the GEMM multiply-adds instead of
  extra traffic.
- **`fused_softmax_naive` — this repo's correctness floor.** One row per CTA,
  256 threads, plain SIMT multiply-add (no tensor cores), the row's scores
  staged in shared memory and walked three times: max, exponentiate-and-sum,
  normalize-and-store. It is the only implementation that skips masked work —
  a causal row evaluates only its visible columns — so a causal shape costs it
  half. Every other implementation computes every tile and masks in the
  epilogue.
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

**The shared mainloop is not pipelined.** `compute_tile()` copies each 64-wide
K tile into shared memory with a synchronous `UniversalCopy`, barriers, loads
the operands from shared memory with elementwise copies (no swizzle, no
`ldmatrix`), runs `cute::gemm`, barriers, and starts the next tile. That is the
version that is easy to see is correct, and it is where most of the gap to
cuBLAS below comes from — see [Next steps](#next-steps).

**Shape constraints.**

| implementation | constraint |
|---|---|
| `fused_softmax_regrow` | `n = 256`, `m % 32 = 0`, `k % 64 = 0` |
| `fused_softmax_recompute` | `n % 128 = 0`, `m % 32 = 0`, `k % 64 = 0` |
| `fused_softmax_naive` | `k + n <= 12288` (48 KB of shared memory) |
| `cublas_unfused_softmax` | `256 <= n <= 32768`, `n % 4 = 0` |
| harness | causal requires `m <= n`; `m*n`, `m*k`, `n*k` must fit in int32 |

## Results

Each table is the median of **four full passes**, each pass itself the mean of
20 timed iterations after 3 warm-up iterations. A first full pass was run and
discarded as GPU clock warm-up: its cuBLAS samples ran up to 1.9x the later
ones and the other kernels up to ~30% slower, the same idle-clock behaviour
documented in `examples/softmax`. Pass-to-pass spread after warm-up is under
~12% at most shapes; on the shortest shapes (20–80 µs) isolated samples deviate
by 20–50%, which is why the tables report the median of four — this is a shared
desktop GPU.

`—` means the implementation does not accept that shape. The ratio columns are
`cublas time / implementation time`, so **above 1.00x means the in-repo kernel
is faster than the baseline**.

**Square, causal mask, `k = 128` (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 72.8 | 31.6 | 18.1 | 33.2 | 0.43x | 1.75x | 0.95x |
| 512 × 512 | 227.9 | 33.2 | — | 43.3 | 0.15x | — | 0.77x |
| 1024 × 1024 | 861.0 | 44.9 | — | 114.2 | 0.05x | — | 0.39x |
| 2048 × 2048 | 3908.9 | 78.2 | — | 339.0 | 0.02x | — | 0.23x |
| 4096 × 4096 | 15731.3 | 437.1 | — | 1068.8 | 0.03x | — | 0.41x |

**Rows, `n = 2048`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 2048 | 1006.1 | 37.7 | — | 143.6 | 0.04x | — | 0.26x |
| 512 × 2048 | 1976.7 | 46.6 | — | 143.2 | 0.02x | — | 0.33x |
| 1024 × 2048 | 3940.6 | 45.2 | — | 218.8 | 0.01x | — | 0.21x |
| 2048 × 2048 | 7837.5 | 82.2 | — | 349.4 | 0.01x | — | 0.24x |
| 4096 × 2048 | 15627.9 | 203.4 | — | 546.0 | 0.01x | — | 0.37x |
| 8192 × 2048 | 31254.9 | 439.2 | — | 946.6 | 0.01x | — | 0.46x |
| 16384 × 2048 | 62441.8 | 896.7 | — | 1855.6 | 0.01x | — | 0.48x |

**Columns, `m = 1024`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 1024 × 256 | 422.7 | 36.4 | 23.3 | 37.1 | 0.09x | 1.56x | 0.98x |
| 1024 × 512 | 841.2 | 37.9 | — | 62.2 | 0.05x | — | 0.61x |
| 1024 × 1024 | 1677.7 | 44.7 | — | 114.0 | 0.03x | — | 0.39x |
| 1024 × 2048 | 3927.5 | 46.0 | — | 217.9 | 0.01x | — | 0.21x |
| 1024 × 4096 | 7931.5 | 79.0 | — | 425.9 | 0.01x | — | 0.19x |
| 1024 × 8192 | 14880.3 | 203.2 | — | 841.9 | 0.01x | — | 0.24x |

**Head dim, 2048 × 2048, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 2048 × 2048, k = 64 | 3901.4 | 82.0 | — | 211.0 | 0.02x | — | 0.39x |
| 2048 × 2048, k = 128 | 7822.1 | 80.2 | — | 338.8 | 0.01x | — | 0.24x |
| 2048 × 2048, k = 256 | 15844.8 | 96.0 | — | 594.9 | 0.01x | — | 0.16x |
| 2048 × 2048, k = 512 | 32335.4 | 151.4 | — | 1100.5 | 0.00x | — | 0.14x |

**Regrow's home shape, `n = 256`, `k = 128`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 256 × 256 | 116.8 | 31.2 | 18.1 | 40.1 | 0.27x | 1.72x | 0.78x |
| 1024 × 256 | 422.6 | 34.9 | 23.2 | 37.1 | 0.08x | 1.50x | 0.94x |
| 4096 × 256 | 1621.6 | 53.4 | 38.3 | 77.2 | 0.03x | 1.39x | 0.69x |
| 16384 × 256 | 6464.9 | 110.7 | 108.2 | 243.3 | 0.02x | 1.02x | 0.45x |

**Regrow's home shape, `m = 4096`, `n = 256`, no mask (µs):**

| shape | naive | cublas (baseline) | regrow | recompute | naive / cublas | regrow / cublas | recompute / cublas |
|---|---|---|---|---|---|---|---|
| 4096 × 256, k = 64 | 823.3 | 49.3 | 29.8 | 54.5 | 0.06x | 1.66x | 0.91x |
| 4096 × 256, k = 128 | 1633.7 | 53.9 | 38.2 | 77.1 | 0.03x | 1.41x | 0.70x |
| 4096 × 256, k = 256 | 3245.9 | 49.7 | 60.5 | 128.6 | 0.02x | 0.82x | 0.39x |
| 4096 × 256, k = 512 | 6577.2 | 65.3 | 103.5 | 231.9 | 0.01x | 0.63x | 0.28x |

## Reading the numbers

**The baseline wins almost everywhere.** Of the three in-repo kernels, only
`fused_softmax_regrow` ever beats `cublas_unfused_softmax`, and only inside its
home shape: `n = 256` with `k <= 128` and `m <= 4096`, by 1.39–1.75x. Once the
GEMM is the dominant cost (k = 256/512, n >= 512) cuBLAS's tensor-core GEMM
takes over and the fused kernels lose. `fused_softmax_recompute`, which is the
kernel that works at every supported `n`, is still 0.78–0.98x on the small `n = 256` shapes
(`m <= 1024`, `k <= 128`), 0.45–0.69x once `m` or `k` grows, 0.61–0.77x at
`n = 512`, and 0.14–0.41x from `n = 1024` on. `fused_softmax_naive` is 2.3x
slower at 256 x 256, 19x at 1024 x 1024, 36–73x at the larger shapes, and 214x
at `k = 512`.

**Why fusion loses where it should win.** The argument for fusion is traffic.
The fused kernels move P once and read Q and K once — 2·M·N bytes plus
`(m + n)·k·2`, which at the default square is ~12% on top. The baseline's
nominal traffic is 5x that (10·M·N), so at a pure 448 GB/s DRAM rate it should
lose. What actually happens depends on the 24 MB L2:

- At 2048 x 2048 the fp32 score buffer is 16.8 MB and fits L2. The baseline's
  nominal 43 MB in 78 µs works out to 550 GB/s, *above* the card's DRAM peak —
  the second kernel is reading scores back out of L2, and the "unfused" traffic
  penalty largely does not exist. cuBLAS sustains 13.7 TFLOP/s of priced GEMM
  work; `recompute` reaches ~6.3 TFLOP/s per GEMM pass and runs two of them, so
  it loses 4.3x.
- At 4096 x 4096 the buffer is 67 MB and no longer fits. The baseline's nominal
  170 MB in 437 µs is 389 GB/s, ~87% of the DRAM peak — this is the shape where
  removing the buffer should pay the most. cuBLAS drops to 9.8 TFLOP/s under
  that traffic, `recompute`'s per-pass GEMM rate rises to ~8.0 TFLOP/s, and the
  totals land at 1069 µs vs 437 µs: the 2x recompute is now most of the gap.
  A hypothetical single-pass version of the same kernel would land at ~535 µs —
  still behind, but within 1.25x, and that residual is the unpipelined
  mainloop.
- At 16384 x 2048 both settle: cuBLAS at 9.6 TFLOP/s for the whole pipeline,
  `recompute` at ~9.3 TFLOP/s per GEMM pass, so the 2x is again the story
  (1856 µs vs 897 µs, 0.48x). Same at the large square: 0.41x, not because the
  fused GEMM is bad at that size but because it is paid for twice.

The rates above are `2·M·N·K / time`; the per-pass fused numbers attribute
both recompute passes entirely to the GEMM, so they are an upper bound.

**Regrow wins only while the GEMM is small.** At `n = 256` the epilogue is a
large share of the baseline's cost: the score round trip and P write are 5x
P's bytes, the second kernel is another launch, and the GEMM is narrow. At
4096 x 256 x 128 `regrow` beats the baseline by 1.39x, at 1024 x 256 by 1.50x,
and at 256 x 256 by 1.72–1.75x. At `k = 64` it is 1.66x. The window closes as
the GEMM grows: at `m = 16384` both reach ~10 TFLOP/s and the win is 1.02x, and
at `k = 256`/`512` cuBLAS's GEMM quality puts `regrow` at 0.82x/0.63x. Note
that `regrow`'s own scaling is clean: at fixed `m = 4096, n = 256` its time
goes 29.8 -> 38.2 -> 60.5 -> 103.5 µs as k doubles, i.e. it tracks the GEMM
work with a fixed epilogue cost.

**Recompute's penalty is exactly the second GEMM.** At `n = 256` it is 2.0–2.3x
slower than `regrow` (77.1 vs 38.3 µs at 4096 rows; 243.3 vs 108.2 µs at 16384
rows), which is the design trade: online statistics let the row exceed the
chip, and the price is one more pass over every column tile. If the row fits,
`regrow` is the right kernel; if it does not, `recompute` is what keeps the
traffic floor.

**The causal mask buys the fused kernels nothing yet.** Only `naive` skips
masked work, and it benefits exactly as expected: at 2048 x 2048 it takes
3908.9 µs causal against 7837.5 µs unmasked, because half the columns are
masked on average. The two CuTe kernels and cuBLAS compute every tile and mask
in the epilogue, so their causal and unmasked times match within noise
(`recompute` 339.0 vs 349.4 µs; cuBLAS 78.2 vs 82.2 µs). A 32 x 128 tile whose
first column is past the last row's visible column is entirely masked and
could be skipped, cutting up to half the GEMM under a causal mask — see
[Next steps](#next-steps).

**Verification.** Every timed shape passes except 1024 x 8192, where all four
implementations (baseline included) report the same `max_rel_diff` of
1.207e-03 against the harness's 1e-3 threshold. That is the output format, not
the kernels: P is fp16, whose smallest normal is 6.1e-5 and whose subnormal
step is 6.0e-8, so a probability `p` carries a relative quantization error up
to `3e-8 / p`. At `n = 8192` the smallest probabilities are ~2.5e-5 — roughly
`1/n` — which is subnormal and lands at ~1.2e-3. The same effect is visible as
the threshold is approached from below: `max_rel_diff` is 4.9e-4 at
`n <= 2048`, 5.3e-4 at `n = 4096`, and 1.2e-3 at `n = 8192`. Since all four
implementations agree to four digits there, either the threshold should scale
with `n` or the comparison should be against an fp16-rounded reference.

## Caveats

- **Shared desktop GPU.** Passes after the discarded warm-up still show
  occasional 20–50% single-sample spikes on the shortest shapes (the 20–50 µs
  kernels at `n = 256`), from desktop memory bursts and GDDR7 power-state
  changes. The medians are stable; individual samples are not. This is the same
  environment documented in `examples/softmax`.
- **The harness's `GB/s` column is not used above.** It prices traffic as
  `2 * M * N * sizeof(half)`, which is twice the P-write bytes its own comment
  describes ("the write of P and nothing else"); the traffic figures in this
  README are computed independently as 2·M·N bytes for P plus `(m + n)·k·2`
  for Q and K.
- **`TFLOP/s` in the harness** prices every implementation at `2·M·N·K`, which
  is right for cuBLAS, `regrow` and `recompute` but overstates `naive`'s work
  under a causal mask by 2x (it skips masked columns), so its printed rate is
  half its real one.

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

- **Pipeline the shared mainloop.** A `cp.async` multi-stage ring for the A/B
  tiles, a swizzled shared-memory layout, and `SM75_U32x4_LDSM_N` for the
  operand loads are the three missing pieces against a tuned GEMM. Today the
  kernel waits on every tile, and at 4096 x 4096 that costs it ~20% per pass
  against cuBLAS despite cuBLAS carrying 5x the traffic.
- **Skip fully-masked causal tiles.** The mask predicate is already in
  `store_scores()`; hoisting it to a per-tile test before `compute_tile()`
  would halve the GEMM work on square causal shapes for both CuTe kernels.
- **Widen `regrow`.** `kBlockN = 512` doubles the accumulator to 64 floats per
  thread (still short of a spill) but needs 68 KB of shared memory, past the
  48 KB opt-in threshold, and halves the CTAs per SM. Holding the score staging
  in fp16 instead of fp32 would buy back 16 KB.
- **Fuse the epilogue into the online pass.** `recompute` currently spends a
  second GEMM to recover the scores. Storing each tile's exponentiated values
  in shared memory as fp16 while the running statistics are known is not
  possible — the final maximum is not known until the last tile — but a
  two-level scheme (per-tile exponent + per-tile scale folded into the output
  on the second visit) is the standard way to avoid the recompute for moderate
  `n`.
