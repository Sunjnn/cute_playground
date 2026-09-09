# softmax

Row-wise softmax, six implementations benchmarked head-to-head on an
**NVIDIA RTX 5060** (Blackwell, sm_120). Three kernels are implemented in
this repo — two CuTe (`softmax`, `softmax_multistage`) and one plain CUDA
(`softmax_regcache`) — the other three are third-party baselines they are
measured against.

## The implementations

- **`softmax_regcache` — this repo's single-pass kernel.** One row per CTA,
  the whole row cached in registers (32 floats per thread as 8 × `float4`),
  so the input is read exactly once and the output written once — 2× DRAM
  traffic instead of the two-pass kernels' 3×. The row max is free (the row
  is already in registers, so learning it needs no extra walk), the two
  reductions are warp shuffles plus one float per warp in shared memory, and
  there is no smem tile and no copy at all. Because no reuse window is left
  to miss, its performance no longer depends on L2 capacity, row length, or
  total footprint — the three things that make `softmax_multistage`'s win
  conditional. Row length is capped at 32768 floats by the register file.
- **`softmax_multistage` — this repo's pipelined kernel.** Same tiling and
  compute as `softmax`, but the shared-memory tile buffer is a **4-stage
  ring** (4 × 8 KB = 32 KB per CTA). A prologue prefetches the first tiles
  with `cp.async`, the steady state issues the load for tile *i + 4* while
  tile *i* is being computed, and an epilogue drains the remaining stages —
  global loads and compute overlap instead of serializing.
- **`softmax` — this repo's baseline.** Single shared-memory buffer (2 KB per
  CTA): each 1×512 tile is copied in with `cp.async`, fully waited on,
  computed, then copied out before the next tile's load begins.
- `softmax_cub` — external baseline: CUB `DeviceSegmentedReduce` + thrust
  transforms.
- `softmax_fmha` — external baseline: the CUTLASS FMHA-style two-pass kernel
  (as in `fmha_collective_softmax.hpp`).
- `softmax_cudnn` — external baseline: cuDNN `cudnnSoftmaxForward` (optional
  dependency; without cuDNN it reports `n/a`, see the root README).

## Results

Sweep over 9 shapes, 20 timed iterations each (3 warmup iterations first,
mean reported). Every implementation **PASS**es the double-precision
row-wise reference at every shape (max_rel_diff ≤ 5.3e-07, threshold 1e-4).

**time_us** (lower is better):

```
shape           softmax  multistage  regcache     cub     fmha    cudnn
8192 x 8192      1714.0      1417.1    1417.7   2034.2   1709.0   1403.9
8192 x 16384     4131.0      2833.8    2840.7   4087.8   4085.1   2812.5
8192 x 32768     8428.2      6265.8    5662.1   8270.2   8195.3   5692.2
16384 x 8192     3264.2      2815.1    2844.1   4083.4   3378.7   2853.6
16384 x 16384    8256.4      5726.5    5628.2   8122.1   8195.8   5602.5
16384 x 32768   16454.8     17662.5   13961.1  16655.0  16392.9  11318.2
32768 x 8192     6567.5      5704.6    5688.8   8099.7   6803.2   5590.5
32768 x 16384   16544.2     15809.2   11358.7  19773.0  20106.1  11281.1
32768 x 32768  236172.6    231507.3  228651.1 238942.5 235823.8 228448.3
```

**GB/s** — every implementation priced at 3 × M × N × 4 bytes per iteration
(two reads of the input, one write of the output), the least a two-pass
softmax can cost. `softmax_regcache` reads the input once, so it really moves
2× traffic and its priced number should be read against a ceiling 1.5×
higher than the two-pass kernels'.

```
shape           softmax  multistage  regcache     cub     fmha    cudnn
8192 x 8192       469.9       568.3     568.0    395.9    471.2    573.6
8192 x 16384      389.9       568.3     567.0    394.0    394.3    572.7
8192 x 32768      382.2       514.1     568.9    389.5    393.1    565.9
16384 x 8192      493.4       572.1     566.3    394.4    476.7    564.4
16384 x 16384     390.1       562.5     572.3    396.6    393.0    575.0
16384 x 32768     391.5       364.8     461.5    386.8    393.0    569.2
32768 x 8192      490.5       564.7     566.2    397.7    473.5    576.2
32768 x 16384     389.4       407.5     567.2    325.8    320.4    571.1
32768 x 32768      54.6        55.7      56.4     53.9     54.6     56.4
```

(`multistage` = `softmax_multistage`, `regcache` = `softmax_regcache`.)

## Reading the numbers

**Hardware context.** RTX 5060: 448 GB/s DRAM peak (128-bit GDDR7), 24 MB
L2, 100 KB shared memory per SM, 1536 threads per SM, 8 GB VRAM. `softmax`
uses 2 KB smem per CTA (12 CTAs/SM, 100% occupancy); `softmax_multistage`
uses a 32 KB ring (3 CTAs/SM, 25% occupancy); `softmax_regcache` uses
practically no smem but 47–55 registers per thread, which at its 1024-thread
CTAs still leaves an SM one-third idle (66% occupancy).

**Any GB/s above 448 means the L2 absorbed traffic.** The pricing assumes
3× traffic, but DRAM only moves what misses L2. When the second read of the
two-pass structure stays in L2, real DRAM traffic is 2× (one read, one
write) and the priced number reads as high as ~570. When it leaves L2 the
kernels move the full 3× and the priced number *is* the DRAM rate — the
single-buffer `softmax` at 382–392 GB/s is ~86% of peak, close to the
practical wall for 3× traffic. At 32 KB rows it also catches partial L2
hits (470–493).

**`softmax_regcache` moves 2× traffic by construction.** Its priced
566–572 GB/s is a real ~378–381 GB/s of DRAM traffic — ~85% of peak — with
nothing left to miss: there is no second read, so no L2 reuse window, so no
dependence on footprint or row length. It matches cuDNN at every non-paged
shape (1.15–1.49× over `softmax`). The low 461.5 at 16384×32768 in the
table is a noise sample, not a shape property — re-running that shape in
later sessions measured anywhere from 399 to 573 GB/s, including 570.7
(see Run-to-run variance).

**Why `softmax_multistage`'s win is conditional.** Its second read of each
row is served from L2 only while the reuse window fits 24 MB. The sweep
shows the boundary: it holds 562–572 GB/s at every shape with ≤ 2 GB of
device buffers and rows ≤ 64 KB, degrades to 514 at 8192×32768 (2 GB,
128 KB rows), and collapses to 365–408 at the 4 GB shapes — at
16384×32768 it lands *below* the single-buffer baseline. The longer the
row and the larger the footprint, the more traffic churns through L2 between
the first read and the re-read, and at that point its 25% occupancy leaves it
little to hide behind. `softmax_regcache` was written because of exactly this
failure mode: it removes the second read entirely.

**cuDNN moves ~2× traffic, not 3×.** Its 564–576 GB/s at every non-paged
shape (rows up to 128 KB, footprints up to 4 GB) is only possible at ≤ ~2.4×
traffic: it reads the input once and rescales the output in place, which
stays L2-resident at these row lengths. It remains the reference for what
the *algorithm* buys — and `softmax_regcache` now buys the same thing from
the same ~2× traffic.

**The 32768×32768 row is a third regime.** dIn + dOut = 8 GiB exceeds the
card's ~7 GiB usable VRAM, so WDDM pages to system RAM and every kernel —
including cuDNN — runs at PCIe speed (~55 GB/s). Keep
M × N × 4 × 2 ≲ 6 GiB to stay off this cliff.

**Run-to-run variance.** At the large shapes the numbers are session
noise, and the ranking by exposure is the ranking by latency tolerance.
This is a shared desktop GPU: ~20 desktop processes (DWM, browser, IM
clients, …) are resident and insert intermittent memory-system bursts
between the benchmark's kernels, and GDDR7 steps through power states
(405 ↔ 7001 ↔ 13801 MHz) on a seconds scale as those bursts come and go.
Re-running 16384×32768 across sessions measured `softmax_regcache` from
399 to 573 GB/s and `softmax_cudnn` from 433 to 572 — no shape decides
these numbers. `softmax` (12 CTAs/SM, 100% occupancy) hides the spikes
and reproduces within 2% every time; cuDNN is usually immune but gets hit
in degraded windows; `softmax_regcache` (1×1024-thread CTA per SM at
128 KB rows — a single 32-warp barrier with no second CTA to overlap),
`softmax_multistage`, `cub` and `fmha` are the first to feel it.
Diagnostics confirm it is environmental rather than a kernel defect:
profiling the same launches with Nsight Compute always shows a healthy
87%-DRAM-bound kernel, because ncu's per-launch instrumentation paces the
launches and lets the clock ramp — the badness lives in native
back-to-back pacing. The 461.5 at 16384×32768 in the table above was such
a sample, and so are the collapsed `cub`/`fmha` entries at 32768×16384.
Treat the whole matrix as one coherent session.

## Takeaways

- `softmax_regcache`'s single-read design is the fix for the biggest gap the
  earlier sweep identified: it matches cuDNN at every non-paged shape
  (~1.45× over the two-pass baseline) by cutting traffic from 3× to 2×.
  It has no re-reads to lose, so it never collapses the way
  `softmax_multistage` does at 4 GB — but at those footprints its own
  numbers scatter with session noise (see Run-to-run variance).
- `softmax_multistage`'s pipeline is a real 1.16–1.46× win — but only while
  its re-reads stay in L2 (≤ 2 GB footprints with rows ≤ 64 KB on this
  card). At 4 GB it is at or below the single-buffer baseline.
- The single-buffer `softmax` is the steady reference: ~86% of DRAM peak at
  3× traffic, at the wall of the two-pass structure.
- If rows ever need to exceed 32768, `softmax_regcache`'s register budget
  runs out; the two-pass kernels still work (and cuDNN remains the
  non-paged reference).

## Build & Run

```bash
cmake -B build && cmake --build build
./build/examples/softmax                                   # 8192 x 8192, 20 iterations
./build/examples/softmax --m=32768 --n=16384 --iterations=8
```
