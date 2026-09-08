# softmax

Row-wise softmax, five implementations benchmarked head-to-head on an
**NVIDIA RTX 5060** (Blackwell, sm_120). The two CuTe kernels (`softmax` and
`softmax_multistage`) are implemented in this repo; the other three are
third-party baselines they are measured against.

## The implementations

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
row-wise reference at every shape (max_rel_diff ≤ 5.0e-07, threshold 1e-4).

**time_us** (lower is better):

```
shape           softmax  multistage     cub     fmha    cudnn
8192 x 8192      1696.8      1435.2   2059.3   1712.0   1406.6
8192 x 16384     4074.4      2843.3   4068.4   4089.2   2800.0
8192 x 32768     8193.8      6273.5   8197.9   8216.1   5663.3
16384 x 8192     3284.9      2835.8   4097.2   3393.7   2882.2
16384 x 16384    8221.0      5732.8   8310.9   8181.7   5625.4
16384 x 32768   16403.2     16387.7  16242.4  17035.4  11310.7
32768 x 8192     6533.9      5654.4   8101.9   6808.9   5633.4
32768 x 16384   16409.0     14900.0  16240.3  16778.5  11238.8
32768 x 32768  196840.4    191056.6 199491.6 197034.2 189463.1
```

**GB/s** — every implementation priced at 3 × M × N × 4 bytes per iteration
(two reads of the input, one write of the output), the least a two-pass
softmax can cost:

```
shape           softmax  multistage     cub     fmha    cudnn
8192 x 8192       474.6       561.1    391.1    470.4    572.5
8192 x 16384      395.3       566.5    395.9    393.9    575.2
8192 x 32768      393.1       513.5    392.9    392.1    568.8
16384 x 8192      490.3       568.0    393.1    474.6    558.8
16384 x 16384     391.8       561.9    387.6    393.7    572.6
16384 x 32768     392.8       393.1    396.6    378.2    569.6
32768 x 8192      493.0       569.7    397.6    473.1    571.8
32768 x 16384     392.6       432.4    396.7    384.0    573.2
32768 x 32768       65.5        67.4     64.6     65.4     68.0
```

(`multistage` = `softmax_multistage`.)

## Reading the numbers

**Hardware context.** RTX 5060: 448 GB/s DRAM peak (128-bit GDDR7), 24 MB
L2, 100 KB shared memory per SM, 1536 threads per SM, 8 GB VRAM. `softmax`
uses 2 KB smem per CTA (12 CTAs/SM, 100% occupancy); `softmax_multistage`
uses a 32 KB ring (3 CTAs/SM, 25% occupancy).

**Any GB/s above 448 means the L2 absorbed traffic.** The pricing assumes
3× traffic, but DRAM only moves what misses L2. When the second read of the
two-pass structure stays in L2, real DRAM traffic is 2× (one read, one
write) and the priced number reads as high as ~570. When it leaves L2 the
kernels move the full 3× and the priced number *is* the DRAM rate — the
single-buffer `softmax` at 393–396 GB/s is ~88% of peak, close to the
practical wall for 3× traffic. At 32 KB rows it also catches partial L2
hits (475–493).

**Why `softmax_multistage`'s win is conditional.** Its second read of each
row is served from L2 only while the reuse window fits 24 MB. The sweep
shows the boundary: it holds 561–570 GB/s at every shape with ≤ 2 GB of
device buffers and rows ≤ 64 KB, degrades to 513 at 8192×32768 (2 GB,
128 KB rows), and collapses to 393–432 at the 4 GB shapes — the longer the
row and the larger the footprint, the more traffic churns through L2 between
the first read and the re-read. At that point its 25% occupancy leaves it
little to hide behind, and it lands at or below the single-buffer baseline.
(The split between L2 churn and TLB pressure at 4 GB has not been profiled.)

**cuDNN moves ~2× traffic, not 3×.** Its 559–575 GB/s at every non-paged
shape (rows up to 128 KB, footprints up to 4 GB) is only possible at ≤ ~2.4×
traffic: it reads the input once and rescales the output in place, which
stays L2-resident at these row lengths. It is the reference for what the
*algorithm* buys — ~1.45× over the 3×-traffic two-pass wall.

**The 32768×32768 row is a third regime.** dIn + dOut = 8 GiB exceeds the
card's ~7 GiB usable VRAM, so WDDM pages to system RAM and every kernel —
including cuDNN — runs at PCIe speed (~66 GB/s). Keep
M × N × 4 × 2 ≲ 6 GiB to stay off this cliff.

**Run-to-run variance.** The latency-sensitive kernels (`multistage`,
`fmha`, `cub`) can vary a lot between sessions — e.g. an earlier ad-hoc run
measured `multistage` at 316 GB/s where this sweep records 432, and cuDNN
at 347 where this sweep records 570; the robust ones (`softmax`, and cuDNN
in this sweep) reproduce within a few percent. Treat this matrix as one
coherent session.

## Takeaways

- `softmax_multistage`'s pipeline is a real 1.16–1.43× win — but only while
  its re-reads stay in L2 (≤ 2 GB footprints with rows ≤ 64 KB on this
  card). At 4 GB it is at or below the single-buffer baseline.
- The single-buffer `softmax` is the steady reference: ~88% of DRAM peak at
  3× traffic, at the wall of the two-pass structure.
- The biggest remaining gap is algorithmic: cuDNN's ~2×-traffic one-pass
  structure is worth ~1.45× at every non-paged shape. Neither pipelining
  depth nor TMA can close it — the traffic itself must shrink.

## Build & Run

```bash
cmake -B build && cmake --build build
./build/examples/softmax                                   # 8192 x 8192, 20 iterations
./build/examples/softmax --m=32768 --n=16384 --iterations=8
```
